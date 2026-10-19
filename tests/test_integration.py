"""Real-data contracts and optional hardware integration, with explicit skips."""
from pathlib import Path
import json
import pytest
import torch
import yaml
from datasets import Dataset
from community_qwen.utils.io import read_jsonl,digest
from community_qwen.data.split import assert_disjoint
from community_qwen.model.load import load_model
from community_qwen.model.lora import attach_lora


def test_actual_dataset_splits_and_source_ids():
    splits={s:read_jsonl(f'data/processed/{s}.jsonl') for s in ['train','validation','test']}
    manifest=json.loads(Path('data/processed/manifest.json').read_text())
    assert_disjoint(splits)
    for s,rows in splits.items():
        assert digest(rows)==manifest['split_sha256'][s]
        ds=Dataset.from_list(rows)
        assert len(ds)==manifest['split'][s]['rows']
        assert len({r['id'] for r in rows})==len(rows)


def test_ood_terms_absent_from_every_split():
    training={r['term_key'] for s in ['train','validation','test'] for r in read_jsonl(f'data/processed/{s}.jsonl')}
    ood=read_jsonl('data/ood/technical.jsonl')
    assert not training&{r['term_key'] for r in ood}
    assert len(read_jsonl('data/ood/general.jsonl'))==25


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA; quantized training not validated on this MPS machine')
def test_real_quantized_adapter_update():
    pytest.importorskip('bitsandbytes')
    c=yaml.safe_load(Path('configs/qlora.yaml').read_text());c['model']['device']='cuda'
    m,t,d=load_model(c,quantized=True);m=attach_lora(m,c['lora'],quantized=True)
    ids=torch.tensor([[1,2,3,4]],device=d)
    before={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    loss=m(input_ids=ids,labels=ids).loss
    assert torch.isfinite(loss);loss.backward();opt.step()
    assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if p.requires_grad)
    assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)


def test_auto_discovery_propagates_checkpoint_identity(monkeypatch):
    import importlib
    module=importlib.import_module('community_qwen.model.load')
    path=yaml.safe_load(Path('configs/lora.yaml').read_text())['model']['local_path']
    monkeypatch.setattr(module,'discover',lambda:path)
    # Substitute only expensive weight I/O; this test validates configuration propagation,
    # not checkpoint quality or model loading (covered by actual run evidence).
    monkeypatch.setattr(module.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:torch.nn.Linear(2,2))
    c={'model':{'local_path':'AUTO_DISCOVER','device':'cpu','dtype':'float32'}}
    _,tokenizer,_=module.load_model(c)
    assert c['model']['local_path']==path and tokenizer.eos_token_id==151645


def test_auto_discovery_supports_preparation_and_evaluation_manifests(monkeypatch):
    import importlib
    module=importlib.import_module('community_qwen.model.load')
    path=yaml.safe_load(Path('configs/lora.yaml').read_text())['model']['local_path']
    monkeypatch.setattr(module,'discover',lambda:path)
    assert module.checkpoint_identity('AUTO_DISCOVER')==module.checkpoint_identity(path)
    tokenizer=module.load_tokenizer('AUTO_DISCOVER')
    assert tokenizer.encode('hello',add_special_tokens=False)==module.load_tokenizer(path).encode('hello',add_special_tokens=False)
