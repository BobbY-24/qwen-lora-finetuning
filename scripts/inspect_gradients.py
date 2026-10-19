import _bootstrap
import json
from pathlib import Path
import torch
import yaml
from peft import PeftModel
from community_qwen.model.load import load_model
from community_qwen.model.lora import attach_lora
from community_qwen.data.collator import AssistantCollator
from community_qwen.data.tokenize import encode_example
from community_qwen.training.validate import validate
from community_qwen.utils.io import read_jsonl,write_json

if __name__=='__main__':
    config=yaml.safe_load(Path('configs/lora.yaml').read_text())
    model,t,device=load_model(config);model=attach_lora(model,config['lora'])
    rows=read_jsonl('data/processed/train.jsonl')[:2]
    batch=AssistantCollator(t.pad_token_id)([encode_example(r,t,256) for r in rows])
    batch={k:v.to(device) for k,v in batch.items()}
    selected={n:p for n,p in model.named_parameters() if 'layers.0.self_attn.q_proj' in n}
    before={n:p.detach().clone() for n,p in selected.items()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001)
    model.train();loss=model(**batch).loss;loss.backward()
    report=[{'parameter':n,'trainable':p.requires_grad,'parameter_norm':float(p.detach().float().norm()),
             'gradient_norm':float(p.grad.float().norm()) if p.grad is not None else None} for n,p in selected.items()]
    opt.step()
    for r in report:r['changed_after_step']=not torch.equal(before[r['parameter']],selected[r['parameter']])
    assert all(not r['changed_after_step'] for r in report if not r['trainable'])
    assert any(r['changed_after_step'] for r in report if r['trainable'])
    # LoRA B starts at zero: A may have zero gradient on first update. This is expected.
    output=Path('experiments/gradient_smoke')
    output.mkdir(exist_ok=True);model.save_pretrained(output/'adapter')
    model.eval()
    with torch.no_grad():expected=model(**batch).logits.cpu()
    del model
    import gc;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    base,_,_=load_model(config)
    reloaded=PeftModel.from_pretrained(base,output/'adapter',local_files_only=True)
    reloaded.eval()
    with torch.no_grad():actual=reloaded(**batch).logits.cpu()
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)
    write_json('reports/gradient_inspection.json',{'loss':float(loss.detach()),'parameters':report,'adapter_reload_logits_match':True})
    print(json.dumps(report,indent=2))
