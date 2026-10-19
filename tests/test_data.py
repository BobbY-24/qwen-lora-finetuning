import pytest
import torch
from community_qwen.data.split import normalize_term,split_terms,assert_disjoint
from community_qwen.data.audit import audit
from community_qwen.data.prompts import messages
from community_qwen.data.collator import AssistantCollator
from community_qwen.data.tokenize import encode_example,prompt_ids
from community_qwen.model.load import load_tokenizer
import yaml
from pathlib import Path


def test_term_normalization_and_leakage():
    assert normalize_term('G.O.A.T')==normalize_term('goat')
    assert normalize_term('high-key')==normalize_term('High key')
    rows=[{'term_key':str(i),'context':f'usage {i}'} for i in range(20)]
    splits=split_terms(rows+rows[:3]);assert_disjoint(splits)
    assert splits==split_terms(rows+rows[:3])
    splits['test'].append(splits['train'][0])
    with pytest.raises(ValueError):assert_disjoint(splits)


def test_audit_rejects_bad_and_duplicate():
    raw=[{'Slang':'test','Description':'a trial','Example':'This test failed.','Context':'answer metadata'}]
    rows,rejected,report=audit(raw+raw+[{'Slang':None}])
    assert len(rows)==1 and len(rejected)==2 and report['missing_values']['Slang']==1


def test_prompts_never_include_target_metadata():
    row={'term':'z','context':'usage z','definition':'SECRET TARGET','source_context_metadata':'LEAK'}
    for i in range(3):
        text=str(messages(row,i))
        assert 'SECRET TARGET' not in text and 'LEAK' not in text and 'usage z' in text


@pytest.fixture(scope='module')
def tokenizer():
    config=yaml.safe_load(Path('configs/lora.yaml').read_text())
    return load_tokenizer(config['model']['local_path'])


def test_actual_tokenizer_masks_prefix_and_padding(tokenizer):
    row={'term':'z','context':'usage z','definition':'A trial or test.'}
    ids=prompt_ids(tokenizer,row)
    e=encode_example(row,tokenizer,256)
    assert e['labels'][:len(ids)]==[-100]*len(ids)
    assert e['labels'][len(ids):]==e['input_ids'][len(ids):]
    assert tokenizer.eos_token_id in e['labels'][len(ids):]
    short=encode_example({**row,'definition':'Test.'},tokenizer,256)
    b=AssistantCollator(tokenizer.pad_token_id)([e,short])
    assert (b['labels'][b['attention_mask']==0]==-100).all()
    assert b['input_ids'].dtype==torch.long
    # EOS/PAD may share IDs; real termination must not be masked just by token identity.
    assert (b['labels']!=-100).sum()>0


def test_truncation_retains_answer_or_errors(tokenizer):
    row={'term':'x','context':'an x','definition':'word '*300}
    n=len(prompt_ids(tokenizer,row))
    e=encode_example(row,tokenizer,n+8)
    assert len(e['input_ids'])==n+8 and sum(x!=-100 for x in e['labels'])==8
    with pytest.raises(ValueError):encode_example(row,tokenizer,n)


def test_padding_can_share_eos_without_masking_real_eos(tokenizer):
    row={'term':'z','context':'usage z','definition':'A trial or test.'}
    long=encode_example(row,tokenizer,256)
    short=encode_example({**row,'definition':'Test.'},tokenizer,256)
    result=AssistantCollator(tokenizer.eos_token_id)([long,short])
    eos=result['input_ids']==tokenizer.eos_token_id
    actual_eos=eos & (result['attention_mask']==1) & (result['labels']!=-100)
    assert actual_eos.sum()==2
    assert (result['labels'][result['attention_mask']==0]==-100).all()
