import pytest
import torch
from community_qwen.evaluation.metrics import rouge_l,score
from community_qwen.evaluation.compare import compare
from community_qwen.evaluation.statistical import paired_bootstrap
from community_qwen.fundamentals.attention_demo import causal_attention
from community_qwen.fundamentals.mlp import TinyMLP


def test_metrics_hand_calculated():
    assert rouge_l('a b','a c')==.5
    assert rouge_l('','a')==0
    assert score('fast move','a fast move')['reference_content_word_coverage']==1
    assert score('','test')['empty']==1


def test_statistics_and_invalid_joins():
    ci=paired_bootstrap([0,0,0],[1,1,1]);assert ci['ci95']==[1.,1.]
    rows=[{'id':str(i),'term_key':str(i),'scores':{'rouge_l':.5}} for i in range(3)]
    assert compare(rows,rows)['mean_delta']==0
    with pytest.raises(ValueError):compare(rows,rows[:-1])
    with pytest.raises(ValueError):compare(rows,rows+[rows[0]])


def test_causal_attention_and_mlp():
    q=torch.randn(2,2,3,4)
    output,weights=causal_attention(q,q,q)
    assert output.shape==q.shape and (weights.triu(1)==0).all()
    torch.testing.assert_close(weights.sum(-1),torch.ones(2,2,3))
    model=TinyMLP();model(torch.randn(2,4)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_greedy_generation_reproducible(tiny_lora):
    tiny_lora.eval();ids=torch.tensor([[1,2,3]])
    a=tiny_lora.generate(ids,max_new_tokens=4,do_sample=False,pad_token_id=0)
    b=tiny_lora.generate(ids,max_new_tokens=4,do_sample=False,pad_token_id=0)
    assert torch.equal(a,b)
