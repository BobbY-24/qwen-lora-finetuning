import copy
import torch
import pytest
from community_qwen.training.optimizer import make_optimizer
from community_qwen.training.scheduler import make_scheduler
from community_qwen.training.validate import validate,target_count
from community_qwen.training.checkpoint import save_checkpoint,load_checkpoint
from community_qwen.training.seed import seed_everything

CFG={'optimizer':'adamw','learning_rate':.01,'weight_decay':.01}


def test_gradients_updates_and_frozen_base(tiny_lora,batch):
    model=tiny_lora
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=make_optimizer(model,CFG)
    loss=model(**batch).loss;assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    opt.step()
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if not p.requires_grad)


def test_accumulation_matches_token_weighted_batch(tiny_lora,batch):
    a=tiny_lora;b=copy.deepcopy(a)
    a(**batch).loss.backward()
    count=target_count(batch)
    for i in range(2):
        micro={k:v[i:i+1] for k,v in batch.items()}
        (b(**micro).loss*target_count(micro)/count).backward()
    for x,y in zip(a.parameters(),b.parameters()):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-4)


def test_validation_no_grad_and_restores_mode(tiny_lora,batch):
    tiny_lora.train();result=validate(tiny_lora,[batch],torch.device('cpu'))
    assert result['target_tokens']==5 and result['perplexity']>0 and tiny_lora.training
    assert all(p.grad is None for p in tiny_lora.parameters())


def test_checkpoint_resume_next_update_identical(tiny_lora,batch,tmp_path):
    model=tiny_lora;opt=make_optimizer(model,CFG);sched=make_scheduler(opt,4,0.)
    def step(m,o,s):
        o.zero_grad();m(**batch).loss.backward();o.step();s.step()
    step(model,opt,sched)
    saved={n:p.detach().clone() for n,p in model.named_parameters()}
    save_checkpoint(tmp_path/'checkpoint',model,opt,sched,{'global_step':1},'fingerprint')
    expected_random=torch.rand(3)
    step(model,opt,sched)
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    state=load_checkpoint(tmp_path/'checkpoint',model,opt,sched,'fingerprint')
    assert state['global_step']==1
    torch.testing.assert_close(torch.rand(3),expected_random)
    for n,p in model.named_parameters():torch.testing.assert_close(p,saved[n])
    step(model,opt,sched)
    for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n])
    assert sched.get_last_lr()[0]<CFG['learning_rate']
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'checkpoint',model,opt,sched,'wrong')


def test_qlora_rejects_mps():
    from community_qwen.model.qlora import quantization_config
    with pytest.raises(RuntimeError,match='CUDA'):quantization_config({},torch.device('mps'),torch.bfloat16)


def test_causal_loss_is_shifted_cross_entropy(tiny_lora,batch):
    from torch.nn import functional as F
    outputs=tiny_lora(**batch)
    manual=F.cross_entropy(outputs.logits[:,:-1,:].reshape(-1,32),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(outputs.loss,manual)
