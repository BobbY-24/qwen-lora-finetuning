import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from community_qwen.model.lora import attach_lora

@pytest.fixture
def tiny_lora():
    torch.manual_seed(42)
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,tie_word_embeddings=True)
    model=Qwen2ForCausalLM(config)
    return attach_lora(model,{'rank':2,'alpha':4,'dropout':0.0})

@pytest.fixture
def batch():
    return {'input_ids':torch.tensor([[1,2,3,4],[2,3,4,5]]),'attention_mask':torch.ones(2,4,dtype=torch.long),
            'labels':torch.tensor([[-100,-100,3,4],[-100,3,4,5]])}
