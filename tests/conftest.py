from pathlib import Path
import pytest
import torch
import yaml
from transformers import Qwen2Config,Qwen2ForCausalLM
from community_qwen.model.lora import attach_lora

# Tests that load the host-specific Qwen checkpoint from configs/lora.yaml; skipped where it is absent (e.g. CI).
_MODEL_PATH=Path(yaml.safe_load((Path(__file__).parents[1]/'configs/lora.yaml').read_text())['model']['local_path'])
_NEEDS_LOCAL_MODEL={'test_actual_tokenizer_masks_prefix_and_padding','test_truncation_retains_answer_or_errors',
                    'test_padding_can_share_eos_without_masking_real_eos','test_auto_discovery_propagates_checkpoint_identity',
                    'test_auto_discovery_supports_preparation_and_evaluation_manifests','test_real_loop_short_window_early_stop_resume'}

def pytest_collection_modifyitems(config,items):
    if (_MODEL_PATH/'config.json').exists(): return
    skip=pytest.mark.skip(reason='Local Qwen checkpoint not installed (configs/lora.yaml model.local_path)')
    for item in items:
        if item.name in _NEEDS_LOCAL_MODEL: item.add_marker(skip)

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
