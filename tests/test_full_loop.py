"""Offline mechanics test: real small randomly initialized Qwen, not quality evidence."""
from pathlib import Path
import copy
import importlib
import json
import torch
import yaml
from transformers import Qwen2Config,Qwen2ForCausalLM
from community_qwen.model.load import load_tokenizer


def test_real_loop_short_window_early_stop_resume(monkeypatch,tmp_path):
    module=importlib.import_module('community_qwen.training.train')
    config=yaml.safe_load(Path('configs/lora.yaml').read_text())
    tokenizer=load_tokenizer(config['model']['local_path'])
    def load_small(config,quantized=False):
        m=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer),hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=256,tie_word_embeddings=True))
        return m,tokenizer,torch.device('cpu')
    monkeypatch.setattr(module,'load_model',load_small)
    config['model']['max_sequence_length']=128
    config['lora'].update(rank=2,alpha=4,dropout=0.)
    config['training'].update(epochs=4,learning_rate=0.,weight_decay=0.,micro_batch_size=1,
        gradient_accumulation_steps=8,patience=1,gradient_checkpointing=False,max_seconds=100)
    rows=[{'id':f'test-{i}','term':'test','term_key':'test','context':'This is a test.','definition':'A trial.'} for i in range(2)]
    first=module.train(config,rows,rows,tmp_path/'first')
    assert first['status']=='early_stopping' and first['global_step']==2
    assert first['examples_processed']==4 and first['tokens_processed']>0
    assert (tmp_path/'first/adapter/adapter_model.safetensors').exists()
    resumed=module.train(config,rows,rows,tmp_path/'resumed',resume=tmp_path/'first/checkpoint-1')
    for key in ['global_step','epoch','examples_processed','tokens_processed','best_validation_loss','final_training_loss']:
        assert resumed[key]==first[key]
