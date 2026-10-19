from .lora import accounting
from .load import checkpoint_identity


def inspect_model(model,tokenizer,path):
    c=model.config
    return dict(checkpoint=checkpoint_identity(path), model_class=type(model).__name__, **accounting(model),
        layers=c.num_hidden_layers, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
        attention_heads=c.num_attention_heads, kv_heads=c.num_key_value_heads,
        head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,
        vocabulary_size=c.vocab_size,max_context_length=c.max_position_embeddings,
        dtype=str(next(model.parameters()).dtype),device=str(next(model.parameters()).device),
        attention_implementation=c._attn_implementation,
        modules=[{'name':n,'class':type(m).__name__} for n,m in model.named_modules() if n in ['model.embed_tokens','model.norm','lm_head'] or n.startswith('model.layers.0')],
        tokenizer_class=type(tokenizer).__name__,tokenizer_size=len(tokenizer),special_tokens=tokenizer.special_tokens_map,
        eos_token=tokenizer.eos_token,pad_token=tokenizer.pad_token)
