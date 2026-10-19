import torch
from community_qwen.data.tokenize import prompt_ids


@torch.inference_mode()
def generate(model,tokenizer,row,device,max_new_tokens=64,context=True,demos=()):
    model.eval()
    if 'prompt' in row:
        ids=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True, return_dict=False,add_generation_prompt=True,enable_thinking=False)
    else:
        ids=prompt_ids(tokenizer,row,context=context,demos=demos)
    if len(ids)+max_new_tokens>model.config.max_position_embeddings:
        raise ValueError('Generation exceeds model context window')
    input_ids=torch.tensor([ids],device=device)
    output=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
        do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,use_cache=True)
    generated=output[0,len(ids):].tolist()
    return {'prediction':tokenizer.decode(generated,skip_special_tokens=True).strip(),
            'prompt_tokens':len(ids),'generated_tokens':len(generated),
            'hit_token_limit':len(generated)==max_new_tokens and generated[-1]!=tokenizer.eos_token_id}
