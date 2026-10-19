from .prompts import messages


def prompt_ids(tokenizer, row, template=0, context=True, demos=()):
    return tokenizer.apply_chat_template(messages(row,template,context,demos), tokenize=True, return_dict=False,
                                          add_generation_prompt=True, enable_thinking=False)


def encode_example(row, tokenizer, max_length, template=0, context=True, demos=()):
    chat = messages(row,template,context,demos)
    prefix = prompt_ids(tokenizer,row,template,context,demos)
    full = tokenizer.apply_chat_template(chat+[{'role':'assistant','content':row['definition']}],
                                         tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prefix is not token aligned; assistant boundary must be audited.')
    if len(prefix) >= max_length-1:
        raise ValueError('Prompt consumes sequence budget; refusing an all-masked/one-token target.')
    ids = full[:max_length]
    labels = [-100]*len(prefix) + ids[len(prefix):]
    return {'input_ids':ids, 'attention_mask':[1]*len(ids), 'labels':labels}
