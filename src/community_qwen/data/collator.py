import torch


class AssistantCollator:
    def __init__(self,pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self,examples):
        length=max(len(e['input_ids']) for e in examples)
        result={k:[] for k in ['input_ids','attention_mask','labels']}
        for e in examples:
            padding=length-len(e['input_ids'])
            for k,pad in [('input_ids',self.pad_token_id),('attention_mask',0),('labels',-100)]:
                result[k].append(e[k]+[pad]*padding)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in result.items()}
