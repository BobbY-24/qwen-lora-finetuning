import math
import torch


def causal_attention(q,k,v):
    scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    future=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).triu(1)
    weights=scores.masked_fill(future,float('-inf')).softmax(-1)
    return weights@v,weights
