import torch
from .mlp import demonstrate


def demonstrate_tensors():
    B,T,D,H,V=2,3,8,2,16
    ids=torch.arange(B*T).reshape(B,T)
    embedding=torch.nn.Embedding(V,D)
    hidden=embedding(ids)
    heads=hidden.reshape(B,T,H,D//H).transpose(1,2)
    logits=hidden@torch.randn(D,V)
    result={'B':list(torch.zeros(B).shape),'BT':list(ids.shape),'BTD':list(hidden.shape),
            'BHTd':list(heads.shape),'BTV':list(logits.shape),'index':ids[0,1].item(),
            'broadcast_shape':list((hidden+torch.ones(D)).shape),'dtype':str(ids.to(torch.float32).dtype),
            'device':str(ids.to('cpu').device),'mlp':demonstrate()}
    return result
