import math
import torch


def target_count(batch):
    # Causal LM shifts labels by one internally; position zero is never predicted.
    return int((batch['labels'][:,1:]!=-100).sum().item())


@torch.no_grad()
def validate(model,loader,device):
    was_training=model.training
    model.eval(); total,n=0.,0
    try:
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            count=target_count(batch)
            if not count:raise ValueError('No supervised tokens')
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
            total+=float(loss)*count;n+=count
    finally:
        model.train(was_training)
    if not n:raise ValueError('Empty validation set')
    ce=total/n
    return {'loss':ce,'perplexity':math.exp(ce) if ce<700 else None,'target_tokens':n}
