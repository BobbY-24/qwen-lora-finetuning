import math
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(optimizer,total_steps,warmup_ratio,kind='cosine'):
    if total_steps<1 or not 0<=warmup_ratio<1 or kind not in ['cosine','linear']:
        raise ValueError('Invalid scheduler configuration')
    warmup=int(total_steps*warmup_ratio)
    def scale(step):
        if step<warmup:return (step+1)/max(1,warmup)
        progress=min(1,(step-warmup)/max(1,total_steps-warmup))
        return .5*(1+math.cos(math.pi*progress)) if kind=='cosine' else 1-progress
    return LambdaLR(optimizer,scale)
