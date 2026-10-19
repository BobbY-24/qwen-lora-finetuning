import numpy as np


def paired_bootstrap(a,b,seed=42,repetitions=10000):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2:raise ValueError('Need >=2 matched independent units')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite scores')
    delta=b-a; rng=np.random.default_rng(seed)
    draws=rng.choice(delta,(repetitions,len(delta)),replace=True).mean(axis=1)
    return {'units':len(delta),'mean_delta':float(delta.mean()),'std_delta':float(delta.std(ddof=1)),
            'ci95':[float(x) for x in np.quantile(draws,[.025,.975])],
            'method':'paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty',
            'seed':seed,'repetitions':repetitions}


def paired_permutation(a,b,seed=42,repetitions=10000):
    delta=np.asarray(b,dtype=float)-np.asarray(a,dtype=float)
    if len(delta)<2:raise ValueError('Too few units')
    rng=np.random.default_rng(seed)
    null=(delta*rng.choice([-1,1],(repetitions,len(delta)))).mean(axis=1)
    return float((1+(np.abs(null)>=abs(delta.mean())-1e-15).sum())/(repetitions+1))
