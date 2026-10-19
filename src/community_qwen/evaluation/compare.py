from collections import defaultdict
from .statistical import paired_bootstrap,paired_permutation


def compare(base,tuned,metric='rouge_l'):
    def index(rows):
        result={r['id']:r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate prediction IDs')
        return result
    a,b=index(base),index(tuned)
    if set(a)!=set(b):raise ValueError('Prediction populations differ')
    groups=defaultdict(lambda:[[],[]])
    for key in sorted(a):
        for field in ['term_key','definition','context']:
            if a[key].get(field)!=b[key].get(field):raise ValueError('Paired source content mismatch')
        group=a[key].get('term_key',key)
        groups[group][0].append(a[key]['scores'][metric]);groups[group][1].append(b[key]['scores'][metric])
    av=[sum(v[0])/len(v[0]) for v in groups.values()];bv=[sum(v[1])/len(v[1]) for v in groups.values()]
    return dict(metric=metric,base_term_macro=sum(av)/len(av),tuned_term_macro=sum(bv)/len(bv),
                **paired_bootstrap(av,bv), exploratory_permutation_p=paired_permutation(av,bv))
