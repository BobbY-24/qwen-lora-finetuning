from collections import Counter, defaultdict
import re
import numpy as np
from .split import normalize_term
from community_qwen.utils.io import digest


def distribution(values):
    return {k:float(v) for k,v in zip(['p50','p75','p90','p95','p99','max'], np.percentile(values,[50,75,90,95,99,100]))} if values else {}


def audit(raw):
    required = ['Slang','Description','Example','Context']
    missing = {k:sum(not isinstance(r.get(k),str) or not r[k].strip() for r in raw) for k in required}
    rows, rejected, seen = [], [], set()
    for i,r in enumerate(raw):
        reasons = []
        if any(not isinstance(r.get(k),str) or not r[k].strip() for k in required[:3]):
            reasons.append('missing_required_value')
        if reasons:
            rejected.append({'source_row':i,'reasons':reasons}); continue
        term, definition, context = [r[k].strip() for k in required[:3]]
        key = normalize_term(term)
        if not key:
            reasons.append('empty_normalized_term')
        signature = (key, definition.casefold(), context.casefold())
        if signature in seen:
            reasons.append('duplicate_training_example')
        seen.add(signature)
        if len(definition)<3:
            reasons.append('definition_too_short')
        if len(definition)>15 and definition.casefold() in context.casefold():
            reasons.append('gold_definition_in_example')
        if reasons:
            rejected.append({'source_row':i,'reasons':reasons}); continue
        rows.append({'id':f'hf-{i:05d}-{digest(r)[:10]}','source_row':i,'term':term,'term_key':key,
                     'context':context,'definition':definition,'source_context_metadata':r.get('Context')})
    context_terms=defaultdict(set)
    for r in rows:
        context_terms[re.sub(r'\s+',' ',r['context'].casefold())].add(r['term_key'])
    clean=[]
    for r in rows:
        if len(context_terms[re.sub(r'\s+',' ',r['context'].casefold())])>1:
            rejected.append({'source_row':r['source_row'],'reasons':['sentence_shared_by_different_terms']})
        else:
            clean.append(r)
    count=Counter(normalize_term(r.get('Slang','')) for r in raw if isinstance(r.get('Slang'),str))
    report = dict(raw_rows=len(raw), columns=sorted(set().union(*(r.keys() for r in raw))), missing_values=missing,
                  unique_normalized_terms=len(count), duplicate_term_rows=sum(v-1 for v in count.values()),
                  retained_rows=len(clean), retained_terms=len({r['term_key'] for r in clean}),
                  rejection_counts=dict(Counter(reason for r in rejected for reason in r['reasons'])),
                  character_lengths={k:distribution([len(r[k]) for r in raw if isinstance(r.get(k),str)]) for k in required},
                  duplicate_examples=len(raw)-len({str(r.get('Example')) for r in raw}),
                  source_sha256=digest(raw))
    return clean,rejected,report
