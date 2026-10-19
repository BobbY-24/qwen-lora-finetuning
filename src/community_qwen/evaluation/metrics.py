"""Transparent lexical proxies, never a claim of semantic correctness."""
import re
from collections import Counter
import numpy as np


def words(text):
    return re.findall(r'\w+',text.casefold())


def rouge_l(prediction,reference):
    a,b=words(prediction),words(reference)
    if not a or not b:return 0.
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b):
            current.append(previous[j]+1 if x==y else max(current[-1],previous[j+1]))
        previous=current
    return 2*previous[-1]/(len(a)+len(b))


STOP=set('a an the is are was were be being been to of and or in on for with as it this that someone something used means mean when you your they their by from very'.split())


def score(prediction,reference):
    tokens=words(prediction); gold=set(words(reference))-STOP
    trigrams=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return {'rouge_l':rouge_l(prediction,reference),
            'reference_content_word_coverage':len(gold&set(tokens))/len(gold) if gold else None,
            'empty':int(not prediction.strip()),'response_words':len(tokens),
            'repeated_trigram_fraction':1-len(set(trigrams))/len(trigrams) if trigrams else 0.}


def summarize(rows):
    if not rows:raise ValueError('Empty evaluation')
    return {k:float(np.mean([r['scores'][k] for r in rows if r['scores'][k] is not None]))
            if any(r['scores'][k] is not None for r in rows) else None for k in rows[0]['scores']}
