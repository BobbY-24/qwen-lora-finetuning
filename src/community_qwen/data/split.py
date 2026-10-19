import random
import re
import unicodedata


def normalize_term(term):
    # Collapse punctuation/spacing variants (e.g. G.O.A.T / goat, high-key / high key).
    return ''.join(c for c in unicodedata.normalize('NFKC', term).casefold() if c.isalnum())


def split_terms(rows, seed=42):
    terms = sorted({r['term_key'] for r in rows})
    if len(terms) < 10:
        raise ValueError('Need at least ten distinct normalized terms for 80/10/10 split')
    random.Random(seed).shuffle(terms)
    n_val = max(1, round(len(terms)*.1))
    sets = {'validation':set(terms[:n_val]), 'test':set(terms[n_val:2*n_val]), 'train':set(terms[2*n_val:])}
    result = {s:[r for r in rows if r['term_key'] in keys] for s,keys in sets.items()}
    assert_disjoint(result)
    return result


def assert_disjoint(splits):
    keys = {s:{r['term_key'] for r in rows} for s,rows in splits.items()}
    for a in keys:
        for b in keys:
            if a < b and keys[a] & keys[b]:
                raise ValueError(f'Term leakage: {a}/{b}')
    # Equal examples may reveal the same answer despite different surface terms.
    contexts = {s:{re.sub(r'\s+',' ',r['context'].casefold()).strip() for r in rows} for s,rows in splits.items()}
    for a in contexts:
        for b in contexts:
            if a < b and contexts[a] & contexts[b]:
                raise ValueError(f'Duplicate usage sentence across {a}/{b}; quarantine before splitting')
