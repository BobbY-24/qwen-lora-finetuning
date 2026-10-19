"""Strict short-answer sanity checks. Summaries remain qualitative."""
import json
from collections import defaultdict


def check_general(row):
    if row['category']=='summarization':return None
    a,b=row['prediction'].strip(),row['definition'].strip()
    if row['id']=='general-19':
        try:return json.loads(a)==json.loads(b)
        except ValueError:return False
    # Honor exact output constraints; do not silently forgive prose around an answer.
    return a==b


def summarize_general(rows):
    categories=defaultdict(list)
    for r in rows:
        result=check_general(r)
        if result is not None:categories[r['category']].append(result)
    return {k:{'correct':sum(v),'total':len(v),'strict_accuracy':sum(v)/len(v)} for k,v in categories.items()}
