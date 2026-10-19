"""Descriptive error strata, separate from paired term-bootstrap inference."""
from pathlib import Path
import json,sys
sys.path.insert(0,'src')
from community_qwen.utils.io import read_jsonl,write_json
from community_qwen.evaluation.metrics import words,summarize
from community_qwen.evaluation.compare import compare
base=read_jsonl('experiments/baseline/test/predictions.jsonl')
tuned=read_jsonl('experiments/lora/run_001/test/predictions.jsonl')
compare(base,tuned)
rows={r['id']:r for r in base};other={r['id']:r for r in tuned}
strata={
    'uppercase_terms':[k for k,r in rows.items() if r['term'].isupper()],
    'other_terms':[k for k,r in rows.items() if not r['term'].isupper()],
    'reference_at_most_3_words':[k for k,r in rows.items() if len(words(r['definition']))<=3],
    'reference_more_than_3_words':[k for k,r in rows.items() if len(words(r['definition']))>3],
}
results={}
for label,ids in strata.items():
    if not ids:continue
    results[label]={'rows':len(ids),'terms':len({rows[k]['term_key'] for k in ids}),
                    'base':summarize([rows[k] for k in ids]),'lora':summarize([other[k] for k in ids])}
write_json('reports/stratified_results.json',results)
text='# Descriptive result strata\n\nThese are row-level summaries on the frozen population, not independent hypothesis tests. A term may occur in multiple reference-length groups.\n\n| Stratum | Rows | Terms | Base ROUGE-L | LoRA ROUGE-L | Base words | LoRA words |\n|---|---:|---:|---:|---:|---:|---:|\n'
for label,r in results.items():
    b,t=r['base'],r['lora'];text+=f'| {label} | {r["rows"]} | {r["terms"]} | {b["rouge_l"]:.4f} | {t["rouge_l"]:.4f} | {b["response_words"]:.1f} | {t["response_words"]:.1f} |\n'
text+='\nReference descriptions are often short acronym expansions. A model learning that response format may gain ROUGE-L through improved brevity as well as content. These observed correlations cannot separate causes. The next decisive step is blinded correctness annotation, including correct low-overlap paraphrases and disputed references.\n'
Path('reports/stratified_results.md').write_text(text)
