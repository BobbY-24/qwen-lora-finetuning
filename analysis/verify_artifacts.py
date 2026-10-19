from pathlib import Path
import sys
sys.path.insert(0,'src')
import json
from community_qwen.utils.io import read_jsonl,digest
from community_qwen.evaluation.metrics import summarize,score
from community_qwen.data.split import assert_disjoint

splits={s:read_jsonl(f'data/processed/{s}.jsonl') for s in ['train','validation','test']}
manifest=json.loads(Path('data/processed/manifest.json').read_text());assert_disjoint(splits)
for s,rs in splits.items():assert digest(rs)==manifest['split_sha256'][s]
results=[]
for p in Path('experiments').rglob('predictions.jsonl'):
    m=json.loads((p.parent/'manifest.json').read_text());rows=read_jsonl(p)
    assert len(rows)==len({r['id'] for r in rows})
    assert {r['id'] for r in rows}==set(m['ids'])
    assert not set(m['demonstration_ids'])&set(m['ids'])
    expected=summarize(rows);s=json.loads((p.parent/'summary.json').read_text())
    assert s['rows']==len(rows) and s['completed']==len(rows) and s['failed']==0
    for r in rows:assert r['scores']==score(r['prediction'],r['definition'])
    for key,value in expected.items():assert abs(s['metrics'][key]-value)<1e-12 if value is not None else s['metrics'][key] is None
    if m['final_test']:
        assert set(m['ids'])=={r['id'] for r in splits['test']}
        source={r['id']:r for r in splits['test']}
        assert all(r['definition']==source[r['id']]['definition'] and r['context']==source[r['id']]['context'] for r in rows)
    results.append({'path':str(p),'rows':len(rows),'ids_unique':True,'metrics_recomputed':True,'complete':True})
Path('reports/artifact_verification.json').write_text(json.dumps({'splits_disjoint':True,'unchanged_split_hashes':True,'evaluations':results},indent=2)+'\n')
print(json.dumps({'verified_evaluations':len(results),'total_predictions':sum(r['rows'] for r in results),'split_integrity':True},indent=2))
