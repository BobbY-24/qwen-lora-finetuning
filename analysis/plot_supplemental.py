from pathlib import Path
import sys
sys.path.insert(0, "src")
from community_qwen.utils.io import read_jsonl
from community_qwen.evaluation.compare import compare
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['one_epoch_rank8','one_epoch_rank4','one_epoch_quarter','one_epoch_seed123']
rows={}
for name in names:
    p=Path('experiments/ablations')/name
    train=json.loads((p/'summary.json').read_text());evaluation=json.loads((p/'test/summary.json').read_text())
    rows[name]={'training':train,'evaluation':evaluation}
for filename,xkey,chosen,xlabel in [
    ('data_scaling','train_rows',['one_epoch_quarter','one_epoch_rank8'],'Training examples (one epoch; update count varies)'),
    ('rank_scaling','trainable_parameters',['one_epoch_rank4','one_epoch_rank8'],'Trainable adapter parameters (one epoch, full data)')]:
    values=sorted([(rows[n]['training'][xkey],rows[n]['evaluation']['metrics']['rouge_l'],n) for n in chosen])
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([v[0] for v in values],[v[1] for v in values],'o-')
    for x,y,n in values:ax.annotate(n.replace('one_epoch_',''),(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(xlabel=xlabel,ylabel='Test ROUGE-L (row mean)',title='Predeclared supplemental experiment; lexical proxy')
    ax.margins(x=.2);ax.set_ylim(0,1);fig.tight_layout();fig.savefig(f'reports/{filename}.png',dpi=150);plt.close(fig)
scores=[rows[n]['evaluation']['metrics']['rouge_l'] for n in ['one_epoch_rank8','one_epoch_seed123']]
stats={'training_seeds':[42,123],'split_seed':42,'epochs':1,'mean_test_rouge_l':float(np.mean(scores)),
       'sample_std_test_rouge_l':float(np.std(scores,ddof=1)),
       'limitation':'Two seeds only; no reliable seed-distribution confidence interval. Main three-epoch run has one seed.'}
contrasts={}
for label,left,right in [
    ('rank4_minus_rank8_one_epoch','experiments/ablations/one_epoch_rank8/test','experiments/ablations/one_epoch_rank4/test'),
    ('quarter_minus_full_one_epoch','experiments/ablations/one_epoch_rank8/test','experiments/ablations/one_epoch_quarter/test'),
    ('base_term_only_minus_sentence','experiments/baseline/test','experiments/ablations/base_term_only'),
    ('lora_term_only_minus_sentence','experiments/lora/run_001/test','experiments/ablations/lora_term_only')]:
    contrasts[label]=compare(read_jsonl(Path(left)/'predictions.jsonl'),read_jsonl(Path(right)/'predictions.jsonl'))
Path('reports/supplemental_experiments.json').write_text(json.dumps({'runs':rows,'seed_statistics':stats,'paired_term_contrasts':contrasts},indent=2)+'\n')
text='\n## Predeclared training ablations and repeatability\n\n| One-epoch condition | Rows | Trainable params | Updates | Test ROUGE-L |\n|---|---:|---:|---:|---:|\n'
for n,r in rows.items():
 t,e=r['training'],r['evaluation'];text+=f'| {n} | {t["train_rows"]} | {t["trainable_parameters"]} | {t["global_step"]} | {e["metrics"]["rouge_l"]:.4f} |\n'
text+='\nThe term-only inference condition changes wording as well as removing the sentence, so it does not cleanly isolate context.\n\nRank comparison uses matched full data and epochs; alpha remains fixed. Data comparison keeps one epoch, so exposure and schedule change with sample size. The main selected three-epoch model is excluded from these curves.\n\n```json\n'+json.dumps(stats,indent=2)+'\n```\n'
text+='\nPaired term contrasts (right condition minus left as named; exploratory, unadjusted):\n\n```json\n'+json.dumps(contrasts,indent=2)+'\n```\n'
p=Path('reports/ablations.md');p.write_text(p.read_text().split('## Predeclared training ablations and repeatability')[0].rstrip()+'\n'+text)
