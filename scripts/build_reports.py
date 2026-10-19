import _bootstrap
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from community_qwen.utils.io import read_jsonl,write_json
from community_qwen.evaluation.compare import compare
from community_qwen.evaluation.regression import summarize_general


def load(path):return json.loads(Path(path).read_text())


def build():
    paths={'Base zero-shot':'experiments/baseline/test','Base few-shot':'experiments/few_shot/test',
           'LoRA':'experiments/lora/run_001/test'}
    actual={name:load(Path(p)/'summary.json') for name,p in paths.items() if (Path(p)/'summary.json').exists()}
    lines=['# Measured comparison','', 'Lexical proxies on frozen unseen-term test examples. Semantic accuracy requires independent review.','',
           '| Condition | Rows | ROUGE-L (row mean) | Reference-word coverage | Assistant PPL |','|---|---:|---:|---:|---:|']
    for name,s in actual.items():
        lines.append(f'| {name} | {s["rows"]} | {s["metrics"]["rouge_l"]:.4f} | {s["metrics"]["reference_content_word_coverage"]:.4f} | {s["assistant_loss"]["perplexity"]:.2f} |')
    comparisons={}
    if 'Base zero-shot' in actual:
        base=read_jsonl(Path(paths['Base zero-shot'])/'predictions.jsonl')
        for name in ['Base few-shot','LoRA']:
            if name in actual:comparisons[name]=compare(base,read_jsonl(Path(paths[name])/'predictions.jsonl'))
    write_json('reports/comparison.json',{'conditions':actual,'paired_term_statistics':comparisons})
    lines+=['','## Paired statistics','', '```json',json.dumps(comparisons,indent=2),'```','',
            'Intervals condition on the trained adapter and capture term resampling, not seed variation. P-values are exploratory and unadjusted. QLoRA was not run on MPS. No semantic or confirmatory significance claim is made.']
    Path('reports/comparison.md').write_text('\n'.join(lines)+'\n')
    if actual:
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(actual.keys(),[s['metrics']['rouge_l'] for s in actual.values()])
        ax.set(ylabel='Mean ROUGE-L token F1',title='Frozen unseen-term test: lexical overlap proxy',ylim=(0,1))
        fig.tight_layout();fig.savefig('reports/base_vs_finetuned.png',dpi=150);plt.close(fig)
    ood={};general={}
    for name,prefix in [('Base','experiments/baseline'),('LoRA','experiments/lora/run_001')]:
        p=Path(prefix)/'ood'
        if (p/'summary.json').exists():ood[name]=load(p/'summary.json')
        p=Path(prefix)/'general'
        if (p/'predictions.jsonl').exists():general[name]=summarize_general(read_jsonl(p/'predictions.jsonl'))
    write_json('reports/ood_evaluation.json',ood);write_json('reports/general_regression.json',general)
    Path('reports/ood_evaluation.md').write_text('# OOD diagnostic\n\nAssistant-authored technical-community examples, checked for term overlap; not a human-validated benchmark.\n\n```json\n'+json.dumps(ood,indent=2)+'\n```\n')
    Path('reports/general_regression.md').write_text('# General capability sanity check\n\nStrict output matching on 20 short-answer prompts; five summaries require review. No generated code is executed.\n\n```json\n'+json.dumps(general,indent=2)+'\n```\n')
    ablation={}
    for name,path in [('Base term only','experiments/ablations/base_term_only'),('LoRA term only','experiments/ablations/lora_term_only')]:
        if (Path(path)/'summary.json').exists():ablation[name]=load(Path(path)/'summary.json')
    Path('reports/ablations.md').write_text('# Actual ablations\n\nContext inference ablation holds weights, decoding and rows fixed and removes the usage sentence. No extra context derived from definition metadata is used.\n\n```json\n'+json.dumps(ablation,indent=2)+'\n```\n')
    write_json('reports/ablations.json',ablation)
    # Plot only actual trained conditions. Require multiple distinct x values, never fabricate curves.
    variants=[]
    for path in Path('experiments').rglob('summary.json'):
        s=load(path)
        test=path.parent/'test'/'summary.json'
        if 'trainable_parameters' in s and test.exists():variants.append((path.parent,s,load(test)['metrics']['rouge_l']))
    for filename,xkey,xlabel in [('data_scaling','train_rows','Training examples'),('rank_scaling','trainable_parameters','Trainable parameters')]:
        if len({s[xkey] for _,s,_ in variants})<2:continue
        fig,ax=plt.subplots(figsize=(7,4))
        for p,s,y in variants:ax.scatter(s[xkey],y,label=p.name)
        ax.set(xlabel=xlabel,ylabel='Test ROUGE-L (row mean)',title='Observed configurations; consult controls before inference');ax.legend()
        fig.tight_layout();fig.savefig(f'reports/{filename}.png',dpi=150);plt.close(fig)

if __name__=='__main__':build()
