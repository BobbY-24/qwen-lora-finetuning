import _bootstrap
import argparse
from pathlib import Path
from community_qwen.utils.io import read_jsonl,write_jsonl
from community_qwen.evaluation.compare import compare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--tuned',required=True)
    p.add_argument('--output',default='reports/error_analysis.md');a=p.parse_args()
    base=read_jsonl(a.base);tuned=read_jsonl(a.tuned);compare(base,tuned)
    b={r['id']:r for r in base};t={r['id']:r for r in tuned}
    pairs=[]
    for key in sorted(b):
        pairs.append({'id':key,'term':b[key]['term'],'context':b[key]['context'],'reference':b[key]['definition'],
                      'base':b[key]['prediction'],'tuned':t[key]['prediction'],
                      'rouge_delta':t[key]['scores']['rouge_l']-b[key]['scores']['rouge_l'],
                      'base_human_correctness':None,'tuned_human_correctness':None,'error_categories':[],
                      'pair_category':None,'notes':''})
    write_jsonl('reports/paired_human_review.jsonl',pairs)
    text='# Error analysis: candidates for independent review\n\nHigher lexical overlap is not equivalent to correctness. Categories improvement, regression and both-fail remain unassigned until human review. Wrong sense, context ignored, hallucination, generic explanation, excessive slang style and possible memorization are review tags, not automatic lexical diagnoses.\n'
    for label,examples in [('Largest overlap increases',sorted(pairs,key=lambda r:-r['rouge_delta'])[:5]),
                           ('Largest overlap decreases',sorted(pairs,key=lambda r:r['rouge_delta'])[:5])]:
        text+='\n## '+label+'\n'
        for r in examples:
            text+=f'\n### {r["term"]} ({r["id"]})\n\nUsage: {r["context"]}\n\nReference: {r["reference"]}\n\nBase: {r["base"]}\n\nLoRA: {r["tuned"]}\n\nROUGE-L change: {r["rouge_delta"]:+.4f}. Human judgment: pending.\n'
    Path(a.output).write_text(text)
