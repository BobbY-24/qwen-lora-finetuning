"""Run the predeclared final protocol serially after main training finishes."""
import _bootstrap
from pathlib import Path
import json
import subprocess
import sys

if __name__=='__main__':
    summary=json.loads(Path('experiments/lora/run_001/summary.json').read_text())
    if summary['status'] not in ['completed','early_stopping']:raise RuntimeError('Main training must finish before final protocol')
    adapter='experiments/lora/run_001/adapter'
    jobs=[
        ['--input','data/processed/test.jsonl','--final-test','--output','experiments/baseline/test'],
        ['--input','data/processed/test.jsonl','--final-test','--shots','2','--output','experiments/few_shot/test'],
        ['--input','data/processed/test.jsonl','--final-test','--adapter',adapter,'--output','experiments/lora/run_001/test'],
        ['--input','data/processed/test.jsonl','--final-test','--term-only','--output','experiments/ablations/base_term_only'],
        ['--input','data/processed/test.jsonl','--final-test','--term-only','--adapter',adapter,'--output','experiments/ablations/lora_term_only'],
    ]
    for input_name,output_name in [('technical','ood'),('general','general')]:
        for prefix,extra in [('experiments/baseline',[]),('experiments/lora/run_001',['--adapter',adapter])]:
            jobs.append(['--input',f'data/ood/{input_name}.jsonl','--output',f'{prefix}/{output_name}']+extra)
    for job in jobs:subprocess.run([sys.executable,'scripts/evaluate.py']+job,check=True)
    subprocess.run([sys.executable,'scripts/error_analysis.py','--base','experiments/baseline/test/predictions.jsonl',
                    '--tuned','experiments/lora/run_001/test/predictions.jsonl'],check=True)
    subprocess.run([sys.executable,'scripts/build_reports.py'],check=True)
