from pathlib import Path
import subprocess
import sys
for name in ['one_epoch_rank8','one_epoch_rank4','one_epoch_quarter','one_epoch_seed123']:
    subprocess.run([sys.executable,'scripts/evaluate.py','--config',f'configs/ablations/{name}.yaml',
                    '--input','data/processed/test.jsonl','--final-test',
                    '--adapter',f'experiments/ablations/{name}/adapter','--output',f'experiments/ablations/{name}/test'],check=True)
