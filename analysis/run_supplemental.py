from pathlib import Path
import subprocess
import sys
for name in ['one_epoch_rank8','one_epoch_rank4','one_epoch_quarter','one_epoch_seed123']:
    subprocess.run([sys.executable,'scripts/train_lora.py','--config',f'configs/ablations/{name}.yaml',
                    '--output',f'experiments/ablations/{name}'],check=True)
