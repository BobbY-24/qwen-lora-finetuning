"""Generate one-factor configs; execution is explicit and serial."""
import _bootstrap
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import yaml

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/lora.yaml');p.add_argument('--execute',action='store_true')
    p.add_argument('--factor',choices=['rank','data','context','targets','optimizer'],required=True);a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    choices={'rank':[4,8,16],'data':[.1,.25,.5,1.],'context':[False,True],
             'targets':[['q_proj','v_proj'],['q_proj','k_proj','v_proj','o_proj']],'optimizer':['adam','adamw']}[a.factor]
    for i,v in enumerate(choices):
        variant=copy.deepcopy(c)
        if a.factor=='rank':variant['lora']['rank']=v
        elif a.factor=='targets':variant['lora']['target_modules']=v
        else:variant['training'][{'data':'data_fraction'}.get(a.factor,a.factor)]=v
        name=f'{a.factor}_{i}'
        path=Path('configs/ablations')/f'{name}.yaml';path.write_text(yaml.safe_dump(variant,sort_keys=False))
        command=[sys.executable,'scripts/train_lora.py','--config',str(path),'--output',f'experiments/ablations/{name}']
        print(' '.join(command))
        if a.execute:subprocess.run(command,check=True)
