import _bootstrap
import argparse
from pathlib import Path
import json
import yaml
from community_qwen.model.load import load_model
from community_qwen.model.inspect import inspect_model
from community_qwen.utils.environment import environment
from community_qwen.utils.io import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/lora.yaml');a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    m,t,d=load_model(c)
    result=inspect_model(m,t,c['model']['local_path'])
    write_json('reports/model_inspection.json',result)
    write_json('reports/environment.json',environment())
    print(json.dumps(result,indent=2))
