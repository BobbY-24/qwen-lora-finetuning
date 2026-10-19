import _bootstrap
import argparse
from pathlib import Path
import yaml
from community_qwen.training.train import train
from community_qwen.utils.io import read_jsonl


def main(quantized=False):
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/qlora.yaml' if quantized else 'configs/lora.yaml')
    p.add_argument('--output',required=True);p.add_argument('--resume');a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    train(c,read_jsonl('data/processed/train.jsonl'),read_jsonl('data/processed/validation.jsonl'),a.output,a.resume,quantized)

if __name__=='__main__':main()
