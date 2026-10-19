import _bootstrap
import argparse
from pathlib import Path
import json
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi
from community_qwen.data.audit import audit, distribution
from community_qwen.data.split import split_terms
from community_qwen.data.tokenize import encode_example
from community_qwen.model.load import load_tokenizer
from community_qwen.utils.io import write_json, write_jsonl, read_jsonl, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/lora.yaml')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    raw_path=Path('data/raw/hf_snapshot.jsonl')
    if raw_path.exists():
        raw=read_jsonl(raw_path)
        provenance=json.loads(Path('data/raw/provenance.json').read_text())
    else:
        if args.offline: raise FileNotFoundError(raw_path)
        revision=HfApi().dataset_info('MLBtrio/genz-slang-dataset', token=False).sha
        ds=load_dataset('MLBtrio/genz-slang-dataset',revision=revision,split='train',cache_dir='.cache/datasets',token=False)
        raw=[dict(r) for r in ds]
        provenance={'dataset':'MLBtrio/genz-slang-dataset','revision':revision,'fingerprint':ds._fingerprint,'raw_sha256':digest(raw)}
        write_jsonl(raw_path,raw); write_json('data/raw/provenance.json',provenance)
    rows,rejected,report=audit(raw)
    splits=split_terms(rows,config['seed'])
    tokenizer=load_tokenizer(config['model']['local_path'])
    # Length choice uses training only; never tune truncation against final test.
    lengths=[len(encode_example(r,tokenizer,32768,template=i%3)['input_ids']) for i,r in enumerate(splits['train'])]
    max_length=min(512, max(64, ((max(lengths)+31)//32)*32))
    report.update(provenance=provenance,split={s:{'rows':len(rs),'terms':len({r['term_key'] for r in rs})} for s,rs in splits.items()},
                  training_token_lengths=distribution(lengths),max_sequence_length=max_length,
                  training_truncated_rows=sum(n>max_length for n in lengths),
                  tokenizer={'vocabulary_size':tokenizer.vocab_size,'length':len(tokenizer),'special_tokens':tokenizer.special_tokens_map,
                             'eos_id':tokenizer.eos_token_id,'pad_id':tokenizer.pad_token_id},
                  slang_tokenization=[{'term':r['term'],'ids':tokenizer.encode(r['term'],add_special_tokens=False)} for r in splits['train'][:30]])
    for name,rs in splits.items():
        target=Path(f'data/processed/{name}.jsonl')
        if target.exists() and read_jsonl(target)!=rs:
            raise ValueError('Refusing to overwrite a different frozen split; create a new dataset version.')
        write_jsonl(target,rs)
    write_jsonl('data/processed/rejected.jsonl',rejected)
    report['split_sha256']={s:digest(rs) for s,rs in splits.items()}
    write_json('data/processed/manifest.json',report)
    Path('reports/dataset_audit.md').write_text('# Dataset audit\n\nThe Example field supplies usage context. Context metadata is excluded because it frequently states the meaning. Descriptions remain uncorrected weak reference labels, not adjudicated contextual ground truth.\n\n```json\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\n```\n')
    print(json.dumps({k:report[k] for k in ['raw_rows','retained_rows','retained_terms','rejection_counts','split','training_token_lengths','max_sequence_length']},indent=2))

if __name__=='__main__': main()
