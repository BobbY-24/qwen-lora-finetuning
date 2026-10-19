import _bootstrap
import argparse
import json
import time
from pathlib import Path
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader
from community_qwen.model.load import load_model,checkpoint_identity
from community_qwen.data.collator import AssistantCollator
from community_qwen.data.tokenize import encode_example
from community_qwen.training.validate import validate
from community_qwen.training.seed import seed_everything
from community_qwen.evaluation.generate import generate
from community_qwen.evaluation.metrics import score,summarize
from community_qwen.utils.io import read_jsonl,write_jsonl,write_json,digest,code_identity
from community_qwen.utils.environment import environment,memory


def evaluate(config,rows,output,adapter=None,shots=0,context=True,limit=None,final_test=False):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    if limit:rows=rows[:limit]
    train=read_jsonl('data/processed/train.jsonl')
    eval_terms={r.get('term_key') for r in rows}
    demos=[];seen=set()
    if shots not in [0,1,2,3]:raise ValueError('Use 0–3 fixed demonstrations')
    for r in train:
        if len(demos)==shots:break
        if r['term_key'] not in eval_terms and r['term_key'] not in seen:
            demos.append(r);seen.add(r['term_key'])
    if len(demos)!=shots:raise ValueError('Insufficient uncontaminated demonstrations')
    adapter_identity=None
    if adapter:
        import hashlib
        adapter_identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(adapter).glob('*') if p.is_file()}
    fingerprint=digest({'config':config,'inputs':rows,'adapter':adapter_identity,'shots':shots,'context':context,
                        'model':checkpoint_identity(config['model']['local_path']),'demos':demos,'code':code_identity()['source_sha256']})
    manifest={'fingerprint':fingerprint,'condition':'LoRA' if adapter else 'few-shot' if shots else 'zero-shot',
              'context':context,'shots':shots,'demonstration_ids':[r['id'] for r in demos],
              'ids':[r['id'] for r in rows],'final_test':final_test,'adapter':adapter,'adapter_identity':adapter_identity,
              'source_sha256':digest(rows),'model':checkpoint_identity(config['model']['local_path'])}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:
            raise ValueError('Existing evaluation fingerprint differs; use a new output directory')
    else:
        write_json(out/'manifest.json',manifest)
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write_json(out/'environment.json',environment())
    existing=read_jsonl(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else []
    done={r['id']:r for r in existing}
    if len(done)!=len(existing) or not set(done)<={r['id'] for r in rows}:raise ValueError('Invalid checkpoint IDs')
    if len(done)==len(rows) and (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    seed_everything(config['seed']);model,tokenizer,device=load_model(config)
    if adapter:model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False)
    start=time.monotonic()
    for row in rows:
        if row['id'] in done:continue
        # Generation receives no reference definition; join reference only after generation.
        public={k:row[k] for k in ['term','context','prompt'] if k in row}
        result=generate(model,tokenizer,public,device,config['evaluation']['max_new_tokens'],context,demos)
        result.update(row);result['scores']=score(result['prediction'],row['definition'])
        done[row['id']]=result
        write_jsonl(out/'predictions.jsonl',[done[r['id']] for r in rows if r['id'] in done])
        print(f'{out.name}: {len(done)}/{len(rows)}',flush=True)
    predictions=[done[r['id']] for r in rows]
    loss=None
    if 'prompt' not in rows[0]:
        # Reference scoring is a separate teacher-forcing pass after predictions are fixed.
        encoded=[encode_example(r,tokenizer,1024,context=context,demos=demos) for r in rows]
        loss=validate(model,DataLoader(encoded,batch_size=1,collate_fn=AssistantCollator(tokenizer.pad_token_id)),device)
    summary=dict(condition=manifest['condition'],rows=len(rows),unique_terms=len({r.get('term_key',r['id']) for r in rows}),
                 metrics=summarize(predictions),assistant_loss=loss,runtime_seconds=time.monotonic()-start,
                 runtime_scope='current invocation; earlier resumed generation not included',**memory(device),
                 token_limit_rate=sum(r['hit_token_limit'] for r in predictions)/len(predictions),
                 completed=len(done),failed=0,semantic_accuracy=None,semantic_review='requires independent human annotation')
    write_json(out/'summary.json',summary)
    write_jsonl(out/'human_review.jsonl',[{**r,'human_correctness':None,'error_categories':[],'review_notes':''} for r in predictions])
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baseline.yaml');p.add_argument('--input',default='data/processed/validation.jsonl')
    p.add_argument('--output',required=True);p.add_argument('--adapter');p.add_argument('--shots',type=int,default=0)
    p.add_argument('--term-only',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--final-test',action='store_true')
    a=p.parse_args()
    if Path(a.input).stem=='test' and not a.final_test:raise ValueError('Use --final-test only after selecting the configuration on validation')
    evaluate(yaml.safe_load(Path(a.config).read_text()),read_jsonl(a.input),a.output,a.adapter,a.shots,not a.term_only,a.limit,a.final_test)

if __name__=='__main__':main()
