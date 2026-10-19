"""Visible single-device training loop; accumulation is weighted by assistant token count."""
import csv
import json
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from community_qwen.data.collator import AssistantCollator
from community_qwen.data.tokenize import encode_example
from community_qwen.model.load import load_model,checkpoint_identity
from community_qwen.model.lora import attach_lora,accounting
from community_qwen.utils.io import digest,write_json,code_identity
from community_qwen.utils.environment import environment,memory
from .seed import seed_everything
from .optimizer import make_optimizer
from .scheduler import make_scheduler
from .validate import validate,target_count
from .checkpoint import save_checkpoint,load_checkpoint


def train(config,train_rows,val_rows,output,resume=None,quantized=False):
    output=Path(output)
    if output.exists() and not resume:raise FileExistsError('Use a new run directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(config['seed'])
    tc=config['training']
    if tc['micro_batch_size']<1 or tc['gradient_accumulation_steps']<1 or tc['epochs']<1:
        raise ValueError('Batch sizes and epochs must be positive')
    model,tokenizer,device=load_model(config,quantized)
    model=attach_lora(model,config['lora'],quantized)
    if tc.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model.config.use_cache=False
    max_length=config['model']['max_sequence_length']
    if max_length=='auto':max_length=json.loads(Path('data/processed/manifest.json').read_text())['max_sequence_length']
    fraction=tc.get('data_fraction',1.)
    if not 0<fraction<=1:raise ValueError('data_fraction must be in (0,1]')
    # Select whole terms; fixed ordering makes scale conditions nested at a fixed seed.
    keys=sorted({r['term_key'] for r in train_rows})
    import random
    random.Random(config['seed']).shuffle(keys)
    chosen=set(keys[:max(1,round(len(keys)*fraction))])
    train_rows=[r for r in train_rows if r['term_key'] in chosen]
    if tc.get('train_limit'):train_rows=train_rows[:tc['train_limit']]
    if tc.get('validation_limit'):val_rows=val_rows[:tc['validation_limit']]
    encoded=[encode_example(r,tokenizer,max_length,i%3,tc.get('context',True)) for i,r in enumerate(train_rows)]
    val_encoded=[encode_example(r,tokenizer,max_length,context=tc.get('context',True)) for r in val_rows]
    if not encoded or not val_encoded:raise ValueError('Empty train/validation')
    collator=AssistantCollator(tokenizer.pad_token_id)
    val_loader=DataLoader(val_encoded,batch_size=1,collate_fn=collator)
    micro=tc['micro_batch_size'];accum=tc['gradient_accumulation_steps']
    batches=math.ceil(len(encoded)/micro)
    total_steps=math.ceil(batches/accum)*tc['epochs']
    if tc.get('max_steps'):total_steps=min(total_steps,tc['max_steps'])
    optimizer=make_optimizer(model,tc)
    scheduler=make_scheduler(optimizer,total_steps,tc['warmup_ratio'],tc['scheduler'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and next(model.parameters()).dtype==torch.float16)
    fp=digest({'config':config,'train':train_rows,'validation':val_rows,'model':checkpoint_identity(config['model']['local_path']),
               'code':code_identity()['source_sha256'],'quantized':quantized})
    state={'epoch':0,'next_batch':0,'global_step':0,'best_validation_loss':float('inf'),'bad_evals':0,
           'examples_processed':0,'tokens_processed':0,'elapsed_seconds':0.,'initial_validation':None}
    if resume:state=load_checkpoint(resume,model,optimizer,scheduler,fp,scaler)
    else:
        (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write_json(output/'environment.json',environment())
        write_json(output/'manifest.json',{'fingerprint':fp,'train_ids':[r['id'] for r in train_rows],
            'validation_ids':[r['id'] for r in val_rows], 'model':checkpoint_identity(config['model']['local_path'])})
    start=time.monotonic();prior_elapsed=state['elapsed_seconds']
    if state['initial_validation'] is None:state['initial_validation']=validate(model,val_loader,device)
    fields=['global_step','epoch','training_loss','validation_loss','learning_rate','gradient_norm','examples_processed','tokens_processed','elapsed_seconds']
    metrics_path=output/'metrics.csv'
    if resume and metrics_path.exists():
        old=list(csv.DictReader(metrics_path.open()))
        with metrics_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r for r in old if int(r['global_step'])<=state['global_step'])
    handle=metrics_path.open('a',newline='');writer=csv.DictWriter(handle,fieldnames=fields)
    if handle.tell()==0:writer.writeheader()
    status='completed';last_loss=None;max_grad=0.;peak_mps_sample=0
    try:
        for epoch in range(state['epoch'],tc['epochs']):
            # Epoch-addressable permutation + private DataLoader RNG avoids perturbing dropout RNG on resume.
            generator=torch.Generator().manual_seed(config['seed']+epoch)
            indices=torch.randperm(len(encoded),generator=generator).tolist()
            loader=DataLoader([encoded[i] for i in indices],batch_size=micro,collate_fn=collator,generator=generator)
            batch_list=iter(loader)
            begin=state['next_batch'] if epoch==state['epoch'] else 0
            for _ in range(begin):next(batch_list)
            model.train()
            for first in range(begin,batches,accum):
                if state['global_step']>=total_steps:break
                window=[next(batch_list) for _ in range(min(accum,batches-first))]
                total_tokens=sum(target_count(b) for b in window)
                optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for batch in window:
                    batch={k:v.to(device) for k,v in batch.items()}
                    n=target_count(batch)
                    outputs=model(**batch)
                    loss=outputs.loss
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(loss*(n/total_tokens)).backward()
                    loss_sum+=float(loss.detach())*n
                    state['examples_processed']+=batch['input_ids'].shape[0]
                    state['tokens_processed']+=n
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],tc['max_grad_norm'],error_if_nonfinite=True)
                max_grad=max(max_grad,float(norm));used_lr=optimizer.param_groups[0]['lr']
                scaler.step(optimizer);scaler.update();scheduler.step()
                state['global_step']+=1;state['epoch']=epoch;state['next_batch']=min(batches,first+accum)
                state['elapsed_seconds']=prior_elapsed+time.monotonic()-start
                last_loss=loss_sum/total_tokens
                row=dict(global_step=state['global_step'],epoch=epoch+state['next_batch']/batches,training_loss=last_loss,
                         validation_loss='',learning_rate=used_lr,gradient_norm=float(norm),examples_processed=state['examples_processed'],
                         tokens_processed=state['tokens_processed'],elapsed_seconds=state['elapsed_seconds'])
                sample=memory(device);peak_mps_sample=max(peak_mps_sample,sample.get('mps_current_allocated_bytes',0))
                end_epoch=state['next_batch']==batches or state['global_step']==total_steps
                timed_out=state['elapsed_seconds']>=tc.get('max_seconds',3600)
                if end_epoch or timed_out:
                    val=validate(model,val_loader,device);row['validation_loss']=val['loss']
                    if val['loss']<state['best_validation_loss']:
                        state['best_validation_loss']=val['loss'];state['bad_evals']=0
                        model.save_pretrained(output/'adapter')
                    else:state['bad_evals']+=1
                writer.writerow(row);handle.flush()
                print(json.dumps(row),flush=True)
                with (output/'training.log').open('a') as log:log.write(json.dumps(row)+'\n')
                if end_epoch or timed_out:
                    if state['next_batch']==batches:state['epoch']=epoch+1;state['next_batch']=0
                    save_checkpoint(output/f'checkpoint-{state["global_step"]}',model,optimizer,scheduler,state,fp,scaler)
                if timed_out:status='time_limit';break
                if state['bad_evals']>=tc['patience']:status='early_stopping';break
            if status!='completed' or state['global_step']>=total_steps:break
    finally:
        handle.close()
    summary=dict(status=status,**accounting(model),**state,final_training_loss=last_loss,maximum_unclipped_gradient_norm=max_grad,
                 runtime_seconds=prior_elapsed+time.monotonic()-start,**memory(device),sampled_peak_mps_allocated_bytes=peak_mps_sample,
                 device=str(device),base_dtype=str(next(model.parameters()).dtype),train_rows=len(train_rows),train_terms=len({r['term_key'] for r in train_rows}),
                 effective_batch_size=micro*accum,quantized=quantized,total_planned_steps=total_steps,
                 adapter_selection='lowest validation loss',best_adapter=str(output/'adapter'))
    write_json(output/'summary.json',summary)
    plot_training(output)
    return summary


def plot_training(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((Path(output)/'metrics.csv').open()))
    if not rows:return
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot([int(r['global_step']) for r in rows],[float(r['training_loss']) for r in rows],label='Train')
    val=[r for r in rows if r['validation_loss']]
    ax[0].plot([int(r['global_step']) for r in val],[float(r['validation_loss']) for r in val],'o-',label='Validation')
    ax[0].set(xlabel='Optimizer step',ylabel='Assistant token cross entropy');ax[0].legend()
    ax[1].plot([int(r['global_step']) for r in rows],[float(r['learning_rate']) for r in rows])
    ax[1].set(xlabel='Optimizer step',ylabel='Learning rate')
    fig.tight_layout();fig.savefig(Path(output)/'learning_curve.png',dpi=150);plt.close(fig)
