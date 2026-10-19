"""Publish complete adapter + training state atomically. Load only trusted local checkpoints."""
from pathlib import Path
import os
import shutil
import torch
from peft import set_peft_model_state_dict, load_peft_weights
from .seed import rng_state,restore_rng


def save_checkpoint(path,model,optimizer,scheduler,state,fingerprint,scaler=None):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp/'adapter')
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'state':state,'fingerprint':fingerprint,'rng':rng_state(),
                'scaler':scaler.state_dict() if scaler else None},temp/'training.pt')
    os.replace(temp,path)


def load_checkpoint(path,model,optimizer,scheduler,fingerprint,scaler=None):
    path=Path(path)
    saved=torch.load(path/'training.pt',map_location='cpu',weights_only=False)
    if saved['fingerprint']!=fingerprint:raise ValueError('Checkpoint model/config/data/code mismatch')
    set_peft_model_state_dict(model,load_peft_weights(str(path/'adapter'),device='cpu'))
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    if scaler and saved['scaler'] is not None:scaler.load_state_dict(saved['scaler'])
    restore_rng(saved['rng'])
    return saved['state']
