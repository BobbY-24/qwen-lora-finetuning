import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():torch.mps.manual_seed(seed)
    torch.set_num_threads(4)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mps':torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])
    if state['mps'] is not None:torch.mps.set_rng_state(state['mps'])
