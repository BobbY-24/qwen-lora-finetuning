import importlib.metadata
import platform
import resource
import shutil
import sys
import torch
from .io import code_identity


def environment():
    packages = {}
    for p in ['torch','transformers','peft','datasets','accelerate','bitsandbytes','numpy','matplotlib','safetensors']:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                platform=platform.platform(), packages=packages, cuda=torch.cuda.is_available(),
                mps=torch.backends.mps.is_available(), disk_free_bytes=shutil.disk_usage('.').free,
                **code_identity())


def memory(device):
    # macOS ru_maxrss is bytes; Linux is KiB. MPS has no peak API: name the sample honestly.
    out = {'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    if device.type == 'cuda':
        out['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    if device.type == 'mps':
        out['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        out['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    return out
