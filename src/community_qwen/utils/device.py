import torch


def select_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process. On this Mac, run with host GPU access or select CPU explicitly.')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device(requested)


def select_dtype(device, name='auto'):
    if name == 'auto':
        name = 'bfloat16' if device.type == 'mps' or (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else 'float32'
    if name not in ['float32', 'bfloat16', 'float16']:
        raise ValueError(name)
    if name == 'float16' and device.type != 'cuda':
        raise ValueError('FP16 training requires CUDA GradScaler in this implementation; use FP32/BF16 here.')
    return getattr(torch, name)
