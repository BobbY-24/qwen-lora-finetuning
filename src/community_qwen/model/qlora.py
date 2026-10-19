"""Optional CUDA bitsandbytes path; never pretend MPS LoRA is QLoRA."""
import importlib.util
from transformers import BitsAndBytesConfig


def quantization_config(config, device, dtype):
    if device.type != 'cuda':
        raise RuntimeError('This QLoRA implementation is gated to CUDA + bitsandbytes. Apple MPS is unsupported; use ordinary LoRA.')
    if importlib.util.find_spec('bitsandbytes') is None:
        raise RuntimeError('Install the qlora optional dependency on a CUDA machine.')
    bits = config.get('quantization', {}).get('bits', 4)
    if bits not in (4, 8):
        raise ValueError('bits must be 4 or 8')
    return BitsAndBytesConfig(load_in_4bit=bits==4, load_in_8bit=bits==8,
        bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
