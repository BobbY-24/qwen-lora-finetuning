from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from community_qwen.utils.device import select_device, select_dtype
from community_qwen.utils.io import digest


def discover():
    roots = [Path.home()/'.cache/huggingface/hub', Path.home()/'models', Path.home()/'Documents/Codex']
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob('**/config.json'):
            if '.venv' in p.parts or 'site-packages' in p.parts:
                continue
            try:
                c = json.loads(p.read_text())
                weights = list(p.parent.glob('*.safetensors'))
                if str(c.get('model_type','')).startswith('qwen') and weights:
                    candidates.append((sum(x.stat().st_size for x in weights), str(p.parent)))
            except (ValueError, OSError):
                continue
    if not candidates:
        raise FileNotFoundError('No local Qwen safetensors checkpoint found. Set model.local_path explicitly; downloads are disabled.')
    return sorted(candidates)[0][1]


def checkpoint_identity(path):
    if str(path) == 'AUTO_DISCOVER':
        path = discover()
    path = Path(path).resolve()
    return {'path': str(path), 'config': json.loads((path/'config.json').read_text()),
            'files': [{'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in sorted(path.glob('*')) if p.is_file()],
            'tokenizer_sha256': digest((path/'tokenizer_config.json').read_text())}


def load_tokenizer(path):
    if str(path) == 'AUTO_DISCOVER':
        path = discover()
    t = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    if not t.chat_template:
        raise ValueError('Checkpoint has no chat template. Supply an audited template before instruction tuning.')
    return t


def load_model(config, quantized=False):
    path = config['model']['local_path']
    if path == 'AUTO_DISCOVER':
        path = discover()
        # Persist the resolved identity for inspection, manifests and checkpoint fingerprints.
        config['model']['local_path'] = path
    device = select_device(config['model'].get('device', 'auto'))
    dtype = select_dtype(device, config['model'].get('dtype', 'auto'))
    kwargs = dict(local_files_only=True, trust_remote_code=False, dtype=dtype, attn_implementation='sdpa')
    if quantized:
        from .qlora import quantization_config
        kwargs.update(quantization_config=quantization_config(config, device, dtype), device_map={'': str(device)})
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not quantized:
        model.to(device)
    return model, load_tokenizer(path), device
