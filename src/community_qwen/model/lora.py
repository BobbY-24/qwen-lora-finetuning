import torch
from peft import LoraConfig, get_peft_model


def target_modules(model, requested='auto_inspect'):
    available = {name.rsplit('.', 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    targets = [name for name in ['q_proj','k_proj','v_proj','o_proj'] if name in available] if requested == 'auto_inspect' else requested
    if not isinstance(targets, list) or not targets or not set(targets) <= available:
        raise ValueError(f'Invalid LoRA targets: {targets}; available linear modules: {sorted(available)}')
    return targets


def attach_lora(model, config, quantized=False):
    from peft import prepare_model_for_kbit_training
    targets = target_modules(model, config.get('target_modules', 'auto_inspect'))
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=config['rank'], lora_alpha=config['alpha'],
        lora_dropout=config['dropout'], target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    return model


def accounting(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_parameters=total, trainable_parameters=trainable, frozen_parameters=total-trainable,
                trainable_percent=100*trainable/total)
