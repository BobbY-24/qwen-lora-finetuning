import torch


def make_optimizer(model,config):
    classes={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}
    if config['optimizer'] not in classes:raise ValueError('Unknown optimizer')
    return classes[config['optimizer']]([p for p in model.parameters() if p.requires_grad],
                                       lr=config['learning_rate'],weight_decay=config['weight_decay'])
