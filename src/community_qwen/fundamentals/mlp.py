import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self,input_dim=4,hidden_dim=8,output_dim=2):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim),nn.GELU())

    def forward(self,x):
        return self.layers(x)


def demonstrate():
    torch.manual_seed(42)
    model=TinyMLP()
    x=torch.randn(8,4); target=torch.randn(8,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss=torch.nn.functional.mse_loss(model(x),target)
    optimizer.zero_grad();loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
    optimizer.step()
    return {'loss':loss.item(),'gradient_norms':gradients,'parameters':sum(p.numel() for p in model.parameters())}
