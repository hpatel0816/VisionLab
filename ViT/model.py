import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        z_val = (x - mean) / torch.sqrt(var + self.epsilon)
        return z_val * self.alpha + self.beta
    
