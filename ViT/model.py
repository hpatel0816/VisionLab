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
    

def test_layer_norm():
    x = torch.randn(10, 20)  # batch of 10 samples, each with 20 features

    # Custom LayerNorm
    my_ln = LayerNormalization()
    my_ln.alpha.data = torch.ones(1)
    my_ln.beta.data = torch.zeros(1)

    # PyTorch's LayerNorm
    torch_ln = nn.LayerNorm(20, eps=1e-6)
    torch_ln.weight.data = torch.ones(20)
    torch_ln.bias.data = torch.zeros(20)

    # Run & compare both
    out_my = my_ln(x)
    out_torch = torch_ln(x)
    print("Mean absolute difference:", torch.mean(torch.abs(out_my - out_torch)))

test_layer_norm()

