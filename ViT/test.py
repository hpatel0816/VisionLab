import torch
import torch.nn as nn

from model import LayerNormalization, PatchEmbeddings

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

    diff = torch.mean(torch.abs(out_my - out_torch))
    threshold = 1e-4
    
    print("Mean absolute difference:", diff)
    assert diff < threshold


def test_patch_embeddings():
    image = torch.randn(4, 3, 224, 224)
    patch_embed = PatchEmbeddings()
    out = patch_embed(image)

    assert out.shape == (4, 196, 768), f"Unexpected shape: {out.shape}"
