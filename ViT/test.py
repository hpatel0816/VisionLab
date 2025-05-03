import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import *

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
    
    assert diff < threshold, f"The mean absolute difference of {diff} is beyond the threshold ({threshold})."


def test_patch_embeddings():
    image = torch.randn(4, 3, 224, 224)
    patch_embed = PatchEmbeddings()
    out = patch_embed(image)

    assert out.shape == (4, 196, 768), f"Expected shape: (4, 196, 768). Recieved shape: {out.shape}."


def test_input_embeddings():
    config = {
        "image_size": 224,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_dims": 768,
        "dropout": 0.0 
    }

    model = InputEmbedding(config)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    # Calculate expected number of patches
    num_patches = (config["image_size"] // config["patch_size"]) ** 2
    expected_shape = (2, num_patches + 1, config["hidden_dims"])

    assert output.shape == expected_shape, f"Expected shape: {expected_shape}. Recieved shape: {output.shape}."


def test_self_attention():
    config = {
        "hidden_dims": 768,
        "num_attn_heads": 12,
        "dropout": 0.1,
    }

    # Sample input: batch size = 2, sequence length = 197 (196 patches + 1 CLS), hidden dim = 768
    input = torch.randn(2, 197, config["hidden_dims"])

    attn_module = MultiHeadAttention(config)
    output, attn_scores = attn_module(input, return_score=True)

    assert output.shape == (2, 197, config["hidden_dims"]), f"Expected shape: {(2, 197, config["hidden_dims"])}. Recieved shape: {output.shape}."
    assert attn_scores.shape == (2, config["num_attn_heads"], 197, 197), f"Expected shape: {(2, config["num_attn_heads"], 197, 197)}. Recieved shape: {attn_scores.shape}."
