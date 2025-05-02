import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        z_val = (x - mean) / torch.sqrt(var + self.epsilon)
        return z_val * self.alpha + self.beta
    

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int = (224,224), patch_size: int = (16,16), num_channels: int = 3, embedding_dims: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dims = embedding_dims
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.projection = nn.Conv2d(self.num_channels, self.embedding_dims, self.patch_size, self.patch_size)

    def forward(self, input_image):
        batch_size, num_channels, height, width = input_image.shape
        assert height * width == self.image_size[0] * self.image_size[1], "Correct image dimensions before parsing"
        return self.projection(input_image).flatten(2).transpose(1,2)
        
