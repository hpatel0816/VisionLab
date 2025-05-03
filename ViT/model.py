import torch
import torch.nn as nn
import math


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
    """
    Converts an image into a flattened sequence of patch embeddings using a Conv2D projection.
    """
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
        # (b, c, h, w) --> (b, emb_dim, h ,w) --> (b, emb_dim, h*w) --> (b, h*w, emb_dim)
        return self.projection(input_image).flatten(2).transpose(1,2)
        

class InputEmbedding(nn.Module):
    """
    Embeds an input image into a sequence of patch tokens with positional information,
    including a learnable token ([CLS]) for classification in final stage of Vision Transformers (ViT).
    """
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["hidden_dims"]))
        self.patch_embeddings = PatchEmbeddings(
            image_size=(config["image_size"], config["image_size"]),
            patch_size=(config["patch_size"], config["patch_size"]),
            num_channels=config["num_channels"],
            embedding_dims=config["hidden_dims"]
        )

        num_patches = self.patch_embeddings.num_patches
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, config["hidden_dims"]))
        self.dropout = nn.Dropout(config["dropout"]) # Dropout layer for regularization, etc

    def forward(self, input_image):
        batch_size = input_image.shape[0]
        image_embedding = self.patch_embeddings(input_image) # (b, h*w, emb_dim)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (1, 1, emb_dim) --> (batch_size, 1, emb_dim)
        cls_and_image_emb = torch.cat((cls_token, image_embedding), dim=1)
        final_embedding = cls_and_image_emb + self.positional_embedding
        final_embedding_with_dropout = self.dropout(final_embedding)

        return final_embedding_with_dropout


class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head self-attention block for Vision Transformers (ViT), allowing each image patch
    to extract contextual relationships from its tokens.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config["hidden_dims"]
        self.num_attn_heads = config["num_attn_heads"]
        assert self.hidden_dim % self.num_attn_heads == 0, "The hidden_dims is not divisible by num_attn_heads"
        self.attention_head_size = self.hidden_dim//self.num_attn_heads
        self.attention_map_size = self.attention_head_size * self.num_attn_heads

        self.query = nn.Linear(self.hidden_dim, self.attention_map_size)
        self.key = nn.Linear(self.hidden_dim, self.attention_map_size)
        self.value = nn.Linear(self.hidden_dim, self.attention_map_size)

        self.dropout = nn.Dropout(config["dropout"])

    def convert_linear_map_to_multihead(self, linear_map):
        # linear_map_shape = (batch, seq_len, hidden_dim) --> (batch, seq_len, num_attn_head, attn_head_size)
        multi_head_shape = linear_map.size()[:-1] + (self.num_attn_heads, self.attention_head_size)
        multi_head_linear_map = linear_map.view(*multi_head_shape)

        return multi_head_linear_map.permute(0, 2, 1, 3) # (batch, num_attn_head, seq_len, attn_head_size)
    
    def forward(self, input_embedding, return_score=False):
        query = self.query(input_embedding)
        key = self.key(input_embedding)
        value = self.value(input_embedding)

        query_multihead = self.convert_linear_map_to_multihead(query)
        key_multihead = self.convert_linear_map_to_multihead(key)
        value_multihead = self.convert_linear_map_to_multihead(value)

        # Compute attention
        attention_score = torch.matmul(query_multihead, key_multihead.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probabilities = nn.Softmax(dim=-1)(attention_score)
        attention_dropout = self.dropout(attention_probabilities)
        attention_output = torch.matmul(attention_dropout, value_multihead)

        final_attn_output = attention_output.permute(0, 2, 1, 3).contiguous() # (batch, seq_len, num_attn_head, attn_head_size)
        reshape_for_final_attn = final_attn_output.size()[:-2] + (self.attention_map_size, )
        reshaped_final_attn_layer = final_attn_output.view(*reshape_for_final_attn)

        return (reshaped_final_attn_layer, attention_probabilities) if return_score else (reshaped_final_attn_layer,)

