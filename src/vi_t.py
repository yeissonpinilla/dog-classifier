import torch
import torch.nn as nn
from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder

class ViT(nn.Module):

    def __init__(self,
    channels = 3,
    image_size = 224,
    patch_size = 16,
    embedding_dim = 768,
    encoder_layers = 12,
    num_heads = 12,
    attn_dropout = 0.0,
    mlp_dropout = 0.1,
    mlp_out_dim = 3072,
    num_classes = 1000,
    embedding_dropout = 0.1):
        super().__init__()
        self.num_patches = (int(image_size // patch_size)) ** 2
        self.class_embedding = nn.Parameter(torch.randn(1,1,embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.patch_embedding = PatchEmbedding(channels, image_size, patch_size, embedding_dim)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(embedding_dim, num_heads, attn_dropout, mlp_out_dim = mlp_out_dim, mlp_dropout = mlp_dropout) for _ in range(encoder_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes))

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
