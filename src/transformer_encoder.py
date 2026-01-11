from multihead_attention import MultiHeadAttention
from mlp import MLP
import torch.nn as nn

class TransformerEncoder(nn.Module):

    def __init__(self, embedding_dim = 768, num_heads = 12, attn_dropout = 0.0, mlp_dropout = 0.1, mlp_out_dim = 3072):
        super().__init__()
        self.msa = MultiHeadAttention(embedding_dim, num_heads, attn_dropout)
        self.mlp = MLP(embedding_dim, mlp_out_dim, mlp_dropout)

    def forward(self, x):
        x = x + self.msa(x)
        x = x + self.mlp(x)
        return x