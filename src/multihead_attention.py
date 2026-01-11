
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim = 768, num_heads = 12, attn_dropout = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.norm(x)
        attn_output, _ = self.multihead_attn(x, x, x, need_weights=False)
        return attn_output