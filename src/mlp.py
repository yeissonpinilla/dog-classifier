import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, embedding_dim = 768, out_dim = 3072, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x