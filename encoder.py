import torch.nn as nn
from .attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    """
    One encoder block: MHSA ➔ AddNorm ➔ FFN ➔ AddNorm
    """

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
