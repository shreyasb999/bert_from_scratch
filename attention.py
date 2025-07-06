import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Classic scaled-dot self-attention.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0, "heads must divide hidden"
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        b, t, h = x.size()                          # (batch, seq, hidden)
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                # each: (b, t, heads, dim)

        # transpose to (b, heads, seq, dim)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:                       # mask: (b, 1, 1, seq)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        context = attn @ v                         # (b, heads, seq, dim)
        context = context.transpose(1, 2).contiguous().view(b, t, h)
        return self.out(context)
