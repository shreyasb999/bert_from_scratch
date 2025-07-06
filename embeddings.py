import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """
    Token + position embeddings with dropout.
    Segment (token-type) embeddings are omitted for simplicity.
    """

    def __init__(self, cfg):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token(input_ids) + self.position(pos_ids)
        return self.dropout(x)
