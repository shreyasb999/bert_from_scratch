import torch.nn as nn
from .embeddings import BertEmbeddings
from .encoder import TransformerEncoderLayer


class BERT(nn.Module):
    """
    Encoder-only BERT (no pre-training heads to keep it tiny).
    """

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids:      (batch, seq)
        attention_mask: (batch, seq) where 1 = keep, 0 = pad
        returns: sequence_output, pooled_output
        """
        if attention_mask is not None:
            # expand mask from (batch, seq) → (batch, 1, 1, seq)
            attention_mask = attention_mask[:, None, None, :]

        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        pooled = self.activation(self.pooler(x[:, 0]))  # CLS token
        return x, pooled
