"""
Optional heads that sit on top of the BERT encoder.
"""
import torch.nn as nn
from .model import BERT
import torch


# ---------- Masked-Language-Model head ---------- #
class BertForMaskedLM(nn.Module):
    """
    BERT + tied-weight masked-LM projection.
    """
    def __init__(self, cfg):
        super().__init__()
        self.bert = BERT(cfg)
        # weight tying = reuse embedding matrix
        embed_weight = self.bert.embeddings.token.weight
        self.mlm_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.decoder.weight = embed_weight
        self.bias = nn.Parameter(nn.init.zeros_(torch.empty(cfg.vocab_size)))
        self.decoder.bias = self.bias
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_output, _ = self.bert(input_ids, attention_mask)
        x = self.mlm_dense(seq_output)
        x = self.act(x)
        x = self.norm(x)
        logits = self.decoder(x)

        loss = None
        if labels is not None:
            # flatten for CE
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return logits, loss


# ---------- Sequence-Classification head ---------- #
class BertForSequenceClassification(nn.Module):
    """
    BERT + simple CLS-pooled classifier (single label).
    """
    def __init__(self, cfg, num_labels: int = 2):
        super().__init__()
        self.bert = BERT(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        _, pooled = self.bert(input_ids, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss
