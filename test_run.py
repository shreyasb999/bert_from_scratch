"""
Quick smoke-test for the simplified BERT implementation.

$ python -m bert_simplified.test_run
"""

import torch
from . import BertConfig, BERT


def main():
    cfg = BertConfig(
        vocab_size=30522,   # Use 10k-30k for a real tokenizer
        hidden_size=256,
        num_heads=4,
        num_layers=4,
    )

    model = BERT(cfg)
    batch, seq_len = 2, 16
    dummy_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attention_mask = torch.ones_like(dummy_ids)

    seq_out, pooled = model(dummy_ids, attention_mask)
    print("sequence_output:", seq_out.shape)    # (2, 16, 256)
    print("pooled_output:  ", pooled.shape)     # (2, 256)


if __name__ == "__main__":
    main()
