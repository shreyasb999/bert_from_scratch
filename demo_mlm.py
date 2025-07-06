"""
Tiny masked-LM training loop that learns to predict random
masked tokens on a toy corpus.

$ python -m bert_simplified.demo_mlm data.txt
"""
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import random

from . import BertConfig, WordPieceTokenizer, BertForMaskedLM

MASK = "[MASK]"

class ToyMLMDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_len=32, mlm_prob=0.15):
        self.tk = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.lines = [l.strip() for l in open(txt_file, encoding="utf-8") if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ids = self.tk.encode(self.lines[idx])
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]

        labels = [-100] * len(ids)
        for i in range(1, len(ids) - 1):  # skip CLS/SEP
            if random.random() < self.mlm_prob:
                labels[i] = ids[i]
                ids[i] = self.tk.vocab[MASK]
        pad = self.tk.vocab["[PAD]"]
        ids += [pad] * (self.max_len - len(ids))
        labels += [-100] * (self.max_len - len(labels))
        attn = [1 if t != pad else 0 for t in ids]
        return torch.tensor(ids), torch.tensor(attn), torch.tensor(labels)

def main():
    txt = sys.argv[1]
    tk = WordPieceTokenizer(vocab_size=8000); tk.train([txt])
    cfg = BertConfig(vocab_size=len(tk.vocab), hidden_size=128, num_heads=2, num_layers=2)
    model = BertForMaskedLM(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)

    ds = ToyMLMDataset(txt, tk)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    for epoch in range(3):
        total = 0
        for ids, attn, labels in dl:
            logits, loss = model(ids, attn, labels)
            loss.backward(); optim.step(); optim.zero_grad()
            total += loss.item()
        print(f"epoch {epoch+1}: loss={total/len(dl):.4f}")

if __name__ == "__main__":
    main()
