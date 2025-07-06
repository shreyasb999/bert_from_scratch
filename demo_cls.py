"""
Toy sequence-classification demo: teaches BERT to distinguish
sentences that contain the word "good" vs "bad".

$ python -m bert_simplified.demo_cls
"""
import torch
from torch.utils.data import DataLoader, Dataset
from . import BertConfig, WordPieceTokenizer, BertForSequenceClassification

sentences = [
    ("this movie is good", 1),
    ("what a good day", 1),
    ("bad weather today", 0),
    ("this is a bad idea", 0),
] * 64  # duplicate to create a mini-dataset

class SimpleClsDS(Dataset):
    def __init__(self, pairs, tk, max_len=16):
        self.tk = tk; self.max_len = max_len
        self.data = pairs
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        txt, label = self.data[idx]
        ids = self.tk.encode(txt)[: self.max_len]
        pad_id = self.tk.vocab["[PAD]"]
        ids += [pad_id] * (self.max_len - len(ids))
        attn = [1 if t != pad_id else 0 for t in ids]
        return torch.tensor(ids), torch.tensor(attn), torch.tensor(label)

def main():
    tk = WordPieceTokenizer(vocab_size=1000); tk.train([s for s, _ in sentences])
    cfg = BertConfig(vocab_size=len(tk.vocab), hidden_size=128, num_heads=2, num_layers=2)
    model = BertForSequenceClassification(cfg, num_labels=2)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ds = SimpleClsDS(sentences, tk)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    for epoch in range(5):
        total = 0
        for ids, attn, y in dl:
            logits, loss = model(ids, attn, y)
            loss.backward(); optim.step(); optim.zero_grad()
            total += loss.item()
        print(f"epoch {epoch+1}: loss={total/len(dl):.4f}")

if __name__ == "__main__":
    main()
