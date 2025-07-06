"""
Ultra-light WordPiece‐style tokenizer:
* train() builds a vocab from raw text files
* encode() turns text -> List[int]
* decode() turns List[int] -> text
This is NOT the full Google WordPiece algorithm, but it’s good
enough for experiments and keeps the code tiny.
"""
import json
from collections import Counter
from pathlib import Path

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"

class WordPieceTokenizer:
    def __init__(self, vocab_size=30_000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special = [PAD, UNK, CLS, SEP, MASK]
        self.vocab = {}                # token -> id
        self.inv_vocab = {}            # id -> token

    # ---------- training ---------- #
    def train(self, text_files):
        counter = Counter()
        for file in text_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    counter.update(line.strip().split())

        # 1. seed with specials
        vocab = list(self.special)

        # 2. add words by freq until we hit vocab_size
        for word, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if word not in vocab:
                vocab.append(word)
            if len(vocab) >= self.vocab_size:
                break

        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    # ---------- serialization ---------- #
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path):
        self.vocab = json.load(open(path, "r", encoding="utf-8"))
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    # ---------- text <-> ids ---------- #
    def encode(self, text, add_special=True):
        tokens = text.strip().split()
        if add_special:
            tokens = [CLS] + tokens + [SEP]
        return [self.vocab.get(t, self.vocab[UNK]) for t in tokens]

    def decode(self, ids, skip_special=True):
        tokens = [self.inv_vocab[i] for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in self.special]
        return " ".join(tokens)
