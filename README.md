# bert_from_scratch


A minimal PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) built entirely from scratch, without relying on the Hugging Face or TensorFlow packages. This repository demonstrates the core mechanics of BERT - embedding layers, multi-head self-attention, transformer encoder blocks, and optional pre-training heads; packaged in a clear, modular directory structure.

---

## Objective

The primary goal of this project is to **demystify BERT** by:

- **Re-implementing** all key components from the original paper in pure PyTorch.
- **Stripping away** unnecessary complexity for a teaching-focused, lightweight model (default: 4 layers, 4 heads, 256 hidden size).
- **Providing** ready-to-run demos for masked language modeling (MLM) and simple sequence classification.

This project is ideal for students, researchers, and engineers who want hands-on experience with transformer internals.

---

## Repository Structure

```
bert_simplified/
├── __init__.py
├── config.py                # Configuration class for model sizes
├── embeddings.py            # Token & position embeddings
├── attention.py             # Multi-head self-attention implementation
├── encoder.py               # Transformer encoder block
├── model.py                 # BERT encoder & pooler (forward pass with mask)
├── tokenizer.py             # Simple WordPiece-style tokenizer
├── heads.py                 # Optional MLM & classification heads
├── demo_mlm.py              # Toy masked-LM training loop
├── demo_cls.py              # Toy sequence-classification demo
└── test_run.py              # Quick smoke-test of forward pass
```

---

## How It Works

1. **Configuration** (`config.py`)

   - All model hyperparameters (hidden size, number of heads/layers, vocab size, etc.) are centralized in `BertConfig` for easy tuning.

2. **Embeddings** (`embeddings.py`)

   - Token + position embeddings combined and subjected to dropout.

3. **Self-Attention** (`attention.py`)

   - Single `MultiHeadSelfAttention` module: projects queries/keys/values, computes scaled-dot-product attention, and concatenates the heads.

4. **Transformer Encoder** (`encoder.py`)

   - One encoder layer: attention → add & norm → feed-forward network → add & norm.

5. **BERT Model** (`model.py`)

   - Stacks multiple encoder layers, pools the `[CLS]` token representation with a linear & Tanh layer, and handles attention masks.

6. **Tokenizer** (`tokenizer.py`)

   - Ultra-light WordPiece-style tokenizer: trains on raw text, builds a vocab, and provides `encode`/`decode`.

7. **Heads** (`heads.py`)

   - **MLM head**: BERT + tied-weight masked-language-model projection and loss.
   - **Sequence-classification head**: simple linear classifier on pooled output.

8. **Demos & Tests**

   - `test_run.py`: verifies model dimensions on dummy inputs.
   - `demo_mlm.py`: small loop that masks tokens in a toy text file and learns to predict them.
   - `demo_cls.py`: toy binary classification ("good" vs "bad" sentences).

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install torch
   ```

2. **Run smoke test**

   ```bash
   python -m bert_simplified.test_run
   ```

3. **Masked-LM demo**

   ```bash
   echo "hello world" > tiny.txt
   python -m bert_simplified.demo_mlm tiny.txt
   ```

4. **Sequence-classification demo**

   ```bash
   python -m bert_simplified.demo_cls
   ```

---

## Future Work

- **Full WordPiece BPE**: replace the simplified tokenizer with a proper subword algorithm.
- **Pre-training on real data**: integrate a two-phase training schedule (seq\_len 128 → 512) on Wiki+Books.
- **Fine-tuning scripts**: add classification, QA, and NER examples using downstream datasets.
- **Mixed precision & gradient accumulation** for large-batch training.
- **Distributed training** via PyTorch DDP.

---


Feel free to open issues or pull requests with suggestions, improvements, or bug fixes. Enjoy diving into the inner workings of BERT!

