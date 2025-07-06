"""
Light-weight BERT-from-scratch package.
Importing `BertConfig` & `BERT` is enough:

from bert_simplified import BertConfig, BERT
"""
from .config import BertConfig
from .model import BERT
from .tokenizer import WordPieceTokenizer
from .heads import BertForMaskedLM, BertForSequenceClassification

__all__ = [
    "BertConfig",
    "BERT",
    "WordPieceTokenizer",
    "BertForMaskedLM",
    "BertForSequenceClassification",
]
