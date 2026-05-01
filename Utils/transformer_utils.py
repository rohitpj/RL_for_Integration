import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy
import ast
import copy

sys.path.append("SymbolicMathematics")
from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser
from sympy import sympify
from sympy import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

#from sympy.integrals import manualintegrate
import SymbolicMathematics
import manualintegrate
import numexpr
import tqdm

from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser

class ExpressionTokenizer:
    def __init__(self, vocab):
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        # Reserve 0 for padding so nn.Embedding(..., padding_idx=0) is consistent.
        self.token_to_id = {self.pad_token: 0}
        for token in vocab:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        if self.unk_token not in self.token_to_id:
            self.token_to_id[self.unk_token] = len(self.token_to_id)

        # Reverse mapping
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def encode(self, token_list, max_length=384):
        """Convert tokens to token IDs and generate attention mask."""
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in token_list]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)

        # Truncate or pad
        token_ids = token_ids[:max_length] + [self.token_to_id[self.pad_token]] * (max_length - len(token_ids))
        attention_mask = attention_mask[:max_length] + [0] * (max_length - len(attention_mask))

        return token_ids, attention_mask

class CLS_Tokenizer:
    def __init__(self, vocab):
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.vobab = vocab

        # Reserve 0 for [PAD] and explicitly register [CLS].
        self.token_to_id = {self.pad_token: 0}
        for token in vocab:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        if self.unk_token not in self.token_to_id:
            self.token_to_id[self.unk_token] = len(self.token_to_id)
        if self.cls_token not in self.token_to_id:
            self.token_to_id[self.cls_token] = len(self.token_to_id)

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def encode(self, token_list, max_length=384):
        token_list = [self.cls_token] + token_list 
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in token_list]

        attention_mask = [1] * len(token_ids)

        # Truncate or pad
        token_ids = token_ids[:max_length] + [self.token_to_id[self.pad_token]] * (max_length - len(token_ids))
        attention_mask = attention_mask[:max_length] + [0] * (max_length - len(attention_mask))

        return token_ids, attention_mask

class CustomDataset(Dataset):
    def __init__(self, tokenized_data, attention_masks, labels):
        self.tokenized_data = tokenized_data
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokenized_data[idx], dtype=torch.long),
            torch.tensor(self.attention_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=384):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, num_classes=24, max_len=384):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        x = self.embedding(x) 
        x = self.pos_encoding(x)

        if attention_mask is not None:
            attention_mask = attention_mask == 0 

        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1) 
        x = self.fc_out(x)

        return x

class CLS_TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, num_classes=24, max_len=384):
        super(CLS_TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        x = self.embedding(x)  
        x = self.pos_encoding(x) 

        if attention_mask is not None:
            attention_mask = attention_mask == 0  

        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)  

        cls_representation = x[:, 0, :]

        logits = self.fc_out(cls_representation)

        return logits


def print_rule_distribution(logits, rule_map, top_k=10):

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]  # Remove batch dimension
    topk_probs, topk_indices = torch.topk(probs, top_k)

    print(f"\nTop {top_k} rule probabilities:")
    print("=" * 40)
    for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist()):
        rule_func = rule_map[idx]
        rule_name = rule_func.__name__ if hasattr(rule_func, "__name__") else str(rule_func)
        print(f"{rule_name:<25} : {prob:.4f}")
    print("=" * 40)
