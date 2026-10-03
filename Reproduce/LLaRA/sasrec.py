"""Local SASRec encoder used for LLaRA item embeddings.

The layout follows the one-block SASRec recommender in the LLaRA reference:
right padding uses item ID ``num_items``, and ``cacu_x`` returns the trained
item embedding table before the language-model projector.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.first = nn.Conv1d(size, size, 1)
        self.second = nn.Conv1d(size, size, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(x.transpose(1, 2)))
        hidden = self.second(hidden).transpose(1, 2)
        return self.norm(x + self.dropout(hidden))


class SASRec(nn.Module):
    def __init__(self, hidden_size: int, item_num: int, state_size: int,
                 dropout: float = 0.1, num_heads: int = 1):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, std=1.0)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def cacu_x(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_embeddings(items)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.size(1) != self.state_size:
            raise ValueError("states must have shape [batch, state_size]")
        batch, length = states.shape
        valid = states.ne(self.item_num)
        positions = torch.arange(length, device=states.device)
        sequence = self.emb_dropout(self.item_embeddings(states) + self.positional_embeddings(positions))
        sequence = sequence * valid.unsqueeze(-1)
        queries = self.ln_1(sequence)
        width = queries.size(-1) // self.num_heads
        q = self.q(queries).reshape(batch, length, self.num_heads, width).transpose(1, 2)
        k = self.k(sequence).reshape(batch, length, self.num_heads, width).transpose(1, 2)
        v = self.v(sequence).reshape(batch, length, self.num_heads, width).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(queries.size(-1))
        causal = torch.ones(length, length, device=states.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, -1e9)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e9)
        weights = self.attn_dropout(scores.softmax(dim=-1)) * valid[:, None, :, None]
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        sequence = queries + attended
        sequence = self.feed_forward(self.ln_2(sequence)) * valid.unsqueeze(-1)
        return self.ln_3(sequence)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(states.size(0), device=states.device)
        hidden = self.encode(states)[rows, lengths - 1]
        return self.s_fc(hidden)
