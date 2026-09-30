"""PyTorch BERT4Rec: bidirectional attention and tied masked-item prediction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class BidirectionalBlock(nn.Module):
    """Explicit attention avoids TransformerEncoder's fused inference path in DP."""
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.qkv = nn.Linear(hidden_units, 3 * hidden_units)
        self.attention_output = nn.Linear(hidden_units, hidden_units)
        self.attention_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_units, 4 * hidden_units), nn.GELU(),
            nn.Linear(4 * hidden_units, hidden_units),
        )
        self.output_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, valid_keys):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid_keys[:, None, None, :],
            dropout_p=self.dropout_rate if self.training else 0.0, is_causal=False,
        )
        context = context.transpose(1, 2).reshape(batch, length, width)
        hidden = self.attention_norm(hidden + self.dropout(self.attention_output(context)))
        return self.output_norm(hidden + self.dropout(self.feed_forward(hidden)))


class BERT4Rec(nn.Module):
    def __init__(self, num_items, maxlen=100, hidden_units=128, num_blocks=2,
                 num_heads=2, dropout_rate=0.2, loss_chunk_size=128):
        super().__init__()
        self.num_items = num_items
        self.mask_id = num_items + 1
        self.loss_chunk_size = loss_chunk_size
        # 0 = padding, 1..N = items, N+1 = MASK. Reserve one position for inference.
        self.item_embedding = nn.Embedding(num_items + 2, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen + 1, hidden_units)
        self.input_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.ModuleList([
            BidirectionalBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_units, hidden_units), nn.GELU(),
            nn.LayerNorm(hidden_units, eps=1e-12),
        )
        self.output_bias = nn.Parameter(torch.zeros(num_items))
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, sequences):
        positions = torch.arange(sequences.size(1), device=sequences.device)
        hidden = self.item_embedding(sequences) + self.position_embedding(positions)
        hidden = self.dropout(self.input_norm(hidden))
        # Only padding is masked; future items remain visible for the Cloze task.
        for block in self.encoder:
            hidden = block(hidden, sequences.ne(0))
        return hidden

    def scores(self, hidden):
        return F.linear(self.output_transform(hidden),
                        self.item_embedding.weight[1:self.num_items + 1], self.output_bias)

    def _chunk_loss(self, hidden, labels):
        return F.cross_entropy(self.scores(hidden).float(), labels, reduction="sum")

    def forward(self, sequences, labels=None):
        hidden = self.encode(sequences)
        if labels is None:
            return self.scores(hidden[:, -1])
        selected = labels.ne(-100)
        hidden, targets = hidden[selected], labels[selected]
        # Return sums/counts so DataParallel weights unequal numbers of masks correctly.
        loss = hidden.sum() * 0.0
        for start in range(0, len(targets), self.loss_chunk_size):
            stop = start + self.loss_chunk_size
            if self.training and torch.is_grad_enabled():
                loss = loss + checkpoint(self._chunk_loss, hidden[start:stop],
                                         targets[start:stop], use_reentrant=False)
            else:
                loss = loss + self._chunk_loss(hidden[start:stop], targets[start:stop])
        return loss.reshape(1), selected.sum().reshape(1)
