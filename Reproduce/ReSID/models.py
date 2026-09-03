"""ReSID components adapted to MediaTek's shared interaction representation.

The implementation keeps the paper pipeline intact at a practical scale:
field-aware masked auto-encoding (FAMAE), globally aligned orthogonal
quantization (GAOQ), then autoregressive semantic-ID recommendation.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def _stable_bucket(token: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_count + 1


def build_item_fields(
    item_texts: Sequence[str], text_fields: int = 4, text_buckets: int = 4096
) -> tuple[torch.Tensor, list[int]]:
    """Build deterministic categorical fields without changing catalog IDs.

    Amazon metadata has already been normalized into ``item_texts`` by ver4.
    The first field is the exact catalog item ID. Remaining fields are stable
    hashes of ordered metadata tokens and act as ReSID's side-information fields.
    """
    if text_fields < 0 or text_buckets < 2:
        raise ValueError("text_fields must be non-negative and text_buckets >= 2")
    num_items = len(item_texts)
    fields = torch.zeros((num_items + 1, text_fields + 1), dtype=torch.long)
    fields[1:, 0] = torch.arange(1, num_items + 1)
    for item_index, text in enumerate(item_texts, start=1):
        tokens = TOKEN_PATTERN.findall(str(text).lower())
        for field_index in range(text_fields):
            token = tokens[field_index] if field_index < len(tokens) else f"<empty:{field_index}>"
            fields[item_index, field_index + 1] = _stable_bucket(
                f"{field_index}:{token}", text_buckets
            )
    return fields, [num_items] + [text_buckets] * text_fields


class FieldAwareMaskedAutoEncoder(nn.Module):
    """Recommendation-native masked target encoder following ReSID FAMAE."""

    def __init__(
        self,
        item_fields: torch.Tensor,
        field_vocab_sizes: Sequence[int],
        max_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        item_candidates: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.register_buffer("item_fields", item_fields, persistent=False)
        self.field_vocab_sizes = tuple(int(value) for value in field_vocab_sizes)
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.item_candidates = item_candidates
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
            for vocab_size in self.field_vocab_sizes
        )
        self.position_embedding = nn.Embedding(max_len + 1, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, std=0.01)

    def _sample_item_candidates(self, labels: torch.Tensor) -> torch.Tensor:
        vocab_size = self.field_vocab_sizes[0]
        if self.item_candidates <= 0 or self.item_candidates >= vocab_size:
            return torch.arange(1, vocab_size + 1, device=labels.device)
        target_count = min(self.item_candidates, vocab_size)
        candidates = torch.unique(labels)
        while candidates.numel() < target_count:
            needed = target_count - candidates.numel()
            negatives = torch.randint(
                1, vocab_size + 1, (needed * 2,), device=labels.device
            )
            candidates = torch.unique(torch.cat((candidates, negatives)))
        # Keep the (small) possible overshoot so every positive is guaranteed to
        # remain present; trimming a sorted set could accidentally drop a label.
        return candidates.sort().values

    def forward(
        self,
        histories: torch.Tensor,
        targets: torch.Tensor,
        masked_fields: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_features = self.item_fields[histories]
        target_features = self.item_fields[targets].unsqueeze(1)
        features = torch.cat((history_features, target_features), dim=1).clone()
        for field_index in masked_fields:
            features[:, -1, field_index] = self.field_vocab_sizes[field_index] + 1

        encoded = 0.0
        for field_index, embedding in enumerate(self.embeddings):
            encoded = encoded + embedding(features[:, :, field_index])
        positions = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        encoded = self.dropout(encoded + self.position_embedding(positions))
        padding_mask = features[:, :, 0].eq(0)
        output = self.encoder(encoded, src_key_padding_mask=padding_mask)
        return output[:, -1], self.item_fields[targets]

    def loss(
        self,
        histories: torch.Tensor,
        targets: torch.Tensor,
        masked_fields: Sequence[int],
    ) -> torch.Tensor:
        representation, target_fields = self(histories, targets, masked_fields)
        losses = []
        scale = math.sqrt(self.hidden_size)
        for field_index in masked_fields:
            labels = target_fields[:, field_index]
            if field_index == 0:
                candidates = self._sample_item_candidates(labels)
                weights = self.embeddings[field_index](candidates)
                logits = F.normalize(representation, dim=-1) @ F.normalize(weights, dim=-1).T
                logits = logits * scale
                candidate_labels = torch.searchsorted(candidates, labels.contiguous())
                losses.append(F.cross_entropy(logits, candidate_labels))
            else:
                vocab_size = self.field_vocab_sizes[field_index]
                weights = self.embeddings[field_index].weight[1 : vocab_size + 1]
                logits = F.normalize(representation, dim=-1) @ F.normalize(weights, dim=-1).T
                losses.append(F.cross_entropy(logits * scale, labels - 1))
        return torch.stack(losses).sum()

    @torch.inference_mode()
    def item_representations(self, batch_size: int = 8192) -> torch.Tensor:
        """Fuse learned field embeddings for every real catalog item."""
        self.eval()
        chunks = []
        for start in range(1, self.item_fields.size(0), batch_size):
            fields = self.item_fields[start : start + batch_size]
            fused = 0.0
            for field_index, embedding in enumerate(self.embeddings):
                fused = fused + embedding(fields[:, field_index])
            chunks.append(F.normalize(fused, dim=-1).cpu())
        return torch.cat(chunks, dim=0)


def _cluster(data: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    clusters = max(1, min(int(clusters), len(data)))
    if clusters == 1:
        return np.zeros(len(data), dtype=np.int64), data.mean(axis=0, keepdims=True)
    if len(data) > 100_000:
        estimator = MiniBatchKMeans(
            n_clusters=clusters,
            batch_size=4096,
            max_iter=100,
            n_init=3,
            random_state=seed,
        )
    else:
        estimator = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    labels = estimator.fit_predict(data)
    return labels.astype(np.int64), estimator.cluster_centers_


def _orthogonal_anchors(count: int, dimension: int, seed: int) -> np.ndarray:
    if count > dimension:
        raise ValueError(
            f"GAOQ codebook-2 size ({count}) cannot exceed embedding dimension ({dimension})"
        )
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, count))
    q, _ = np.linalg.qr(matrix)
    return q.T.astype(np.float32)


def globally_aligned_orthogonal_quantization(
    embeddings: torch.Tensor | np.ndarray,
    codebook1_size: int,
    codebook2_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Run ReSID GAOQ and return unique, one-based three-level SIDs."""
    from scipy.optimize import linear_sum_assignment

    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or not len(x):
        raise ValueError("embeddings must be a non-empty [items, dimensions] matrix")
    codebook1_size = min(codebook1_size, len(x))
    codebook2_size = min(codebook2_size, x.shape[1])
    level1, level1_centers = _cluster(x, codebook1_size, seed)
    anchors = _orthogonal_anchors(codebook2_size, x.shape[1], seed)
    level2 = np.zeros(len(x), dtype=np.int64)

    for first_code in np.unique(level1):
        indices = np.flatnonzero(level1 == first_code)
        local_labels, local_centers = _cluster(x[indices], codebook2_size, seed)
        residual_centers = local_centers - level1_centers[first_code]
        residual_centers /= np.maximum(
            np.linalg.norm(residual_centers, axis=1, keepdims=True), 1e-12
        )
        similarities = residual_centers @ anchors.T
        rows, columns = linear_sum_assignment(-similarities)
        mapping = np.zeros(len(local_centers), dtype=np.int64)
        mapping[rows] = columns
        level2[indices] = mapping[local_labels]

    # The third level resolves every remaining collision within a global (c1,c2)
    # prefix. IDs are local to a prefix, as in the reference GAOQ implementation.
    level3 = np.zeros(len(x), dtype=np.int64)
    next_id: dict[tuple[int, int], int] = {}
    for index, prefix in enumerate(zip(level1.tolist(), level2.tolist())):
        level3[index] = next_id.get(prefix, 0)
        next_id[prefix] = level3[index] + 1
    codes = np.stack((level1 + 1, level2 + 1, level3 + 1), axis=1)
    if len(np.unique(codes, axis=0)) != len(codes):
        raise RuntimeError("GAOQ failed to assign a unique semantic ID to every item")
    return torch.from_numpy(codes).long()


class GenerativeSIDRecommender(nn.Module):
    """Autoregressively predicts the three GAOQ codes of the next item."""

    def __init__(
        self,
        item_codes: torch.Tensor,
        max_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if item_codes.ndim != 2 or item_codes.size(1) != 3:
            raise ValueError("item_codes must have shape [num_items, 3]")
        code_sizes = tuple(int(item_codes[:, level].max()) for level in range(3))
        offsets = (0, code_sizes[0], code_sizes[0] + code_sizes[1])
        codes_with_padding = torch.zeros(
            (item_codes.size(0) + 1, 3), dtype=torch.long
        )
        codes_with_padding[1:] = item_codes
        zero_codes = item_codes - 1
        pairs, pair_inverse = torch.unique(
            zero_codes[:, :2], dim=0, return_inverse=True
        )
        self.register_buffer("item_codes", item_codes.clone())
        self.register_buffer("codes_with_padding", codes_with_padding)
        self.register_buffer("zero_codes", zero_codes)
        self.register_buffer("pairs", pairs)
        self.register_buffer("pair_inverse", pair_inverse)
        self.code_sizes = code_sizes
        self.offsets = offsets
        self.max_len = max_len
        self.hidden_size = hidden_size
        total_vocabulary = sum(code_sizes)
        self.start_token = total_vocabulary + 1
        self.token_embedding = nn.Embedding(
            total_vocabulary + 2, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_len * 3, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )
        self.decoder = nn.GRUCell(hidden_size, hidden_size)
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden_size, size) for size in code_sizes
        )

    def history_tokens(self, shifted_item_histories: torch.Tensor) -> torch.Tensor:
        codes = self.codes_with_padding[shifted_item_histories]
        offsets = torch.tensor(self.offsets, device=codes.device).view(1, 1, 3)
        tokens = torch.where(codes.eq(0), codes, codes + offsets)
        return tokens.reshape(tokens.size(0), -1)

    def encode(self, shifted_item_histories: torch.Tensor) -> torch.Tensor:
        tokens = self.history_tokens(shifted_item_histories)
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=tokens.eq(0))
        return hidden[:, -1]

    def forward(
        self, shifted_item_histories: torch.Tensor, target_item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        context = self.encode(shifted_item_histories)
        target_codes = self.codes_with_padding[target_item_ids]
        decoder_input = self.token_embedding(
            torch.full(
                (context.size(0),), self.start_token, dtype=torch.long, device=context.device
            )
        )
        hidden = self.decoder(decoder_input, context)
        logits = []
        for level, head in enumerate(self.output_heads):
            logits.append(head(hidden))
            if level < 2:
                token = target_codes[:, level] + self.offsets[level]
                hidden = self.decoder(self.token_embedding(token), hidden)
        return tuple(logits)

    def loss(
        self, shifted_item_histories: torch.Tensor, target_item_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = self(shifted_item_histories, target_item_ids)
        labels = self.codes_with_padding[target_item_ids] - 1
        return sum(
            F.cross_entropy(level_logits, labels[:, level])
            for level, level_logits in enumerate(logits)
        )

    @torch.inference_mode()
    def score_all_items(
        self, shifted_item_histories: torch.Tensor, pair_chunk_size: int = 512
    ) -> torch.Tensor:
        """Score the exact full catalog by semantic-ID log likelihood."""
        context = self.encode(shifted_item_histories)
        batch_size = context.size(0)
        start = self.token_embedding(
            torch.full(
                (batch_size,), self.start_token, dtype=torch.long, device=context.device
            )
        )
        hidden0 = self.decoder(start, context)
        log_prob1 = F.log_softmax(self.output_heads[0](hidden0).float(), dim=-1)

        first_tokens = torch.arange(
            1, self.code_sizes[0] + 1, device=context.device
        )
        hidden1 = self.decoder(
            self.token_embedding(first_tokens)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(-1, self.hidden_size),
            hidden0.unsqueeze(1)
            .expand(-1, self.code_sizes[0], -1)
            .reshape(-1, self.hidden_size),
        ).reshape(batch_size, self.code_sizes[0], self.hidden_size)
        log_prob2 = F.log_softmax(self.output_heads[1](hidden1).float(), dim=-1)

        first = self.zero_codes[:, 0]
        second = self.zero_codes[:, 1]
        third = self.zero_codes[:, 2]
        scores = log_prob1[:, first] + log_prob2[:, first, second]

        for pair_start in range(0, len(self.pairs), pair_chunk_size):
            pair_end = min(pair_start + pair_chunk_size, len(self.pairs))
            pair = self.pairs[pair_start:pair_end]
            pair_hidden1 = hidden1[:, pair[:, 0]]
            second_tokens = pair[:, 1] + 1 + self.offsets[1]
            hidden2 = self.decoder(
                self.token_embedding(second_tokens)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .reshape(-1, self.hidden_size),
                pair_hidden1.reshape(-1, self.hidden_size),
            ).reshape(batch_size, len(pair), self.hidden_size)
            log_prob3 = F.log_softmax(self.output_heads[2](hidden2).float(), dim=-1)
            item_mask = (self.pair_inverse >= pair_start) & (
                self.pair_inverse < pair_end
            )
            item_indices = torch.nonzero(item_mask, as_tuple=False).squeeze(1)
            local_pairs = self.pair_inverse[item_indices] - pair_start
            scores[:, item_indices] += log_prob3[
                :, local_pairs, third[item_indices]
            ]
        return scores
