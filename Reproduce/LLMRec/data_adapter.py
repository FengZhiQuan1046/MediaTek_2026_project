"""Adapt exact ver4 InteractionData to the tensors consumed by LLMRec."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import random
import re
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent
MEDIATEK_ROOT = ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data import MIN_INTERACTIONS  # noqa: E402
from src.data_mamba_rl import load_recommendation_data  # noqa: E402


def cache_path(args):
    identity = {
        "dataset": args.dataset, "data_path": None, "max_events": args.max_events,
        "min_rating": args.min_rating, "min_interactions": MIN_INTERACTIONS,
        "schema_version": 2,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    safe = args.dataset.replace(":", "_").replace("/", "_")
    return Path(args.cache_dir) / "mamba_multi_agent_data" / f"{safe}_{digest}.pkl"


def load_data(args, logger):
    path = cache_path(args)
    if path.exists():
        with path.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", path)
    else:
        data = load_recommendation_data(
            args.dataset, None, args.cache_dir, args.max_events, args.min_rating
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        logger.info("INTERACTION_CACHE miss_saved path=%s", path)
    return data, path


def hashed_text_features(texts, dimension=64):
    """Bounded-memory deterministic metadata features for Amazon items."""
    result = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())[:256]:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            result[row, value % dimension] += 1.0 if value >> 63 else -1.0
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(norms, 1.0)
    return result


class LLMRecData:
    def __init__(self, data, batch_size, feature_dim, seed):
        self.raw = data
        self.batch_size = batch_size
        self.n_users, self.n_items = data.num_users, data.num_items
        self.train_items = data.train_by_user
        self.val_set = {user: [item] for user, item in data.valid_target.items()}
        self.test_set = {user: [item] for user, item in data.test_target.items()}
        self.exist_users = [u for u, items in self.train_items.items() if items]
        self.n_train = sum(map(len, self.train_items.values()))
        self.rng = random.Random(seed)

        rows, cols = [], []
        for user, items in self.train_items.items():
            rows.extend([user] * len(items)); cols.extend(items)
        values = np.ones(len(rows), dtype=np.float32)
        self.train_matrix = sp.csr_matrix(
            (values, (rows, cols)), shape=(self.n_users, self.n_items)
        )
        self.text_features = hashed_text_features(data.item_texts, feature_dim)
        # Amazon image content is not part of ver4's normalized boundary.
        self.image_features = np.zeros((self.n_items, 1), dtype=np.float32)
        counts = np.maximum(np.asarray(self.train_matrix.sum(1)), 1.0)
        self.user_features = (self.train_matrix @ self.text_features) / counts
        self.item_attributes = {"title": self.text_features.copy()}
        self.augmented_pairs = {}
        for user in self.exist_users:
            positive = self.rng.choice(self.train_items[user])
            seen = set(self.train_items[user])
            negative = self.rng.randrange(self.n_items)
            while negative in seen:
                negative = self.rng.randrange(self.n_items)
            self.augmented_pairs[user] = (positive, negative)

    def sample(self):
        users = (
            self.rng.sample(self.exist_users, self.batch_size)
            if self.batch_size <= len(self.exist_users)
            else [self.rng.choice(self.exist_users) for _ in range(self.batch_size)]
        )
        positives, negatives = [], []
        for user in users:
            positives.append(self.rng.choice(self.train_items[user]))
            seen = set(self.train_items[user])
            negative = self.rng.randrange(self.n_items)
            while negative in seen:
                negative = self.rng.randrange(self.n_items)
            negatives.append(negative)
        return users, positives, negatives


def normalized_graph(matrix, device):
    row_sum = np.asarray(matrix.sum(1)).ravel()
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    normalized = sp.diags(inverse) @ matrix
    coo = normalized.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()
