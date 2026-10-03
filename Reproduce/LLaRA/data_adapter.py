"""Use ver4's own cache, transition selection, and evaluation user protocol."""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parent
VER4_ROOT = ROOT.parents[1] / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.train_mamba_rl import build_transitions, load_recommendation_data_cached  # noqa: E402


def load_ver4_data(dataset, cache_dir, max_events=None, min_rating=4.0):
    options = SimpleNamespace(dataset=dataset, data_path=None, cache_dir=cache_dir,
                              max_events=max_events, min_rating=min_rating,
                              refresh_data_cache=False)
    return load_recommendation_data_cached(options, logging.getLogger("llara.data"))


def item_names(data):
    names = []
    for index, value in enumerate(data.item_texts):
        compact = " ".join(value.split())[:64]
        names.append(f"item {index}: {compact or f'Amazon product {index}'}")
    return names


def make_examples(data, split, maxlen, maximum, seed):
    if split == "train":
        rows = build_transitions(data, maximum, seed)
        examples = []
        for row in rows:
            history = data.train_by_user[row.user][max(0, row.end - maxlen):row.end]
            if getattr(data, "reverse_history_input", False):
                history = history[::-1]
            examples.append((row.user, history, row.target))
        return examples
    if split not in {"valid", "test"}:
        raise ValueError(f"Unknown data split: {split}")
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    if maximum > 0 and len(users) > maximum:
        users = [users[index * len(users) // maximum] for index in range(maximum)]
    examples = []
    for user in users:
        raw = list(data.train_by_user[user])
        if split == "test":
            raw.append(data.valid_target[user])
        history = raw[-maxlen:]
        if getattr(data, "reverse_history_input", False):
            history = history[::-1]
        examples.append((user, history, targets[user]))
    return examples
