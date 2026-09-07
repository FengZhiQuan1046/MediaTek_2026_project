"""Exact ver4 data boundary and LLaRA session conversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parent
MEDIATEK_ROOT = ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data import MIN_INTERACTIONS  # noqa: E402
from src.data_mamba_rl import load_recommendation_data  # noqa: E402


def shared_cache_path(dataset, cache_dir, max_events=None, min_rating=4.0):
    identity = {"dataset": dataset, "data_path": None, "max_events": max_events,
                "min_rating": min_rating, "min_interactions": MIN_INTERACTIONS,
                "schema_version": 2}
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    safe = dataset.replace(":", "_").replace("/", "_")
    return Path(cache_dir) / "mamba_multi_agent_data" / f"{safe}_{digest}.pkl"


def load_ver4_data(dataset, cache_dir, max_events=None, min_rating=4.0):
    artifact = shared_cache_path(dataset, cache_dir, max_events, min_rating)
    if artifact.exists():
        with artifact.open("rb") as stream:
            return pickle.load(stream), artifact
    data = load_recommendation_data(dataset, None, cache_dir, max_events, min_rating)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(artifact)
    return data, artifact


def item_names(data):
    names = []
    for index, text in enumerate(data.item_texts):
        compact = " ".join(text.split())[:120]
        names.append(f"item {index}: {compact or f'Amazon product {index}'}")
    return names


def make_examples(data, split, maxlen, maximum, seed):
    examples = []
    if split == "train":
        for user, sequence in data.train_by_user.items():
            for end in range(1, len(sequence)):
                examples.append((user, sequence[max(0, end - maxlen):end], sequence[end]))
    else:
        targets = data.valid_target if split == "valid" else data.test_target
        for user, target in targets.items():
            history = list(data.train_by_user[user])
            if split == "test":
                history.append(data.valid_target[user])
            examples.append((user, history[-maxlen:], target))
    if maximum and len(examples) > maximum:
        random.Random(seed).shuffle(examples)
        examples = examples[:maximum]
    return examples
