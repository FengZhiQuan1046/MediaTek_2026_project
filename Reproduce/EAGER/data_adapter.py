"""Convert the exact ver4 chronological split to EAGER tensors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent
VER4_ROOT = ROOT.parents[1] / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data import MIN_INTERACTIONS  # noqa: E402
from src.data_mamba_rl import load_recommendation_data  # noqa: E402


def cache_path(args) -> Path:
    identity = {
        "dataset": args.dataset, "data_path": None, "max_events": args.max_events,
        "min_rating": args.min_rating, "min_interactions": MIN_INTERACTIONS,
        "schema_version": 2,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    safe = args.dataset.replace(":", "_").replace("/", "_")
    return Path(args.cache_dir) / "mamba_multi_agent_data" / f"{safe}_{digest}.pkl"


def load_data(args, logger):
    artifact = cache_path(args)
    if artifact.exists():
        with artifact.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", artifact)
    else:
        data = load_recommendation_data(
            args.dataset, args.data_path, args.cache_dir, args.max_events, args.min_rating
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(artifact)
        logger.info("INTERACTION_CACHE miss_saved path=%s", artifact)
    return data, artifact


def padded(history, maxlen, padding_id):
    history = list(history)[-maxlen:]
    return [padding_id] * (maxlen - len(history)) + history


def training_tensors(data, maxlen=19, min_history=4, maximum=0, seed=2024):
    records = []
    for sequence in data.train_by_user.values():
        for end in range(min_history, len(sequence)):
            records.append((padded(sequence[:end], maxlen, data.num_items), sequence[end]))
    if maximum and len(records) > maximum:
        random.Random(seed).shuffle(records)
        records = records[:maximum]
    if not records:
        raise RuntimeError("EAGER needs at least one training prefix with min_history events")
    return (
        torch.tensor([row[0] for row in records], dtype=torch.long),
        torch.tensor([row[1] for row in records], dtype=torch.long),
    )


def evaluation_tensors(data, split, maxlen=19, user_limit=0):
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    if user_limit and len(users) > user_limit:
        users = [users[index * len(users) // user_limit] for index in range(user_limit)]
    histories, labels, raw = [], [], []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        raw.append(history)
        histories.append(padded(history, maxlen, data.num_items))
        labels.append(targets[user])
    return users, torch.tensor(histories), torch.tensor(labels), raw
