"""Mamba4Rec adapter for the exact MediaTek ver4 data boundary."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch


MAMBA4REC_ROOT = Path(__file__).resolve().parent
MEDIATEK_ROOT = MAMBA4REC_ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data import MIN_INTERACTIONS  # noqa: E402
from src.data_mamba_rl import load_recommendation_data  # noqa: E402


def interaction_cache_path(args) -> Path:
    identity = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "max_events": args.max_events,
        "min_rating": args.min_rating,
        "min_interactions": MIN_INTERACTIONS,
        "schema_version": 2,
    }
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    return Path(args.cache_dir) / "mamba_multi_agent_data" / f"{safe_dataset}_{digest}.pkl"


def load_data_cached(args, logger: logging.Logger):
    artifact = interaction_cache_path(args)
    if artifact.exists() and not args.refresh_data_cache:
        with artifact.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", artifact)
        return data, artifact
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


def pad_sequences(histories: list[list[int]], maxlen: int, device: str):
    clipped_histories = [history[-maxlen:] for history in histories]
    lengths = torch.tensor([len(history) for history in clipped_histories], device=device)
    width = max(int(lengths.max()), 1)
    sequences = torch.zeros((len(histories), width), dtype=torch.long, device=device)
    for row, clipped in enumerate(clipped_histories):
        if clipped:
            sequences[row, :len(clipped)] = torch.tensor(
                [item + 1 for item in clipped], dtype=torch.long, device=device
            )
    return sequences, lengths


def sample_prefix_batch(data, batch_size: int, maxlen: int, rng: random.Random, device: str):
    """Sample CE next-item examples only from ver4's training partition."""
    eligible = [user for user, history in data.train_by_user.items() if len(history) > 1]
    if not eligible:
        raise RuntimeError("Mamba4Rec needs a user with at least two training interactions")
    histories = []
    targets = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        sequence = data.train_by_user[rng.choice(eligible)]
        target_position = rng.randrange(1, len(sequence))
        histories.append(sequence[:target_position])
        targets[row] = sequence[target_position]
    sequences, lengths = pad_sequences(histories, maxlen, device)
    return sequences, lengths, torch.from_numpy(targets).to(device)


def evaluation_batch(data, users: list[int], split: str, maxlen: int, device: str):
    if split not in {"valid", "test"}:
        raise ValueError(f"split must be valid or test, got {split!r}")
    histories = []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        histories.append(history[-maxlen:])
    sequences, lengths = pad_sequences(histories, maxlen, device)
    return sequences, lengths, histories


def mask_seen_items(scores: torch.Tensor, histories: list[list[int]], gold: torch.Tensor):
    for row, history in enumerate(histories):
        seen = set(history) - {int(gold[row])}
        if seen:
            scores[row, list(seen)] = -torch.inf
    return scores


def ranking_metrics(scores: torch.Tensor, gold: torch.Tensor):
    ranks = (scores >= scores.gather(1, gold[:, None])).sum(1)
    totals = {}
    for cutoff in (5, 10):
        hits = ranks <= cutoff
        count = float(hits.sum().item())
        totals[f"recall@{cutoff}"] = count
        totals[f"hit@{cutoff}"] = count
        totals[f"ndcg@{cutoff}"] = float(torch.where(
            hits, 1.0 / torch.log2(ranks.float() + 1.0),
            torch.zeros_like(ranks, dtype=torch.float),
        ).sum().item())
    return totals
