"""Data boundary shared with MediaTek ver4.

The functions in this module intentionally consume ``InteractionData`` directly.
No ReSID-specific filtering, remapping, or re-splitting is allowed here.
"""
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


RESID_ROOT = Path(__file__).resolve().parent
MEDIATEK_ROOT = RESID_ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data_mamba_rl import load_recommendation_data  # noqa: E402
from src.data import MIN_INTERACTIONS  # noqa: E402


def cache_identity(args) -> dict:
    """Return the common project cache identity."""
    return {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "max_events": args.max_events,
        "min_rating": args.min_rating,
        "min_interactions": MIN_INTERACTIONS,
        "schema_version": 2,
    }


def interaction_cache_path(args) -> Path:
    identity = cache_identity(args)
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    return (
        Path(args.cache_dir)
        / "mamba_multi_agent_data"
        / f"{safe_dataset}_{digest}.pkl"
    )


def load_data_cached(args, logger: logging.Logger):
    """Load the project's normalized InteractionData and shared cache."""
    artifact = interaction_cache_path(args)
    if artifact.exists() and not args.refresh_data_cache:
        with artifact.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", artifact)
        return data, artifact

    data = load_recommendation_data(
        args.dataset,
        args.data_path,
        args.cache_dir,
        args.max_events,
        args.min_rating,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(artifact)
    logger.info("INTERACTION_CACHE miss_saved path=%s", artifact)
    return data, artifact


def evaluation_histories(
    data, users: list[int], split: str, max_len: int
) -> list[list[int]]:
    """Build histories with validation visible only when evaluating test."""
    if split not in {"valid", "test"}:
        raise ValueError(f"split must be valid or test, got {split!r}")
    histories = []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        histories.append(history[-max_len:])
    return histories


def pad_item_histories(
    histories: list[list[int]], max_len: int, device: str
) -> torch.Tensor:
    """Left-pad zero-based catalog IDs after shifting real IDs by one."""
    result = torch.zeros((len(histories), max_len), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        clipped = history[-max_len:]
        if clipped:
            result[row, -len(clipped) :] = torch.tensor(
                [item + 1 for item in clipped], dtype=torch.long, device=device
            )
    return result


def sample_prefix_batch(
    data,
    batch_size: int,
    max_len: int,
    rng: random.Random,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-item examples strictly from each user's training partition."""
    eligible = [
        user for user, history in data.train_by_user.items() if len(history) > 1
    ]
    if not eligible:
        raise RuntimeError("ReSID needs at least one user with two training interactions")

    histories: list[list[int]] = []
    targets = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        sequence = data.train_by_user[rng.choice(eligible)]
        target_position = rng.randrange(1, len(sequence))
        histories.append(sequence[:target_position])
        targets[row] = sequence[target_position] + 1
    return (
        pad_item_histories(histories, max_len, device),
        torch.from_numpy(targets).to(device, non_blocking=True),
    )


def mask_seen_items(
    scores: torch.Tensor,
    histories: list[list[int]],
    gold: torch.Tensor,
) -> torch.Tensor:
    """Apply the project's common seen-item masking rule."""
    for row, history in enumerate(histories):
        seen = set(history) - {int(gold[row])}
        if seen:
            scores[row, list(seen)] = -torch.inf
    return scores


def ranking_metrics(
    scores: torch.Tensor,
    gold: torch.Tensor,
    cutoffs: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Return metric sums using the project's conservative tie handling."""
    ranks = (scores >= scores.gather(1, gold.unsqueeze(1))).sum(dim=1)
    totals: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = ranks <= cutoff
        count = float(hits.sum().item())
        totals[f"recall@{cutoff}"] = count
        totals[f"hit@{cutoff}"] = count
        totals[f"ndcg@{cutoff}"] = float(
            torch.where(
                hits,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            )
            .sum()
            .item()
        )
    return totals
