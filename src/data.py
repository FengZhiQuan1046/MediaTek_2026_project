from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable



@dataclass
class InteractionData:
    """Integer-indexed implicit interactions and text associated with every item."""

    train_by_user: dict[int, list[int]]
    valid_target: dict[int, int]
    test_target: dict[int, int]
    item_texts: list[str]
    num_users: int
    num_items: int


def _normalise_row(row: dict, item_texts: dict[str, str] | None = None) -> tuple[str, str, int, str] | None:
    user = row.get("user_id") or row.get("user")
    item = row.get("parent_asin") or row.get("asin") or row.get("item_id")
    if not user or not item:
        return None
    # Use item metadata when available; a review title/text is only a fallback.
    text = (item_texts or {}).get(str(item)) or str(row.get("title") or row.get("text") or item)
    timestamp = int(row.get("timestamp") or 0)
    return str(user), str(item), timestamp, text


def _metadata_text(row: dict) -> str:
    """Make a compact LLM input from the Amazon item-metadata fields."""
    parts = [row.get("title"), row.get("subtitle"), row.get("main_category")]
    parts.extend(row.get("features") or [])
    parts.extend(row.get("description") or [])
    return " ".join(str(part).strip() for part in parts if part and str(part).strip())


def load_amazon(subset: str, max_events: int | None, cache_dir: str | None = None) -> list[tuple[str, str, int, str]]:
    """Load reviews and item metadata from matching Amazon Reviews 2023 configs.

    The raw review configuration has names such as ``raw_review_All_Beauty``;
    its matching metadata configuration is ``raw_meta_All_Beauty``.
    """
    import datasets
    from datasets import load_dataset

    major_version = int(datasets.__version__.split(".", 1)[0])
    if major_version >= 4:
        raise RuntimeError(
            "McAuley-Lab/Amazon-Reviews-2023 uses a dataset loading script, which datasets 4.x no "
            "longer supports. Install the project requirements (datasets==3.6.0), then retry."
        )

    if not subset.startswith("raw_review_"):
        raise ValueError("--subset must start with 'raw_review_', e.g. raw_review_All_Beauty")
    # Keep the DatasetDict form used in the official review-loading example.
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", subset,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    review_split = reviews["full"]
    limited_reviews = review_split if max_events is None else review_split.select(range(min(max_events, len(review_split))))

    metadata_subset = subset.replace("raw_review_", "raw_meta_", 1)
    metadata = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", metadata_subset, split="full",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    required_items = {str(row["parent_asin"]) for row in limited_reviews if row.get("parent_asin")}
    item_texts = {
        str(row["parent_asin"]): text
        for row in metadata
        if row.get("parent_asin") in required_items
        if (text := _metadata_text(row))
    }

    rows = []
    for row in limited_reviews:
        normal = _normalise_row(row, item_texts)
        if normal is not None:
            rows.append(normal)
    if not rows:
        raise RuntimeError(f"No usable user/item rows found in subset {subset!r}.")
    return rows


def synthetic_events() -> list[tuple[str, str, int, str]]:
    """A deterministic tiny dataset for a complete local smoke test."""
    rows = []
    titles = ["skin cleanser", "vitamin serum", "face cream", "hair shampoo", "body lotion"]
    for user in range(24):
        for step in range(5):
            item = (user + step * 2) % len(titles)
            rows.append((f"u{user}", f"i{item}", step + user * 10, titles[item]))
    return rows


def build_data(events: Iterable[tuple[str, str, int, str]], min_user_events: int = 3) -> InteractionData:
    per_user: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for user, item, ts, text in events:
        per_user[user].append((ts, item, text))
    per_user = {u: sorted(xs) for u, xs in per_user.items() if len(xs) >= min_user_events}
    if not per_user:
        raise RuntimeError("No users have enough events after filtering.")

    user_map = {u: idx for idx, u in enumerate(sorted(per_user))}
    item_map: dict[str, int] = {}
    item_text: dict[str, str] = {}
    for entries in per_user.values():
        for _, item, text in entries:
            item_map.setdefault(item, len(item_map))
            item_text.setdefault(item, text)

    train_by_user, valid, test = {}, {}, {}
    for user, entries in per_user.items():
        ids = [item_map[item] for _, item, _ in entries]
        uid = user_map[user]
        train_by_user[uid] = ids[:-2]
        valid[uid], test[uid] = ids[-2], ids[-1]
    item_texts = [""] * len(item_map)
    for item, iid in item_map.items():
        item_texts[iid] = item_text[item]
    return InteractionData(train_by_user, valid, test, item_texts, len(user_map), len(item_map))


def edge_index(data: InteractionData) -> torch.Tensor:
    """Return directed edges, with users offset by zero and items by num_users."""
    import torch

    src, dst = [], []
    for user, history in data.train_by_user.items():
        for item in set(history):
            src.extend((user, data.num_users + item))
            dst.extend((data.num_users + item, user))
    return torch.tensor([src, dst], dtype=torch.long)
