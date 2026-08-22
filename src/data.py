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


def _normalise_row(row: dict) -> tuple[str, str, int, str] | None:
    user = row.get("user_id") or row.get("user")
    item = row.get("parent_asin") or row.get("asin") or row.get("item_id")
    if not user or not item:
        return None
    # raw_review configs contain title; metadata enrichment is intentionally optional.
    text = str(row.get("title") or row.get("text") or row.get("item_title") or item)
    timestamp = int(row.get("timestamp") or 0)
    return str(user), str(item), timestamp, text


def load_amazon(subset: str, max_events: int) -> list[tuple[str, str, int, str]]:
    """Stream a bounded Amazon Reviews 2023 subset to avoid a full dataset download."""
    from datasets import load_dataset

    stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", subset, split="full", streaming=True
    )
    rows = []
    for row in stream:
        normal = _normalise_row(row)
        if normal is not None:
            rows.append(normal)
        if len(rows) >= max_events:
            break
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
