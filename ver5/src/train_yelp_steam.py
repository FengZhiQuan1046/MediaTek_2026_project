"""Add Steam support to the existing trainer without changing its source.

Steam input: CSV or JSONL (optionally .gz), with user_id/username/steamid,
item_id/product_id/app_id, timestamp/date, and optional item_text/title/name.
The Australian Steam nested ``reviews`` JSONL format is also supported.
Only item metadata is used as text; review bodies are never used as features.
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path


def timestamp(value):
    if value is None or value == "":
        raise ValueError("Steam interactions require timestamps for chronological evaluation")
    try:
        return int(float(value))
    except (ValueError, TypeError):
        text = str(value).strip().removeprefix("Posted ").rstrip(".")
        try:
            date = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            date = datetime.strptime(text, "%B %d, %Y")
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return int(date.timestamp())


def first(row, keys, default=None):
    return next((row[k] for k in keys if row.get(k) is not None and row[k] != ""), default)


def load_steam_events(data_path, max_events=None, min_rating=4.0):
    path = Path(data_path)
    if not path.is_file():
        raise ValueError("STEAM_DATA_PATH must point to a Steam CSV/JSONL file (optionally .gz)")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive")
    opener = gzip.open if path.suffix == ".gz" else open
    suffix = Path(path.stem).suffix if path.suffix == ".gz" else path.suffix
    if suffix not in {".csv", ".json", ".jsonl"}:
        raise ValueError("Steam input must be CSV or JSONL, optionally gzip-compressed")
    events = []
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        def json_rows():
            for line in stream:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield ast.literal_eval(line)
        rows = csv.DictReader(stream) if suffix == ".csv" else json_rows()
        for row in rows:
            user = first(row, ("user_id", "username", "steamid", "user"))
            for review in row.get("reviews", [row]):
                positive = first(review, ("recommended", "recommend", "is_recommended"))
                if positive is not None and str(positive).lower() in {"false", "0", "no"}:
                    continue
                rating = first(review, ("rating", "stars", "score"))
                if rating is not None and float(rating) < min_rating:
                    continue
                item = first(review, ("item_id", "product_id", "app_id", "item"))
                if user is None or item is None:
                    raise ValueError("Steam row lacks a user or item identifier")
                date = timestamp(first(review, ("timestamp", "unixReviewTime", "date", "posted")))
                text = str(first(review, ("item_text", "title", "name"), str(item)))
                events.append((str(user), str(item), date, text))
                if max_events is not None and len(events) >= max_events:
                    return events
    if not events:
        raise ValueError("No positive Steam interactions found")
    return events


def main():
    from src import train_mamba_rl as trainer
    from src.data import build_data

    original_loader = trainer.load_recommendation_data

    def loader(dataset, data_path, cache_dir, max_events=None, min_rating=4.0):
        if dataset.lower() == "steam":
            return build_data(load_steam_events(data_path, max_events, min_rating))
        return original_loader(dataset, data_path, cache_dir, max_events, min_rating)

    trainer.load_recommendation_data = loader
    trainer.main()


if __name__ == "__main__":
    main()
