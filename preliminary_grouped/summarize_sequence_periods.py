"""Summarize test-history lengths and 30-day period counts for eight datasets."""
from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "ver4"))  # Needed to read InteractionData caches.
OUTPUT = HERE / "outputs" / "eight_dataset_sequence_period_statistics.json"
AMAZON = ("All_Beauty", "Baby_Products", "Sports_and_Outdoors", "Toys_and_Games")
ADDITIONAL = ("Steam", "MovieLens1M", "Yelp19", "Yelp23")


def statistics(lengths, periods):
    assert lengths and len(lengths) == len(periods)
    assert min(lengths) >= 1 and min(periods) >= 1
    return {
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "min_period_count": min(periods),
        "max_period_count": max(periods),
        "mean_period_count": sum(periods) / len(periods),
    }


def amazon_dataset(name):
    report_path = HERE / "outputs" / "amazon_group_lengths" / f"{name}.json"
    report = json.loads(report_path.read_text())
    source = Path(report["provenance"]["figure_sources"])
    positions = source / report["dataset"].replace(":", "_") / "alignment_per_position.csv"
    capped_by_user = {}
    with positions.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "test":
                continue
            user, length = int(row["user"]), int(row["history_length"])
            if user in capped_by_user:
                assert capped_by_user[user] == length
            capped_by_user[user] = length

    capped_report = report["figure_cohorts"]["figure1_alignment_and_figure2_preference_all_available"]
    full_report = report["full_untruncated_observed_history_reference"]
    users = sorted(capped_by_user)
    assert len(users) == capped_report["users"] == full_report["users"]
    cache = Path(report["provenance"]["preliminary_cache"])
    with cache.open("rb") as stream:
        data = pickle.load(stream)
    full_lengths = [len(data.train_by_user[user]) + 1 for user in users]
    assert all(user in data.valid_target and user in data.test_target for user in users)
    capped_lengths = [capped_by_user[user] for user in users]
    assert all(min(length, 100) == capped_by_user[user] for user, length in zip(users, full_lengths))
    assert sum(capped_lengths) == capped_report["dated_interactions"] + capped_report["missing_timestamp_interactions"]
    assert sum(full_lengths) == full_report["dated_interactions"] + full_report["missing_timestamp_interactions"]

    def with_periods(lengths, period_report):
        period = period_report["groups_per_user"]
        assert period["count"] == len(lengths)
        assert abs(period["mean"] * len(lengths) - period_report["groups"]) < 1e-6
        return {
            "min_sequence_length": min(lengths),
            "max_sequence_length": max(lengths),
            "mean_sequence_length": sum(lengths) / len(lengths),
            "min_period_count": int(period["min"]),
            "max_period_count": int(period["max"]),
            "mean_period_count": period["mean"],
        }

    return {
        "dataset_id": report["dataset"], "users": len(users),
        "plotted_capped_100": with_periods(capped_lengths, capped_report),
        "full_observed_history": with_periods(full_lengths, full_report),
        "sources": {"period_report": str(report_path), "alignment_positions": str(positions),
                    "interaction_cache": str(cache)},
    }


def additional_dataset(name):
    root = HERE / "outputs" / "additional_dated_20261023" / name
    counts = root / "intermediate" / "per_user_group_counts.csv.gz"
    report_path = root / "group_count_summary.json"
    report = json.loads(report_path.read_text())
    with gzip.open(counts, "rt", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["in_sampled_cohort"] == "True"]
    capped_lengths = [int(row["capped_interactions"]) for row in rows]
    capped_periods = [int(row["capped_groups"]) for row in rows]
    full_lengths = [int(row["full_interactions"]) for row in rows]
    full_periods = [int(row["full_groups"]) for row in rows]
    reference = report["sampled_preliminary_cohort_capped"]
    assert len(rows) == reference["users"]
    assert sum(capped_lengths) == reference["total_observed_interactions"]
    assert sum(capped_periods) == reference["total_groups"]
    assert all(min(length, 100) == capped for length, capped in zip(full_lengths, capped_lengths))
    return {
        "dataset_id": name, "users": len(rows),
        "plotted_capped_100": statistics(capped_lengths, capped_periods),
        "full_observed_history": statistics(full_lengths, full_periods),
        "sources": {"per_user_counts": str(counts), "group_count_report": str(report_path)},
    }


def main():
    datasets = {name: amazon_dataset(name) for name in AMAZON}
    datasets.update({name: additional_dataset(name) for name in ADDITIONAL})
    result = {
        "scope": "Test observed history: training interactions plus the observed validation item; test target excluded",
        "cohort": "Amazon all-available test users in the original grouped figures; additional datasets' sampled preliminary test users",
        "period_rule": "Sort observed interactions by time; start a new period only when the gap between consecutive timestamps exceeds 30 days",
        "history_variants": {"plotted_capped_100": "most recent 100 observed items, grouped after truncation",
                             "full_observed_history": "all observed items before the test target"},
        "datasets": datasets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
