"""Checkpoint-free, hypothesis-testing preliminaries for the fixed ver4 split.

This deliberately evaluates proxies, not an untrained imitation of ver4's
learned preference or selective-state branches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer


HERE = Path(__file__).resolve().parent
VER4 = HERE.parent / "ver4"
if str(VER4) not in sys.path:
    sys.path.insert(0, str(VER4))  # Needed to unpickle ver4.src.data.InteractionData.
from src.data import MIN_INTERACTIONS  # noqa: E402

LAG_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16),
            (17, 32), (33, 64), (65, 100))
FIXED_K = (8, 16, 32)
METRICS = ("raw_cosine", "random_cosine", "excess_cosine",
           "raw_cosine_z", "random_cosine_z", "excess_cosine_z",
           "raw_cluster_match", "random_cluster_match", "excess_cluster_match")


def bootstrap_mean(values, repetitions, seed):
    """User-level percentile bootstrap; one value per user is passed in."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return (math.nan, math.nan, math.nan)
    mean = float(values.mean())
    if len(values) == 1 or repetitions < 1:
        return (mean, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions):
        means[index] = values[rng.integers(len(values), size=len(values))].mean()
    return (mean, *map(float, np.quantile(means, [0.025, 0.975])))


def fixed_cohort(rows, k):
    return [row for row in rows if row["history_length"] >= k]


def historical_summary(rows, lag):
    return float(np.mean([row["excess_cosine"] for row in rows if row["lag"] == lag]))


def pairwise_accuracy(positive, negatives):
    negatives = np.asarray(negatives)
    return float(((positive > negatives) + .5 * (positive == negatives)).mean())


def safe_name(dataset):
    return dataset.replace(":", "_").replace("/", "_")


def ver4_cache_path(cache_dir, dataset):
    identity = {"dataset": dataset, "data_path": None, "max_events": None,
                "min_rating": 4.0, "min_interactions": MIN_INTERACTIONS,
                "schema_version": 2}
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cache_dir) / "mamba_multi_agent_data" / f"{safe_name(dataset)}_{digest}.pkl"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty_summary():
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=HERE.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return ["unknown"]


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_mean(matrix):
    mean = np.asarray(matrix, dtype=np.float32).mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def subset_cosine_normalizer(reference_vectors, pair_count, rng):
    """Estimate a train-only within-subset cosine null for z-normalization."""
    reference_vectors = np.asarray(reference_vectors, dtype=np.float32)
    if len(reference_vectors) < 2:
        raise RuntimeError("At least two usable training items are required for normalization")
    count = min(pair_count, len(reference_vectors) * (len(reference_vectors) - 1))
    left = rng.integers(0, len(reference_vectors), size=count)
    # A non-zero cyclic offset guarantees that an item is never paired with itself.
    offset = rng.integers(1, len(reference_vectors), size=count)
    right = (left + offset) % len(reference_vectors)
    similarities = np.einsum("ij,ij->i", reference_vectors[left], reference_vectors[right])
    mean = float(similarities.mean())
    std = float(similarities.std(ddof=1)) if count > 1 else 0.0
    if not np.isfinite(std) or std <= 1e-12:
        raise RuntimeError("Within-subset cosine standard deviation is zero")
    return mean, std, int(count)


def full_catalog_cosine_normalizer(item_texts, vectorizer, pair_count, rng):
    """Estimate cosine normalization from pairs sampled over the full catalog."""
    item_count = len(item_texts)
    if item_count < 2:
        raise RuntimeError("At least two catalog items are required for normalization")
    chunks = []
    attempts = 0
    while sum(len(chunk) for chunk in chunks) < pair_count and attempts < 100:
        needed = pair_count - sum(len(chunk) for chunk in chunks)
        draw_count = min(max(needed * 2, 1024), 16384)
        left = rng.integers(0, item_count, size=draw_count)
        right = (left + rng.integers(1, item_count, size=draw_count)) % item_count
        left_vectors = vectorizer.transform([item_texts[int(item)] for item in left])
        right_vectors = vectorizer.transform([item_texts[int(item)] for item in right])
        valid = ((left_vectors.multiply(left_vectors).sum(axis=1).A1 > 0)
                 & (right_vectors.multiply(right_vectors).sum(axis=1).A1 > 0))
        similarities = left_vectors.multiply(right_vectors).sum(axis=1).A1[valid]
        if len(similarities):
            chunks.append(np.asarray(similarities[:needed], dtype=np.float32))
        attempts += 1
    if not chunks or sum(len(chunk) for chunk in chunks) < pair_count:
        raise RuntimeError("Could not sample enough usable full-catalog cosine pairs")
    similarities = np.concatenate(chunks)[:pair_count]
    std = float(similarities.std(ddof=1)) if len(similarities) > 1 else 0.0
    if not np.isfinite(std) or std <= 1e-12:
        raise RuntimeError("Full-catalog cosine standard deviation is zero")
    return float(similarities.mean()), std, int(len(similarities))


def bin_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}-{hi}"


def summarize_lags(rows, args, dataset, split):
    output = []
    for cohort in ("all_available", *(f"fixed_K{k}" for k in FIXED_K)):
        k = None if cohort == "all_available" else int(cohort[7:])
        subset = rows if k is None else fixed_cohort(rows, k)
        for lo, hi in LAG_BINS:
            # A fixed-K panel is fixed only for bins entirely within [1,K].
            if k is not None and hi > k:
                continue
            selected = [row for row in subset if lo <= row["lag"] <= hi]
            if not selected:
                continue
            # Each user contributes at most one mean for this lag bin.
            by_user = defaultdict(list)
            for row in selected:
                by_user[row["user"]].append(row)
            for metric in METRICS:
                values = [np.mean([row[metric] for row in user_rows])
                          for user_rows in by_user.values()]
                mean, lower, upper = bootstrap_mean(values, args.bootstrap, args.seed + lo + hi)
                output.append({"dataset": dataset, "split": split, "cohort": cohort,
                               "lag_bin": bin_name(lo, hi), "lag_start": lo, "lag_end": hi,
                               "metric": metric, "mean": mean, "ci_low": lower,
                               "ci_high": upper, "n_users": len(by_user),
                               "n_positions": len(selected),
                               "units": ("within_subset_z" if metric.endswith("_z") else
                                         "cosine" if "cosine" in metric else "probability_difference"),
                               "status": "exploratory_n_lt_100" if len(by_user) < 100 else "estimated"})
    return output


def summarize_old(rows, args, dataset, split):
    if not rows:
        return []
    metrics = ("old_item_auc", "old_preference_proxy_auc", "recent_margin",
               "preference_minus_item_auc", "full_margin", "full_minus_recent_margin", "old_fraction",
               "low_alignment_old_fraction", "low_alignment_old_cluster_excess")
    output = []
    for metric in metrics:
        values = [row[metric] for row in rows if math.isfinite(row[metric])]
        mean, lower, upper = bootstrap_mean(values, args.bootstrap, args.seed + len(metric))
        output.append({"dataset": dataset, "split": split, "metric": metric,
                       "mean": mean, "ci_low": lower, "ci_high": upper,
                       "n_users": len(values), "units": "pairwise_accuracy" if metric.endswith("auc") else "score_difference_or_fraction",
                       "status": "exploratory_n_lt_100" if len(values) < 100 else "estimated"})
    return output


def load_model_item_influence(training_output_dir, dataset, args):
    """Load trained-model leave-one-out rows and summarize every exact lag."""
    source = Path(training_output_dir) / safe_name(dataset) / "history_item_influence.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing trained-model history influence {source}; rerun preliminary/run.sh"
        )
    rows = []
    with source.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {
                "dataset": dataset, "split": raw["split"], "user": int(raw["user"]),
                "history_length": int(raw["history_length"]), "lag": int(raw["lag"]),
                "item": int(raw["item"]), "target": int(raw["target"]),
                "negative_count": int(raw["negative_count"]),
                "margin_with_item": float(raw["margin_with_item"]),
                "margin_without_item": float(raw["margin_without_item"]),
                "margin_influence": float(raw["margin_influence"]),
                "target_score_influence": float(raw["target_score_influence"]),
                "rank_with_item": int(raw["rank_with_item"]),
                "rank_without_item": int(raw["rank_without_item"]),
                "rank_improvement": int(raw["rank_improvement"]),
            }
            rows.append(row)
    summary = []
    for lag in sorted({row["lag"] for row in rows if row["split"] == "test"}):
        selected = [row["margin_influence"] for row in rows
                    if row["split"] == "test" and row["lag"] == lag]
        mean, low, high = bootstrap_mean(selected, args.bootstrap, args.seed + lag)
        values = np.asarray(selected, dtype=np.float64)
        summary.append({
            "dataset": dataset, "split": "test", "lag": lag,
            "mean_margin_influence": mean, "ci_low": low, "ci_high": high,
            "help_percent": float(100 * np.mean(values > 0)),
            "hurt_percent": float(100 * np.mean(values < 0)),
            "neutral_percent": float(100 * np.mean(values == 0)),
            "n_users": len(values),
        })
    return rows, summary, source


def paired_decay(rows, args, dataset, split):
    """Paired lag-1 minus lag-9:16 contrast in the same history>=16 users."""
    eligible = defaultdict(dict)
    for row in rows:
        if row["history_length"] >= 16 and row["lag"] <= 16:
            eligible[row["user"]][row["lag"]] = row
    output = []
    for metric in ("excess_cosine_z", "excess_cluster_match"):
        values = []
        for user_rows in eligible.values():
            if 1 in user_rows and all(lag in user_rows for lag in range(9, 17)):
                values.append(user_rows[1][metric] - np.mean(
                    [user_rows[lag][metric] for lag in range(9, 17)]))
        mean, low, high = bootstrap_mean(values, args.bootstrap, args.seed + len(metric))
        output.append({"dataset": dataset, "split": split, "cohort": "fixed_K16",
                       "metric": f"lag1_minus_lag9to16_{metric}", "mean": mean,
                       "ci_low": low, "ci_high": high, "n_users": len(values),
                       "status": "exploratory_n_lt_100" if len(values) < 100 else "estimated"})
    return output


def summarize_information_coverage(rows, dataset, split, max_lag):
    """How much positive next-item alignment is contained in the latest X items."""
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(row)
    curves, users = [], []
    for user, user_rows in by_user.items():
        signal_by_lag = np.zeros(max_lag, dtype=np.float64)
        for row in user_rows:
            if 1 <= row["lag"] <= max_lag:
                # Negative excess alignment is evidence against, not a non-negative
                # quantity that can be assigned a meaningful percentage share.
                signal_by_lag[row["lag"] - 1] = max(0.0, row["excess_cosine_z"])
        total = float(signal_by_lag.sum())
        if total <= 0:
            continue
        cumulative = 100.0 * np.cumsum(signal_by_lag) / total
        curve = {"dataset": dataset, "split": split, "user": user,
                 "history_length": max(row["history_length"] for row in user_rows),
                 "positive_signal_total": total}
        for threshold in (50, 80, 90):
            curve[f"items_for_{threshold}pct"] = int(np.searchsorted(
                cumulative, threshold, side="left") + 1)
        users.append(curve)
        curves.append(cumulative)
    if not curves:
        return [], users
    matrix = np.vstack(curves)
    summary = []
    for index in range(max_lag):
        values = matrix[:, index]
        summary.append({"dataset": dataset, "split": split, "recent_items": index + 1,
                        "mean_percent": float(values.mean()),
                        "median_percent": float(np.median(values)),
                        "q25_percent": float(np.quantile(values, .25)),
                        "q75_percent": float(np.quantile(values, .75)),
                        "n_users": len(values)})
    return summary, users


def sample_matches(item, blocked, pop_bins, pools, amount, rng, exclude_target=None):
    """Train-only popularity matching, with explicit nearest-bin fallback."""
    original_bin = int(pop_bins[item])
    for distance in range(5):
        candidate_bins = [b for b in range(5) if abs(b - original_bin) <= distance]
        pool = np.concatenate([pools[b] for b in candidate_bins if len(pools[b])]) if any(len(pools[b]) for b in candidate_bins) else np.empty(0, dtype=np.int32)
        if not len(pool):
            continue
        eligible = pool[~np.isin(pool, blocked)]
        if exclude_target is not None:
            eligible = eligible[eligible != exclude_target]
        if len(eligible):
            return rng.choice(eligible, size=amount, replace=len(eligible) < amount), distance
    return np.empty(0, dtype=np.int32), -1


def select_users(data, cap, seed):
    users = np.array(sorted(set(data.valid_target) & set(data.test_target)), dtype=np.int64)
    if cap and cap < len(users):
        users = np.sort(np.random.default_rng(seed).choice(users, cap, replace=False))
    return users


def analyse_dataset(dataset, args, output):
    source = ver4_cache_path(args.cache_dir, dataset)
    if not source.exists():
        raise FileNotFoundError(f"Missing exact ver4 cache {source}; run ver4 preprocessing first")
    with source.open("rb") as stream:
        data = pickle.load(stream)
    users = select_users(data, args.max_users, args.seed)
    print(f"{dataset}: {len(users)} sampled / {len(data.valid_target)} eligible users; {data.num_items} items", flush=True)
    train_counts = np.zeros(data.num_items, dtype=np.int64)
    for history in data.train_by_user.values():
        np.add.at(train_counts, history, 1)
    train_ids = np.flatnonzero(train_counts).astype(np.int32)
    rng = np.random.default_rng(args.seed + int(hashlib.sha1(dataset.encode()).hexdigest()[:8], 16))
    reference_ids = np.sort(rng.choice(train_ids, min(len(train_ids), args.reference_items), replace=False))
    log_counts = np.log1p(train_counts)
    edges = np.quantile(log_counts[train_ids], [0.2, 0.4, 0.6, 0.8])
    pop_bins = np.searchsorted(edges, log_counts, side="right").astype(np.int8)
    pools = [reference_ids[pop_bins[reference_ids] == index] for index in range(5)]

    needed = set(map(int, reference_ids))
    for user in users:
        train = data.train_by_user[int(user)]
        needed.update(train[-args.max_lag:])
        needed.add(data.valid_target[int(user)])
        needed.add(data.test_target[int(user)])
    all_ids = np.array(sorted(needed), dtype=np.int32)
    row_of = {int(item): index for index, item in enumerate(all_ids)}
    vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                                   alternate_sign=False, norm="l2", stop_words="english")
    vectors = np.empty((len(all_ids), 1024), dtype=np.float32)
    for start in range(0, len(all_ids), 1024):
        batch = all_ids[start:start + 1024]
        vectors[start:start + len(batch)] = vectorizer.transform(
            [data.item_texts[int(item)] for item in batch]
        ).astype(np.float32).toarray()
    usable = np.linalg.norm(vectors, axis=1) > 0
    fit_ids = reference_ids[usable[[row_of[int(item)] for item in reference_ids]]]
    cosine_mean, cosine_std, cosine_pair_count = full_catalog_cosine_normalizer(
        data.item_texts, vectorizer, args.normalization_pairs, rng)
    fit_ids = rng.choice(fit_ids, min(len(fit_ids), args.cluster_items), replace=False)
    n_clusters = min(args.clusters, len(fit_ids))
    if n_clusters < 2:
        raise RuntimeError(f"Not enough training items with text for {dataset}")
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed,
                               batch_size=min(1024, len(fit_ids)), n_init=3,
                               max_iter=100)
    clusterer.fit(vectors[[row_of[int(item)] for item in fit_ids]])
    clusters = clusterer.predict(vectors)
    centres = clusterer.cluster_centers_.astype(np.float32)
    centre_norm = np.linalg.norm(centres, axis=1, keepdims=True)
    centres /= np.maximum(centre_norm, 1e-12)
    fallback = Counter()
    target_in_random = 0
    all_lag_rows, all_old_rows, summary_rows, old_summary, contrasts = [], [], [], [], []
    coverage_summary, coverage_users = [], []
    for split in ("valid", "test"):
        lag_rows, old_rows = [], []
        for user in users:
            user = int(user)
            history = list(data.train_by_user[user])
            if split == "test":
                history.append(data.valid_target[user])
            history = history[-args.max_lag:]
            target = int(data.valid_target[user] if split == "valid" else data.test_target[user])
            target_row = row_of[target]
            if not history or not usable[target_row]:
                continue
            target_vector = vectors[target_row]
            target_cluster = int(clusters[target_row])
            history_rows = np.array([row_of[item] for item in history], dtype=np.int32)
            blocked = np.array(history, dtype=np.int32)
            user_lags = []
            for lag, item in enumerate(reversed(history), 1):
                item_row = row_of[item]
                if not usable[item_row]:
                    continue
                matches, widening = sample_matches(item, blocked, pop_bins, pools,
                                                   args.random_matches, rng)
                fallback[widening] += 1
                if not len(matches):
                    continue
                match_rows = np.array([row_of[int(x)] for x in matches], dtype=np.int32)
                match_rows = match_rows[usable[match_rows]]
                if not len(match_rows):
                    continue
                target_in_random += int(np.count_nonzero(all_ids[match_rows] == target))
                raw = float(vectors[item_row] @ target_vector)
                random_cos = float(np.mean(vectors[match_rows] @ target_vector))
                raw_z = (raw - cosine_mean) / cosine_std
                random_z = (random_cos - cosine_mean) / cosine_std
                raw_cluster = float(clusters[item_row] == target_cluster)
                random_cluster = float(np.mean(clusters[match_rows] == target_cluster))
                row = {"dataset": dataset, "split": split, "user": user,
                       "history_length": len(history), "lag": lag,
                       "item": item, "target": target, "raw_cosine": raw,
                       "random_cosine": random_cos, "excess_cosine": raw - random_cos,
                       "raw_cosine_z": raw_z, "random_cosine_z": random_z,
                       "excess_cosine_z": raw_z - random_z,
                       "raw_cluster_match": raw_cluster,
                       "random_cluster_match": random_cluster,
                       "excess_cluster_match": raw_cluster - random_cluster,
                       "matching_widened_bins": widening}
                lag_rows.append(row)
                user_lags.append(row)

            if len(history) <= args.short_window:
                continue
            old_items = history[:-args.short_window]
            old_rows_index = np.array([row_of[item] for item in old_items], dtype=np.int32)
            recent_rows_index = history_rows[-args.short_window:]
            if not usable[old_rows_index].all() or not usable[recent_rows_index].all():
                continue
            negatives, widening = sample_matches(target, blocked, pop_bins, pools,
                                                 args.random_matches, rng, exclude_target=target)
            fallback[widening] += 1
            if not len(negatives):
                continue
            negative_rows = np.array([row_of[int(item)] for item in negatives], dtype=np.int32)
            negative_rows = negative_rows[usable[negative_rows]]
            if not len(negative_rows):
                continue
            candidate_rows = np.concatenate(([target_row], negative_rows))
            candidate_vectors = vectors[candidate_rows]
            old_sim = vectors[old_rows_index] @ candidate_vectors.T
            item_scores = ((old_sim - cosine_mean) / cosine_std).max(axis=0)
            old_cluster_count = np.bincount(clusters[old_rows_index], minlength=n_clusters)
            proxy_scores = old_cluster_count[clusters[candidate_rows]] / len(old_items)
            recent_centroid = normalized_mean(vectors[recent_rows_index])
            full_centroid = normalized_mean(vectors[history_rows])
            recent_scores = (candidate_vectors @ recent_centroid - cosine_mean) / cosine_std
            full_scores = (candidate_vectors @ full_centroid - cosine_mean) / cosine_std
            recent_margin = float(recent_scores[0] - recent_scores[1:].mean())
            full_margin = float(full_scores[0] - full_scores[1:].mean())
            low_old = [row for row in user_lags if row["lag"] > args.short_window
                       and row["excess_cosine_z"] <= 0]
            old_rows.append({"dataset": dataset, "split": split, "user": user,
                             "history_length": len(history), "old_count": len(old_items),
                             "old_fraction": len(old_items) / len(history),
                             "old_item_auc": pairwise_accuracy(item_scores[0], item_scores[1:]),
                             "old_preference_proxy_auc": pairwise_accuracy(proxy_scores[0], proxy_scores[1:]),
                             "preference_minus_item_auc": pairwise_accuracy(proxy_scores[0], proxy_scores[1:]) - pairwise_accuracy(item_scores[0], item_scores[1:]),
                             "recent_margin": recent_margin, "full_margin": full_margin,
                             "full_minus_recent_margin": full_margin - recent_margin,
                             "low_alignment_old_fraction": len(low_old) / len(history),
                             "low_alignment_old_cluster_excess": float(np.mean(
                                 [row["excess_cluster_match"] for row in low_old])) if low_old else math.nan})
        all_lag_rows.extend(lag_rows)
        all_old_rows.extend(old_rows)
        summary_rows.extend(summarize_lags(lag_rows, args, dataset, split))
        old_summary.extend(summarize_old(old_rows, args, dataset, split))
        contrasts.extend(paired_decay(lag_rows, args, dataset, split))
        split_coverage, split_coverage_users = summarize_information_coverage(
            lag_rows, dataset, split, args.max_lag)
        coverage_summary.extend(split_coverage)
        coverage_users.extend(split_coverage_users)
        print(f"  {split}: {len({r['user'] for r in lag_rows})} users, {len(lag_rows)} positions, {len(old_rows)} old-history readouts", flush=True)

    prefix = output / safe_name(dataset)
    prefix.mkdir(parents=True, exist_ok=True)
    write_csv(prefix / "alignment_per_position.csv", all_lag_rows,
              ["dataset", "split", "user", "history_length", "lag", "item", "target",
               *METRICS, "matching_widened_bins"])
    write_csv(prefix / "alignment_summary.csv", summary_rows,
              ["dataset", "split", "cohort", "lag_bin", "lag_start", "lag_end",
               "metric", "mean", "ci_low", "ci_high", "n_users", "n_positions", "units", "status"])
    write_csv(prefix / "old_proxy_per_user.csv", all_old_rows,
              ["dataset", "split", "user", "history_length", "old_count",
               "old_fraction", "old_item_auc", "old_preference_proxy_auc", "preference_minus_item_auc",
               "recent_margin", "full_margin", "full_minus_recent_margin",
               "low_alignment_old_fraction", "low_alignment_old_cluster_excess"])
    write_csv(prefix / "old_proxy_summary.csv", old_summary,
              ["dataset", "split", "metric", "mean", "ci_low", "ci_high", "n_users", "units", "status"])
    write_csv(prefix / "paired_decay.csv", contrasts,
              ["dataset", "split", "cohort", "metric", "mean", "ci_low", "ci_high", "n_users", "status"])
    write_csv(prefix / "information_coverage.csv", coverage_summary,
              ["dataset", "split", "recent_items", "mean_percent", "median_percent",
               "q25_percent", "q75_percent", "n_users"])
    write_csv(prefix / "information_coverage_per_user.csv", coverage_users,
              ["dataset", "split", "user", "history_length", "positive_signal_total",
               "items_for_50pct", "items_for_80pct", "items_for_90pct"])
    diagnostics = {"dataset": dataset, "cache": str(source), "cache_sha256": file_sha256(source),
                   "eligible_users": len(data.valid_target), "sampled_users": len(users),
                   "num_items": data.num_items, "train_items": len(train_ids),
                   "reference_items": len(reference_ids), "vectorized_items": len(all_ids),
                   "cosine_normalization": {"basis": "random distinct usable item pairs sampled uniformly from the full dataset catalog",
                                            "catalog_items": data.num_items,
                                            "mean": cosine_mean, "std": cosine_std,
                                            "sampled_pairs": cosine_pair_count},
                   "information_coverage_users": {
                       split: sum(row["split"] == split for row in coverage_users)
                       for split in ("valid", "test")},
                   "text_coverage": float(usable.mean()), "cluster_fit_items": len(fit_ids),
                   "clusters": n_clusters, "matching_fallback_counts": dict(fallback),
                   "target_occurrences_in_random_references": target_in_random,
                   "timestamp_status": "unavailable_in_ver4_cache",
                   "text_provenance": "metadata_or_review_fallback_unknown_per_item"}
    (prefix / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return summary_rows, old_summary, contrasts, coverage_summary, coverage_users, diagnostics


def plot_results(summary, old_summary, influence_summary, coverage_summary, output):
    datasets = sorted({row["dataset"] for row in summary})
    names = {"amazon-all-beauty": "All Beauty", "amazon:Baby_Products": "Baby Products",
             "amazon-sports-and-outdoors": "Sports & Outdoors", "amazon-toys-and-games": "Toys & Games",
             "amazon:Musical_Instruments": "Musical Instruments", "amazon:Video_Games": "Video Games",
             "amazon:Software": "Software"}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for metric, filename, title, ylabel in (
            ("excess_cosine_z", "figure1_history_alignment.pdf",
             "How item relevance changes further back in the history",
             "Similarity to the next item above the subset baseline"),
            ("excess_cluster_match", "figure2_preference_proxy.pdf",
             "Does the user's broad preference remain visible in older items?",
             "Extra chance of sharing the next item's preference group")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for axis, cohort in zip(axes, ("all_available", "fixed_K16")):
            for dataset in datasets:
                rows = [row for row in summary if row["dataset"] == dataset and row["split"] == "test"
                        and row["metric"] == metric and row["cohort"] == cohort
                        and row["n_users"] >= 100]
                rows.sort(key=lambda row: row["lag_start"])
                if not rows:
                    continue
                x = np.array([(row["lag_start"] + row["lag_end"]) / 2 for row in rows])
                y = np.array([row["mean"] for row in rows])
                axis.plot(x, y, marker="o", label=names.get(dataset, dataset))
                axis.fill_between(x, [row["ci_low"] for row in rows],
                                  [row["ci_high"] for row in rows], alpha=.14)
            axis.axhline(0, color="black", lw=.8)
            axis.set_xlim(left=.5)
            axis.set_xlabel("How many positions back (1 = most recent item)")
            axis.set_title("All users with available history" if cohort == "all_available" else "Users with at least 16 history items")
            axis.grid(alpha=.2)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
        fig.text(.5, .12, "Higher means more useful for predicting the next item. Shaded areas are 95% confidence intervals; only points with at least 100 users are shown.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(bottom=.25, top=.84, wspace=.12)
        fig.savefig(figures / filename)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    for dataset in datasets:
        rows = [row for row in influence_summary
                if row["dataset"] == dataset and row["split"] == "test"]
        rows.sort(key=lambda row: row["lag"])
        if not rows:
            continue
        x = np.asarray([row["lag"] for row in rows])
        y = np.asarray([row["mean_margin_influence"] for row in rows])
        axis.plot(x, y, marker="o", markersize=3, label=names.get(dataset, dataset))
        axis.fill_between(x, [row["ci_low"] for row in rows],
                          [row["ci_high"] for row in rows], alpha=.14)
    axis.axhline(0, color="black", lw=.8)
    axis.set_xlim(left=.5)
    axis.set_xlabel("Item position in history (1 = most recent)")
    axis.set_ylabel("Next-item margin with item − margin after removing it")
    axis.set_title("How each history item helps or hurts the next-item prediction")
    axis.grid(alpha=.2)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    fig.text(.5, .02,
             "Above zero: keeping the item helps. Below zero: keeping it hurts. "
             "Shaded areas are 95% user-bootstrap confidence intervals.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(figures / "figure3_old_encoding_proxy.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    for dataset in datasets:
        rows = [row for row in coverage_summary
                if row["dataset"] == dataset and row["split"] == "test"]
        rows.sort(key=lambda row: row["recent_items"])
        if not rows:
            continue
        x = np.array([row["recent_items"] for row in rows])
        axis.plot(x, [row["mean_percent"] for row in rows],
                  label=names.get(dataset, dataset))
        axis.fill_between(x, [row["q25_percent"] for row in rows],
                          [row["q75_percent"] for row in rows], alpha=.12)
    axis.set_xlim(left=1)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Number of most recent items included (X)")
    axis.set_ylabel("Useful signal for the next item already covered (%)")
    axis.set_title("How much next-item information comes from the most recent X items?")
    for percentage in (50, 80, 90):
        axis.axhline(percentage, color="gray", lw=.7, ls="--", alpha=.45)
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    fig.text(.5, .02,
             "Signal means positive similarity to the next item beyond matched random items.\n"
             "Lines show the user average; shaded areas cover the middle 50% of users.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(figures / "figure4_recent_information_coverage.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--max-lag", type=int, default=100)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--random-matches", type=int, default=20)
    parser.add_argument("--reference-items", type=int, default=20000)
    parser.add_argument("--normalization-pairs", type=int, default=100000)
    parser.add_argument("--cluster-items", type=int, default=10000)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=25252)
    args = parser.parse_args()
    if min(args.max_lag, args.short_window, args.random_matches, args.reference_items,
           args.normalization_pairs,
           args.cluster_items, args.clusters) < 1 or args.max_users < 0 or args.bootstrap < 0:
        parser.error("sizes must be positive; max-users and bootstrap must be nonnegative")
    if args.short_window >= args.max_lag:
        parser.error("short-window must be smaller than max-lag")
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    if not datasets:
        parser.error("at least one dataset is required")
    run = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "mode": "full" if args.max_users == 0 else "pilot",
                "ver4_revision": git_revision(), "settings": vars(args),
                "method": "trained_model_item_deletion_plus_full_catalog_normalized_text_diagnostics_v4",
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "no_checkpoints_or_weights_saved": True,
                "limitations": ["Figure 3 uses sampled negatives rather than full-catalog reranking",
                                "Figures 1, 2, and 4 remain model-free text diagnostics",
                                "no timestamps in cached InteractionData",
                                "item_texts may include review fallback; provenance unavailable"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Output: {run}", flush=True)
    try:
        summary, old_summary, contrasts, coverage_summary, coverage_users, diagnostics = [], [], [], [], [], []
        influence_rows, influence_summary = [], []
        skipped = []
        for dataset in datasets:
            source = ver4_cache_path(args.cache_dir, dataset)
            if not source.exists():
                skipped.append({"dataset": dataset, "status": "missing_exact_ver4_cache",
                                "expected_path": str(source)})
                print(f"Skipping {dataset}: exact ver4 cache missing ({source})", flush=True)
                continue
            a, b, c, d, e, f = analyse_dataset(dataset, args, run)
            summary.extend(a)
            old_summary.extend(b)
            contrasts.extend(c)
            coverage_summary.extend(d)
            coverage_users.extend(e)
            diagnostics.append(f)
            dataset_rows, dataset_summary, influence_source = load_model_item_influence(
                args.training_output_dir, dataset, args
            )
            influence_rows.extend(dataset_rows)
            influence_summary.extend(dataset_summary)
            write_csv(run / safe_name(dataset) / "model_item_influence_per_position.csv",
                      dataset_rows,
                      ["dataset", "split", "user", "history_length", "lag", "item", "target",
                       "negative_count", "margin_with_item", "margin_without_item",
                       "margin_influence", "target_score_influence", "rank_with_item",
                       "rank_without_item", "rank_improvement"])
            diagnostics[-1]["model_item_influence_source"] = str(influence_source)
        if not diagnostics:
            raise RuntimeError("No exact ver4 dataset caches available")
        write_csv(run / "alignment_summary.csv", summary,
                  ["dataset", "split", "cohort", "lag_bin", "lag_start", "lag_end",
                   "metric", "mean", "ci_low", "ci_high", "n_users", "n_positions", "units", "status"])
        write_csv(run / "old_proxy_summary.csv", old_summary,
                  ["dataset", "split", "metric", "mean", "ci_low", "ci_high", "n_users", "units", "status"])
        write_csv(run / "paired_decay.csv", contrasts,
                  ["dataset", "split", "cohort", "metric", "mean", "ci_low", "ci_high", "n_users", "status"])
        write_csv(run / "information_coverage.csv", coverage_summary,
                  ["dataset", "split", "recent_items", "mean_percent", "median_percent",
                   "q25_percent", "q75_percent", "n_users"])
        write_csv(run / "information_coverage_per_user.csv", coverage_users,
                  ["dataset", "split", "user", "history_length", "positive_signal_total",
                   "items_for_50pct", "items_for_80pct", "items_for_90pct"])
        write_csv(run / "model_item_influence_summary.csv", influence_summary,
                  ["dataset", "split", "lag", "mean_margin_influence", "ci_low", "ci_high",
                   "help_percent", "hurt_percent", "neutral_percent", "n_users"])
        plot_results(summary, old_summary, influence_summary, coverage_summary, run)
        audit = ["# ver4 source audit", "", f"Repository revision: `{manifest['ver4_revision']}`.",
                 "", "Dirty worktree at run start (preserved, not modified by this analysis):"]
        audit.extend(f"- `{line}`" for line in git_dirty_summary())
        audit.extend(["", "## Data and model contract", "",
                      "- Input is `ver4/src/train_mamba_rl.py`'s exact schema-v2 cached `InteractionData`; no legacy digest substitution.",
                      "- `ver4/src/data.py` uses one-pass user/item minimum interactions of 5, sorts each user by timestamp, then holds out the last two items for validation and test.",
                      "- Validation history is train-only; test history is train plus the observed validation item. This analysis caps both at 100 interactions by default.",
                      "- The cache preserves integer item IDs and item text, but not the item-ID reverse mapping, timestamps or per-text metadata/review provenance.",
                      "- Item IDs are built from all filtered interactions before splitting; matching pools, popularity counts, reference cluster fit and text features selected for fit use training items only.",
                      "- `ver4/run_amazons_full_rl.sh` specifies frozen Mamba item encoding, dimension 128 online model, LoRA rank 16, long horizon 100, short window 10, and Long/Short/Preference/Graph enabled by default.",
                      "- The pretrained Mamba text encoder is an offline item-feature cache; online selective-state specialists and preference GRU are distinct modules. When Long is disabled, ver4 limits Preference to the short window too.",
                      "- Missing exact caches are prepared before this analysis by the fixed ver4 LoRA Mamba + LightGCN training path; model weights/checkpoints are not saved.",
                      "- Training graph and catalog priors are built from training histories, but they are not used directly by this text-hash diagnostic.",
                      "- Figure 3 uses leave-one-item-out scores computed from the restored best trained model before that model is released. No checkpoint or model weights are saved.",
                      "- Figures 1, 2, and 4 remain model-free text diagnostics and must not be interpreted as trained-state interventions.",
                      "- The auxiliary text-hash feature is a deterministic 1,024-dimensional word/bigram representation. Cluster labels are fitted on sampled training items only; they are *proxy* preferences.",
                      "- Random controls match training-popularity quintiles within the same Amazon dataset and exclude observed history. Exact-bin and widened-bin frequencies are in diagnostics.json.",
                      "- Every cosine is normalized with the mean and standard deviation of random distinct usable item pairs sampled uniformly from the subset's complete catalog. Normalizer values are stored in diagnostics.json.",
                      "- Day-based analysis is unavailable because timestamps were discarded by the ver4 cache. Interaction lag must not be described as elapsed time.",
                      "- User bootstrap CIs are conditional on this split, feature proxy and (when capped) sampled users; they do not measure training-seed uncertainty.", ""])
        (run / "audit.md").write_text("\n".join(audit), encoding="utf-8")

        def cell(dataset, metric, lag_bin=None, cohort="all_available"):
            if lag_bin is None:
                return next((row for row in old_summary if row["dataset"] == dataset
                             and row["split"] == "test" and row["metric"] == metric), None)
            return next((row for row in summary if row["dataset"] == dataset
                         and row["split"] == "test" and row["cohort"] == cohort
                         and row["metric"] == metric and row["lag_bin"] == lag_bin), None)

        def shown(row):
            return "NA" if row is None else f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}], n={row['n_users']}"

        def contrast(dataset, metric):
            return next((row for row in contrasts if row["dataset"] == dataset
                         and row["split"] == "test" and row["metric"] == metric), None)

        report = ["# Preliminary ver4 diagnostic", "", f"Mode: **{manifest['mode']}**; split: chronological leave-two-out.",
                  "", "These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.",
                  "Text-hash features are independent of recommendation training but item-text provenance is unknown.",
                  "The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.",
                  "No timestamps survive the ver4 cached split; all lag results are interaction-based.",
                  "", "## Coverage"]
        for d in diagnostics:
            report.append(f"- {d['dataset']}: {d['sampled_users']}/{d['eligible_users']} users; "
                          f"{d['num_items']} items; text coverage {d['text_coverage']:.3f}; "
                          f"{d['reference_items']} train-reference items.")
        for d in skipped:
            report.append(f"- {d['dataset']}: skipped; exact ver4 cache unavailable at {d['expected_path']}.")
        report.extend(["", "## Test-split estimates (95% user-bootstrap intervals)", "",
                       "These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.", ""])
        for d in diagnostics:
            dataset = d["dataset"]
            report.extend([f"### {dataset}", "",
                           f"- Fixed-K16 normalized item excess cosine, lag 1: {shown(cell(dataset, 'excess_cosine_z', '1', 'fixed_K16'))}.",
                           f"- Fixed-K16 normalized item excess cosine, lags 9-16: {shown(cell(dataset, 'excess_cosine_z', '9-16', 'fixed_K16'))}.",
                           f"- Paired lag-1 minus lag-9:16 normalized excess cosine: {shown(contrast(dataset, 'lag1_minus_lag9to16_excess_cosine_z'))}.",
                           f"- Fixed-K16 coarse preference-cluster excess, lags 9-16: {shown(cell(dataset, 'excess_cluster_match', '9-16', 'fixed_K16'))}.",
                           f"- Old-history item-level pairwise accuracy: {shown(cell(dataset, 'old_item_auc'))}.",
                           f"- Old-history preference-proxy pairwise accuracy: {shown(cell(dataset, 'old_preference_proxy_auc'))}.",
                           f"- Paired preference-minus-item accuracy: {shown(cell(dataset, 'preference_minus_item_auc'))}.",
                           f"- Fraction of capped history in old low-alignment positions: {shown(cell(dataset, 'low_alignment_old_fraction'))} (post-hoc descriptor, not noise prevalence).",
                           f"- Cluster excess among those positions: {shown(cell(dataset, 'low_alignment_old_cluster_excess'))}.",
                           f"- Full-minus-recent mean-pooling margin: {shown(cell(dataset, 'full_minus_recent_margin'))}.", ""])
        report.extend(["", "## How to read the results", "",
                       "`alignment_summary.csv` shows whether recent and older items look more like the next item than matched random items do.",
                       "If a confidence interval crosses zero, the experiment cannot say whether that history range helps.",
                       "`information_coverage.csv` shows what percentage of all positive next-item signal is already contained in the most recent X items.",
                       "`old_proxy_summary.csv` compares a direct old-item match with a broader preference summary, using the same targets and negatives.",
                       "For Figure 3, positive influence means keeping that exact item raises the trained model's true-target margin over the same sampled negatives; negative influence means it lowers the margin.",
                       "The preparatory run writes these influence statistics but no checkpoint or weights.",
                       "", "## Figures", "", "- figures/figure1_history_alignment.pdf",
                       "- figures/figure2_preference_proxy.pdf", "- figures/figure3_old_encoding_proxy.pdf",
                       "- figures/figure4_recent_information_coverage.pdf", ""])
        (run / "report.md").write_text("\n".join(report), encoding="utf-8")
        (run / "commands.txt").write_text(
            "Reproduce with the same repository and cache:\n"
            + " ".join(["bash run.sh", f"--cache-dir={args.cache_dir}",
                        f"--training-output-dir={args.training_output_dir}",
                        f"--output-dir={args.output_dir}", f"--datasets={args.datasets}",
                        f"--max-users={args.max_users}", f"--max-lag={args.max_lag}",
                        f"--short-window={args.short_window}", f"--random-matches={args.random_matches}",
                        f"--reference-items={args.reference_items}",
                        f"--normalization-pairs={args.normalization_pairs}", f"--cluster-items={args.cluster_items}",
                        f"--clusters={args.clusters}", f"--bootstrap={args.bootstrap}",
                        f"--seed={args.seed}"]) + "\n", encoding="utf-8")
        manifest["status"] = "partial_missing_datasets" if skipped else "complete"
        manifest["datasets"] = diagnostics
        manifest["skipped_datasets"] = skipped
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
