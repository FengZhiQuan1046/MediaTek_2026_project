#!/usr/bin/env python3
"""Paired GPU-1 throughput audit for ver5 without editing ver5/src.

The only fast-path change is to gather already projected full-catalog item
vectors for each history position. The original evaluation function still
computes the states, rankings, exclusions, and every reported diagnostic.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE = HERE / "gpu1_runs" / "ours_full_beauty"
OUTPUT = HERE / "gpu1_runs" / "ours_throughput_optimized"
sys.path.insert(0, str(PROJECT))


def assert_gpu1():
    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = json.loads((BASE / "profile.json").read_text())["gpu_uuid"]
    if visible != uuid or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Bind physical GPU 1 by UUID {uuid}; visible={visible!r}")


def compare_metrics(reference, trial):
    keys = sorted(set(reference) & set(trial))
    for key in keys:
        if key.startswith(("recall@", "hit@")) and reference[key] != trial[key]:
            raise AssertionError(f"Ranking metric changed: {key}: {reference[key]} vs {trial[key]}")
        if key.startswith("ndcg@") and abs(float(reference[key]) - float(trial[key])) > 1e-7:
            raise AssertionError(f"NDCG changed beyond reduction order: {key}: {reference[key]} vs {trial[key]}")
        if key not in ("users_per_second", "scores_per_second", "sequence_length") and isinstance(reference[key], (int, float)):
            if abs(float(reference[key]) - float(trial[key])) > 1e-4:
                raise AssertionError(f"Diagnostic changed: {key}: {reference[key]} vs {trial[key]}")
    if reference.get("sequence_length") != trial.get("sequence_length"):
        raise AssertionError("Per-length ranking summary changed")


def compare_samples(reference, trial, path="samples"):
    if isinstance(reference, dict):
        if set(reference) != set(trial):
            raise AssertionError(f"Sample fields changed at {path}")
        for key in reference:
            compare_samples(reference[key], trial[key], f"{path}.{key}")
    elif isinstance(reference, list):
        if len(reference) != len(trial):
            raise AssertionError(f"Sample count changed at {path}")
        for index, (left, right) in enumerate(zip(reference, trial)):
            compare_samples(left, right, f"{path}[{index}]")
    elif isinstance(reference, float):
        if abs(reference - trial) > 1e-4:
            raise AssertionError(f"Sample value changed at {path}: {reference} vs {trial}")
    elif reference != trial:
        raise AssertionError(f"Sample value changed at {path}: {reference} vs {trial}")


def main():
    assert_gpu1()
    from src import train_mamba_rl as train

    fast_run = len(sys.argv) > 1 and sys.argv[1] == "--fast-run"
    original_evaluate = train.evaluate
    state = {"selected": "cached_projected_items_and_packed_history" if fast_run else None,
             "report": None}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    original_history_batch = train.evaluation_history_batch

    def packed_history_batch(data, users, split, max_history, device):
        histories = []
        for user in users:
            history = list(data.train_by_user[user])
            if split == "test":
                history.append(data.valid_target[user])
            histories.append(history[-max_history:])
        sizes = [len(history) for history in histories]
        padded = np.zeros((len(users), max(max(sizes), 1)), dtype=np.int64)
        for row, history in enumerate(histories):
            model_history = history[::-1] if getattr(data, "reverse_history_input", False) else history
            padded[row, :len(history)] = model_history
        return (torch.from_numpy(padded).to(device),
                torch.tensor(sizes, device=device), histories)

    def cached_evaluate(model, data, split, batch_size, max_history, device, *args, pack_history=False, **kwargs):
        original_score = model.full_catalog_scores

        def cached_score(histories, lengths, item_vectors=None, graph_items=None, user_ids=None):
            if item_vectors is None:
                return original_score(histories, lengths, item_vectors, graph_items, user_ids)
            original_project = model.project_ids
            try:
                # project_all() has just computed these exact item representations.
                # Evaluation histories need only gather them by item ID.
                model.project_ids = lambda ids, graph_items=None: item_vectors[ids]
                return original_score(histories, lengths, item_vectors, graph_items, user_ids)
            finally:
                model.project_ids = original_project

        model.full_catalog_scores = cached_score
        if pack_history:
            train.evaluation_history_batch = packed_history_batch
        try:
            return original_evaluate(model, data, split, batch_size, max_history, device, *args, **kwargs)
        finally:
            model.full_catalog_scores = original_score
            train.evaluation_history_batch = original_history_batch

    def paired_evaluate(model, data, split, batch_size, max_history, device, *args, **kwargs):
        if split != "test":
            if fast_run:
                return cached_evaluate(model, data, split, batch_size, max_history, device, *args,
                                       pack_history=True, **kwargs)
            return original_evaluate(model, data, split, batch_size, max_history, device, *args, **kwargs)
        if state["selected"] is not None:
            return cached_evaluate(model, data, split, batch_size, max_history, device, *args,
                                   pack_history=state["selected"] == "cached_projected_items_and_packed_history", **kwargs)

        reference, reference_samples = original_evaluate(
            model, data, split, batch_size, max_history, device, *args, **kwargs)
        results = []
        for mode, packed in (("cached_projected_items", False),
                             ("cached_projected_items_and_packed_history", True)):
            trials = []
            for _ in range(2):
                metrics, samples = cached_evaluate(
                    model, data, split, batch_size, max_history, device, *args,
                    pack_history=packed, **kwargs)
                compare_metrics(reference, metrics)
                compare_samples(reference_samples, samples)
                trials.append(float(metrics["users_per_second"]))
            results.append({"mode": mode, "batch_size": batch_size,
                            "users_per_second_trials": trials,
                            "median_users_per_second": statistics.median(trials)})
        chosen = max(results, key=lambda row: row["median_users_per_second"])
        state["selected"] = chosen["mode"]
        state["report"] = {
            "model": "Ours (ver5)", "physical_gpu_index": 1,
            "gpu_uuid": json.loads((BASE / "profile.json").read_text())["gpu_uuid"],
            "baseline_batch_size": batch_size,
            "baseline_users_per_second_same_weights": reference["users_per_second"],
            "selected_batch_size": batch_size,
            "selected_mode": chosen["mode"],
            "optimized_users_per_second": chosen["median_users_per_second"],
            "speedup_vs_paired_baseline": chosen["median_users_per_second"] / reference["users_per_second"],
            "evaluated_users": reference["evaluated_users"],
            "hit_and_recall_exact_ndcg_within_1e-7": True,
            "all_diagnostics_within_1e-4": True,
            "sampled_recommendations_equal_with_1e-4_float_tolerance": True,
            "candidates": results,
            "change": "Evaluation-only reuse of preprojected catalog item vectors for history positions; optional single-transfer CPU packing of each history batch; batch size unchanged",
            "memory_source": "gpu1_runs/ours_full_beauty/profile.json (prior measurement; not rerun)",
        }
        (OUTPUT / "throughput_optimized.json").write_text(json.dumps(state["report"], indent=2) + "\n")
        print(json.dumps(state["report"], indent=2), flush=True)
        return cached_evaluate(model, data, split, batch_size, max_history, device, *args,
                               pack_history=chosen["mode"] == "cached_projected_items_and_packed_history", **kwargs)

    train.evaluate = paired_evaluate
    if fast_run:
        argv = sys.argv[2:]
        if not argv:
            raise ValueError("Pass train_mamba_rl arguments after --fast-run")
    else:
        baseline_command = json.loads((BASE / "profile.json").read_text())["command"]
        if baseline_command[1:3] != ["-m", "src.train_mamba_rl"]:
            raise RuntimeError("Unexpected original command format")
        argv = baseline_command[3:]
        index = argv.index("--output-run-dir") + 1
        argv[index] = str((OUTPUT / "model_output").relative_to(PROJECT))
    sys.argv = ["src.train_mamba_rl", *argv]
    started = time.perf_counter()
    train.main()
    name = "fast_run_wall_seconds" if fast_run else "paired_pilot_wall_seconds"
    print(f"{name}={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
