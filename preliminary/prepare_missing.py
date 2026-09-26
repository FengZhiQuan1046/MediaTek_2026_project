"""Train a single full-history Mamba+LoRA+LightGCN before preliminary statistics."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VER4 = HERE.parent / "ver4"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from experiment import safe_name, ver4_cache_path  # noqa: E402


# The fixed per-subset training settings from ver4/run_amazons_full_rl.sh.
# A positive validation interval is capped to the number of steps in an epoch by
# ver4, so a deliberately large value below means exactly one periodic
# validation/test at the end of every epoch regardless of subset size.
SUBSET_SETTINGS = {
    "amazon-all-beauty": {"max_transitions": 500_000,
                          "candidates": 64, "popularity_alpha": -0.25, "transition_beta": 4.0},
    "amazon:Baby_Products": {"max_transitions": 1_500_000,
                             "candidates": 192, "popularity_alpha": 0.30, "transition_beta": 0.5},
    "amazon-sports-and-outdoors": {"max_transitions": 1_000_000,
                                    "candidates": 256, "popularity_alpha": 0.35, "transition_beta": 0.5},
    "amazon-toys-and-games": {"max_transitions": 1_500_000,
                               "candidates": 192, "popularity_alpha": 0.30, "transition_beta": 0.5},
    # Smaller Amazon Reviews 2023 raw categories; conservative single-GPU settings.
    "amazon:Musical_Instruments": {"max_transitions": 500_000,
                                   "candidates": 64, "popularity_alpha": 0.30, "transition_beta": 0.5},
    "amazon:Video_Games": {"max_transitions": 500_000,
                           "candidates": 64, "popularity_alpha": 0.30, "transition_beta": 0.5},
    "amazon:Software": {"max_transitions": 500_000,
                        "candidates": 64, "popularity_alpha": 0.30, "transition_beta": 0.5},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--mamba-model-id", required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-history", type=int, required=True)
    parser.add_argument("--influence-user-limit", type=int, required=True)
    parser.add_argument("--validation-user-limit", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.lora_rank < 1:
        parser.error("--lora-rank must be positive")
    if args.epochs < 1 or args.max_history < 2:
        parser.error("--epochs must be positive and --max-history must be at least 2")
    if args.influence_user_limit < 0:
        parser.error("--influence-user-limit cannot be negative")
    inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited_visible != args.gpu_ids:
        raise RuntimeError(
            f"GPU isolation mismatch: run.sh selected {args.gpu_ids}, "
            f"but CUDA_VISIBLE_DEVICES={inherited_visible!r}"
        )
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "preparation_manifest.json"
    records = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous, list):
                selected = set(datasets)
                records = [row for row in previous if row.get("dataset") not in selected]
        except (json.JSONDecodeError, OSError):
            records = []
    for dataset in datasets:
        if dataset not in SUBSET_SETTINGS:
            raise ValueError(f"No fixed ver4 subset settings registered for {dataset}")
        settings = SUBSET_SETTINGS[dataset]
        cache = ver4_cache_path(args.cache_dir, dataset)
        cache_status = "existing_reused_for_retraining" if cache.exists() else "created_by_training"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output / safe_name(dataset) / f"single_mamba_lora_lightgcn_{stamp}"
        score_file = run_dir / f"{safe_name(dataset)}_scores.json"
        influence_file = output / safe_name(dataset) / "history_item_influence.csv"
        run_dir.mkdir(parents=True, exist_ok=False)
        command = [
            "bash", str(VER4 / "run_mamba_rl.sh"), dataset, args.gpu_ids,
            "--enable-lora", "1", "--use-graph-embeddings",
            # Exactly one full-history sequence branch: no coordinator and no
            # short/preference agents. LightGCN remains fused with this branch.
            "--use-long", "1", "--use-short", "0", "--use-preference", "0",
            "--no-save-model-weights", "--no-generate-reasons",
            "--mamba-model-id", args.mamba_model_id,
            "--batch-size", "128", "--eval-batch-size", "64",
            "--monitor-metric", "ndcg@10", "--early-stopping-patience", "6",
            "--lr-patience", "2",
            "--dim", "128", "--lora-rank", str(args.lora_rank), "--lora-alpha", "32.0",
            "--lora-dropout", "0.05", "--short-window", "10",
            "--preference-count", "32", "--preference-hidden", "128",
            "--preference-temperature", "0.2", "--preference-score-weight", "0.2",
            "--max-history", str(args.max_history), "--mamba-encode-batch-size", "32",
            "--mamba-max-tokens", "32", "--specialist-lr", "1e-4",
            "--coordinator-lr", "1e-4", "--joint-lr", "5e-5",
            "--preference-coef", "0.2", "--preference-transition-coef", "0.1",
            "--preference-balance-coef", "0.01", "--preference-separation-coef", "0.01",
            "--preference-sharpness-coef", "0.05", "--future-horizon", "3",
            "--future-decay", "0.5", "--hard-negative-pool-multiplier", "12",
            "--hard-negative-fraction", "0.75", "--hard-negative-warmup-epochs", "2",
            "--use-in-batch-negatives", "--preference-contrastive-coef", "0.05",
            "--agent-diversity-coef", "0.02", "--no-full-catalog-supervised",
            "--max-transitions", str(settings["max_transitions"]),
            "--validate-every-steps", str(2 ** 31 - 1),
            "--candidates", str(settings["candidates"]),
            "--popularity-alpha", str(settings["popularity_alpha"]),
            "--transition-beta", str(settings["transition_beta"]),
            "--specialist-epochs", "0",
            "--coordinator-epochs", "0",
            "--joint-epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--validation-user-limit", str(args.validation_user_limit),
            "--periodic-test-user-limit", "0",
            "--history-influence-output", str(influence_file),
            "--history-influence-user-limit", str(args.influence_user_limit),
            "--history-influence-negatives", "64",
            "--output-run-dir", str(run_dir), "--score-file", str(score_file),
            "--experiment-note", "preliminary single full-history Mamba+LoRA+LightGCN; no coordinator or short/preference agents; no saved weights",
        ]
        environment = os.environ.copy()
        environment.update({"PYTHON_BIN": args.python_bin, "CACHE_DIR": args.cache_dir,
                            "CUDA_VISIBLE_DEVICES": args.gpu_ids,
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                            "REQUESTED_GPU_IDS": args.gpu_ids,
                            "EXPECTED_VISIBLE_GPUS": str(len(args.gpu_ids.split(","))),
                            "ENABLE_LORA": "1", "USE_GRAPH_EMBEDDINGS": "1",
                            "SAVE_MODEL_WEIGHTS": "0", "GENERATE_REASONS": "0",
                            "MAMBA_MODEL_ID": args.mamba_model_id})
        record = {"dataset": dataset, "status": "retraining", "cache": str(cache),
                  "cache_status": cache_status,
                  "run_dir": str(run_dir), "settings": settings, "command": command,
                  "architecture": {
                      "sequence_model": "single_full_history_selective_mamba_lora",
                      "lightgcn": True,
                      "active_sequence_branches": ["long"],
                      "coordinator": False,
                      "short_agent": False,
                      "preference_agent": False,
                  },
                  "training_epochs": args.epochs,
                  "history_item_influence": str(influence_file),
                  "model_weights_saved": False,
                  "evaluation_schedule": {
                      "periodic_validation": "once_per_epoch",
                      "periodic_test": "once_per_epoch",
                      "final_full_validation_and_test": True,
                  }}
        records.append(record)
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(
            f"{dataset}: retrain single full-history Mamba+LoRA(rank={args.lora_rank}) "
            "+ LightGCN (no coordinator/short/preference agents); "
            "full validation/test once per epoch",
            flush=True,
        )
        subprocess.run(command, cwd=VER4, env=environment, check=True)
        if not cache.exists():
            raise RuntimeError(f"ver4 training finished but expected data cache was not created: {cache}")
        if not influence_file.exists():
            raise RuntimeError(
                f"ver4 training finished but history influence statistics are missing: {influence_file}"
            )
        record["status"] = "trained"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
