#!/usr/bin/env python3
"""Summarize existing Full_Beauty artifacts without claiming a new benchmark."""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VER5 = HERE.parent
FIELDS = (
    "model", "status", "run_dir", "trainable_parameters", "model_id",
    "batch_size", "eval_batch_size", "logged_wall_seconds", "test_users_per_second", "test_scores_per_second",
    "gpu1_peak_mib", "note",
)


def newest(pattern, accept=lambda config: True):
    runs = []
    for metrics_file in ROOT.glob(pattern):
        run_dir = metrics_file.parent
        config_file = run_dir / "config.json"
        log_files = list(run_dir.glob("train*.log"))
        if not log_files:
            continue
        try:
            log = log_files[0].read_text(errors="replace")
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
            if not config:
                match = re.search(r"EXPERIMENT_CONFIG (\{[^\n]+\})", log)
                config = json.loads(match.group(1)) if match else {}
            if accept(config):
                runs.append((metrics_file.stat().st_mtime, metrics_file, log, config))
        except (OSError, ValueError):
            continue
    return max(runs, default=None, key=lambda row: row[0])


def row(model, result, note=""):
    base = dict.fromkeys(FIELDS, "")
    base.update(model=model, status="no_archived_run", note=note)
    if result is None:
        return base
    _, metrics_file, log, config = result
    metrics = json.loads(metrics_file.read_text())
    test = metrics.get("test", {})
    params = [int(value) for value in re.findall(r"trainable_parameters=(\d+)", log)]
    stamps = re.findall(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}),\d+ \|", log, re.M)
    logged_wall = (datetime.fromisoformat(stamps[-1]) - datetime.fromisoformat(stamps[0])).total_seconds() if len(stamps) >= 2 else ""
    base.update(
        status="archived_log_only",
        run_dir=str(metrics_file.parent.relative_to(ROOT)),
        trainable_parameters=max(params) if params else "",
        model_id=config.get("mamba_model_id") or (config.get("large_model_id", "") if config.get("use_large_mamba") else ""),
        batch_size=config.get("batch_size", ""),
        eval_batch_size=config.get("eval_batch_size", ""),
        logged_wall_seconds=logged_wall,
        test_users_per_second=test.get("users_per_second", ""),
        test_scores_per_second=test.get("scores_per_second", ""),
        note=note,
    )
    return base


def main():
    paths = {
        "Ours (ver5)": ("MediaTek_2026_project/ver5/outputs_mamba_rl/amazons_long_short_S1_P1_G1_C1_SW4/Full_Beauty/*/metrics.json", lambda c: c.get("joint_epochs") == 0 and c.get("preference_count") == 32, "Frozen text encoder excluded from logged trainable count; cache encoding is a separate phase."),
        "SIGMA": (None, None, "Upstream code exists, but no matching project run or GPU memory trace."),
        "Mamba4Rec": ("MediaTek_2026_project/Reproduce/Mamba4Rec/outputs/Full_Beauty/*/metrics.json", lambda c: c.get("use_large_mamba") is False, "Official small backbone variant; archived log does not contain GPU peak."),
        "EAGER": ("MediaTek_2026_project/Reproduce/EAGER/outputs/Full_Beauty/*/metrics.json", lambda c: True, "Multiphase method: full peak must include semantic encoding, DIN and recommender training."),
        "BERT4Rec": ("MediaTek_2026_project/Reproduce/BERT4Rec/outputs/Full_Beauty/*/metrics.json", lambda c: True, "Archived log does not contain GPU peak."),
        "ReSID": ("MediaTek_2026_project/Reproduce/ReSID/outputs/Full_Beauty/*/metrics.json", lambda c: True, "Multiphase method: full peak must include FAMAE and recommender training."),
        "LLM-SRec": (None, None, "Upstream code exists but no project-compatible run or GPU memory trace."),
    }
    rows = [row(name, newest(pattern, predicate) if pattern else None, note)
            for name, (pattern, predicate, note) in paths.items()]
    output = HERE / "archive_summary.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Existing Full Beauty run records",
        "",
        "These values come from archived logs, not a controlled GPU 1 benchmark. Blank GPU peak means no measurement was recorded. Different batch sizes, implementations, and hardware make throughput values descriptive only.",
        "",
        "| Model | Trainable parameters logged | Log span (s) | Test users/s | GPU 1 peak (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in rows:
        def display(value):
            return f"{value:,.1f}" if isinstance(value, float) else (str(value) if value != "" else "—")
        lines.append("| " + " | ".join(display(item[key]) for key in
                     ("model", "trainable_parameters", "logged_wall_seconds", "test_users_per_second", "gpu1_peak_mib")) + " |")
    lines += ["", "`archive_summary.csv` contains the exact values, source run directories, and caveats.", ""]
    (HERE / "archive_summary.md").write_text("\n".join(lines))
    print(output)
    for item in rows:
        print(item["model"], item["status"], item["trainable_parameters"], item["test_users_per_second"])


if __name__ == "__main__":
    main()
