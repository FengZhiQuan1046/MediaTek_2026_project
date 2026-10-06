#!/usr/bin/env python3
"""Summarize completed physical GPU 1 pilot profiles; failed runs stay empty."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "gpu1_runs"
MODELS = {
    "Ours (ver5)": "ours_full_beauty",
    "SIGMA": "sigma_full_beauty",
    "Mamba4Rec": "mamba4rec_full_beauty",
    "EAGER": "eager_full_beauty",
    "BERT4Rec": "bert4rec_full_beauty",
    "ReSID": "resid_full_beauty",
    "LLM-SRec": "llm_srec_full_beauty",
}
FIELDS = ("model", "status", "physical_gpu_index", "peak_process_memory_mib",
          "wall_seconds", "test_users_per_second", "throughput_source",
          "sampling_interval_seconds", "command", "profile_path")


def main():
    rows = []
    for name, directory in MODELS.items():
        profile = RUNS / directory / "profile.json"
        row = dict.fromkeys(FIELDS, "")
        row["model"] = name
        row["status"] = "not_run"
        if profile.exists():
            data = json.loads(profile.read_text())
            success = data.get("exit_code") == 0 and data.get("sampled_process_peak_mib") is not None
            row.update(status="measured_pilot" if success else f"failed_exit_{data.get('exit_code')}",
                       physical_gpu_index=data.get("physical_gpu_index", ""),
                       peak_process_memory_mib=data.get("sampled_process_peak_mib") if success else "",
                       wall_seconds=data.get("wall_seconds") if success else "",
                       sampling_interval_seconds=data.get("sampling_interval_seconds", ""),
                       command=" ".join(data.get("command", [])),
                       profile_path=str(profile.relative_to(HERE)))
            metrics_file = RUNS / directory / "model_output" / "metrics.json"
            if success and metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
                row["test_users_per_second"] = metrics.get("test", {}).get("users_per_second", "")
                row["throughput_source"] = str(metrics_file.relative_to(HERE))
            if success and name == "Ours (ver5)":
                optimized = RUNS / "ours_throughput_optimized" / "throughput_optimized.json"
                if optimized.exists():
                    audit = json.loads(optimized.read_text())
                    if audit.get("physical_gpu_index") != 1 or not audit.get("sampled_recommendations_equal_with_1e-4_float_tolerance"):
                        raise ValueError("Optimized throughput failed GPU-1 or ranking audit")
                    row["test_users_per_second"] = audit["optimized_users_per_second"]
                    row["throughput_source"] = str(optimized.relative_to(HERE))
                    row["status"] = "optimized_throughput_prior_memory"
                    row["wall_seconds"] = ""
        rows.append(row)
    output = HERE / "gpu1_summary.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Physical GPU 1 pilot profiles", "",
             "Full Beauty, one shortened run per available implementation. Memory is a 50 ms NVML process sample; time includes Python startup, cache access, training, and full evaluation. These are pilot overheads, not matched full-training requirements.", "",
             "| Model | Status | Peak MiB | Wall seconds | Test users/s |", "|---|---|---:|---:|---:|"]
    for item in rows:
        peak = item["peak_process_memory_mib"] or "—"
        wall = f"{item['wall_seconds']:.2f}" if item["wall_seconds"] else "—"
        throughput = f"{item['test_users_per_second']:,.1f}" if item["test_users_per_second"] else "—"
        lines.append(f"| {item['model']} | {item['status']} | {peak} | {wall} | {throughput} |")
    lines += ["", "Exact commands, source profiles and failure statuses are in `gpu1_summary.csv`.", ""]
    (HERE / "gpu1_summary.md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
