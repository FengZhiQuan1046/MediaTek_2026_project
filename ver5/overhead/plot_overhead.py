#!/usr/bin/env python3
"""Two-panel paper figure from measured physical GPU 1 pilot profiles."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ver5-overhead-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
RUNS = HERE / "gpu1_runs"
ROWS = [
    ("Ours (ver5)", "ours_full_beauty"),
    ("SIGMA", "sigma_full_beauty"),
    ("Mamba4Rec", "mamba4rec_full_beauty"),
    ("EAGER", "eager_full_beauty"),
    ("BERT4Rec", "bert4rec_full_beauty"),
    ("ReSID", "resid_full_beauty"),
    ("LLM-SRec", "llm_srec_full_beauty"),
]


def load():
    rows = []
    for label, directory in ROWS:
        profile_path = RUNS / directory / "profile.json"
        metrics_path = RUNS / directory / "model_output" / "metrics.json"
        profile = json.loads(profile_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        if profile["exit_code"] != 0 or profile["physical_gpu_index"] != 1:
            raise RuntimeError(f"Invalid physical GPU 1 profile: {profile_path}")
        test = metrics["test"]
        throughput = float(test["users_per_second"])
        throughput_source = metrics_path.relative_to(HERE).as_posix()
        if label == "Ours (ver5)":
            optimized_path = RUNS / "ours_throughput_optimized" / "throughput_optimized.json"
            if optimized_path.exists():
                optimized = json.loads(optimized_path.read_text())
                if optimized.get("physical_gpu_index") != 1 or not optimized.get("sampled_recommendations_equal_with_1e-4_float_tolerance"):
                    raise RuntimeError("Optimized throughput audit failed")
                throughput = float(optimized["optimized_users_per_second"])
                throughput_source = optimized_path.relative_to(HERE).as_posix()
        memory = float(profile["sampled_process_peak_mib"])
        if throughput <= 0 or memory <= 0:
            raise ValueError(label)
        rows.append(dict(model=label, throughput_users_per_second=throughput,
                         peak_process_memory_mib=memory,
                         wall_seconds="" if label == "Ours (ver5)" and throughput_source != metrics_path.relative_to(HERE).as_posix() else profile["wall_seconds"],
                         evaluated_users=test["evaluated_users"],
                         catalog_items=metrics.get("items", metrics.get("num_items", 4090)),
                         throughput_source=throughput_source,
                         profile=profile_path.relative_to(HERE).as_posix(),
                         metrics=metrics_path.relative_to(HERE).as_posix()))
    return rows


def style():
    family = "Times New Roman" if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist) else "Liberation Serif"
    # Match the preliminary/aggregated palette, serif face, spine and grid weights.
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "axes.edgecolor": "#666666",
        "grid.color": "#D8DCE1", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def plot(rows):
    style()
    labels = [row["model"] for row in rows]
    y = list(range(len(rows)))
    colors = ["#D55E00" if label == "Ours (ver5)" else "#0072B2" for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True,
                             gridspec_kw={"wspace": .22})
    throughput = [row["throughput_users_per_second"] for row in rows]
    memory = [row["peak_process_memory_mib"] / 1024 for row in rows]
    left, right = axes
    left.barh(y, throughput, color=colors, height=.63, zorder=3)
    right.barh(y, memory, color=colors, height=.63, zorder=3)
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.set_axisbelow(True)
        ax.grid(axis="x", zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.minorticks_off()
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlim(10, 5e4)
    left.set_xticks([10, 100, 1000, 10000])
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
    left.set_xlabel("Test users per second (log scale)")
    left.set_title("(a) Full-catalog throughput", loc="left", fontweight="bold")
    right.set_xscale("log")
    right.set_xlim(.4, 65)
    right.set_xticks([.5, 1, 2, 4, 8, 16, 32])
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    right.set_xlabel("Peak process GPU memory (GiB, log scale)")
    right.set_title("(b) GPU memory", loc="left", fontweight="bold")
    right.tick_params(labelleft=False)
    for index, (speed, gib) in enumerate(zip(throughput, memory)):
        left.text(speed * 1.09, index, f"{speed/1000:.1f}k" if speed >= 1000 else f"{speed:.1f}",
                  va="center", ha="left", fontsize=7)
        right.text(gib * 1.09, index, f"{gib:.1f}", va="center", ha="left", fontsize=7)
    fig.subplots_adjust(left=.17, right=.975, top=.88, bottom=.17)
    out = HERE / "figure_gpu1_overhead"
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out.with_suffix("." + suffix), dpi=400, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    rows = load()
    with (HERE / "figure_gpu1_overhead_data.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    print(HERE / "figure_gpu1_overhead.pdf")


if __name__ == "__main__":
    main()
