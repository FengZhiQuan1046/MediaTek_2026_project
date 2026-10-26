"""Combine measured gate=0.9 and decay=1.0 for all eight datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from additional_datasets import STYLES
from period_plot_style import LINESTYLES
from plot_all_eight_decay_grid import AMAZON, ADDITIONAL, SPECS, datasets
import plot_amazon_decay_grid as amazon_plot
import plot_additional_decay_grid as additional_plot

HERE = Path(__file__).resolve().parent
GATE_ROWS = (("090", 0.9),)
DECAY_ROWS = (("100", 1.0),)


def load_pair(paths, parameter, expected):
    amazon, additional = (json.loads(path.read_text()) for path in paths)
    for label, record in (("Amazon", amazon), ("Additional", additional)):
        observed = {entry[parameter] for entry in record["sources"]}
        if observed != expected or len(record["sources"]) != 12:
            raise ValueError(f"{label} {parameter} sources are {observed}; expected {expected}")
    if set(amazon["statistics"]) != {row[0] for row in AMAZON}:
        raise ValueError("Amazon source must contain four subsets")
    return amazon["statistics"], additional["statistics"]


def sources():
    gate_paths = (HERE / "outputs/amazon_gate_grid/plotted_data.json",
                  HERE / "outputs/additional_gate_grid/plotted_data.json")
    gate_amazon, gate_additional = load_pair(gate_paths, "gate", {0.3, 0.7, 0.9})
    collect_args = SimpleNamespace(cache_dir=HERE.parent.parent / "cache",
                                   bootstrap=2000, seed=25252, max_x=20)
    decay_amazon, amazon_sources = amazon_plot.collect(
        collect_args, decays=(("100", 1.0, "#009E73", "^", "-."),))
    decay_additional, additional_sources = additional_plot.collect(
        collect_args, decays=DECAY_ROWS)
    if set(decay_amazon) != {row[0] for row in AMAZON} or set(decay_additional["100"]) != set(ADDITIONAL):
        raise ValueError("Measured decay=1.0 must include all eight datasets")
    rows = [("gate", value, tag, gate_amazon, gate_additional)
            for tag, value in GATE_ROWS]
    rows += [("decay", value, tag, decay_amazon, decay_additional)
             for tag, value in DECAY_ROWS]
    return rows, gate_paths, amazon_sources + additional_sources


def plot(args, rows):
    amazon_plot.original_style.style(args.font_path)
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 3.55))
    for row, (parameter, value, tag, amazon, additional) in enumerate(rows):
        for col, (metric, title, xlabel, ylabel, xkey, low, high) in enumerate(SPECS):
            ax = axes[row, col]
            amazon_plot.original_style.decorate(
                ax, f"({chr(97 + 4 * row + col)}) {title} ({parameter}={value:.1f})",
                ylabel, xlabel if row == len(rows) - 1 else "", zero="coverage" not in metric)
            for label, color, marker, line, points in datasets(amazon, additional, tag, metric):
                points = [p for p in points if p[xkey] <= (100 if metric == "item_coverage" else 20)
                          and ("influence" not in metric or p["n_users"] >= args.min_users)]
                if not points:
                    continue
                band = [p for p in points if p.get(low) is not None and p.get(high) is not None
                        and np.isfinite(p[low]) and np.isfinite(p[high])]
                if band:
                    ax.fill_between([p[xkey] for p in band], [p[low] for p in band],
                                    [p[high] for p in band], color=color, alpha=.045,
                                    linewidth=0, zorder=1)
                ax.plot([p[xkey] for p in points], [p["mean"] for p in points],
                        color=color, linestyle=line, lw=1.35, marker=marker,
                        markersize=2.7, markeredgewidth=.55, markerfacecolor="white",
                        markevery=max(1, len(points) // 10), zorder=3)
            ax.set_xlim(.5, 100.5 if metric == "item_coverage" else 20.5)
            ax.set_xticks([1, 20, 40, 60, 80, 100] if metric == "item_coverage"
                          else [1, 5, 10, 15, 20])
            if "coverage" in metric:
                ax.set_ylim(0, 103)
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                ax.axhline(80, color="#777777", lw=.65, ls="--", zorder=1)
            ax.title.set_fontsize(7.4)
            ax.xaxis.label.set_size(7.2)
            ax.yaxis.label.set_size(7.2)
            ax.tick_params(labelsize=6.3)

    handles = [Line2D([], [], color=color, marker=marker, linestyle=LINESTYLES[key],
                      lw=1.35, markersize=3.5, markerfacecolor="white", label=label)
               for key, label, color, marker in AMAZON]
    handles.extend(Line2D([], [], color=STYLES[key][1], marker=STYLES[key][2],
                          linestyle=LINESTYLES[key], lw=1.35, markersize=3.5,
                          markerfacecolor="white", label=STYLES[key][0])
                   for key in ADDITIONAL)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .996),
               ncol=4, frameon=False, fontsize=8, columnspacing=1.4, handlelength=2.5)
    fig.subplots_adjust(left=.075, right=.96, top=.79, bottom=.09,
                        wspace=.29, hspace=.52)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = "all_eight_gate_decay_item_period_grid"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.output / f"{stem}.{ext}", dpi=400,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/all_eight_gate_decay_grid")
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    rows, gate_paths, decay_sources = sources()
    plot(args, rows)
    provenance = {
        "figure_status": "all_measured",
        "rows": [{"parameter": "gate", "display_value": value,
                  "measured_source_value": value, "source_tag": tag}
                 for tag, value in GATE_ROWS]
                + [{"parameter": "decay", "display_value": value,
                    "measured_source_value": value, "source_tag": tag}
                   for tag, value in DECAY_ROWS],
        "source_files": [{"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in gate_paths],
        "measured_decay_sources": decay_sources,
        "dataset_count": 8,
        "min_users_for_influence": args.min_users,
    }
    (args.output / "plotted_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Saved", args.output / "all_eight_gate_decay_item_period_grid.pdf", flush=True)


if __name__ == "__main__":
    main()
