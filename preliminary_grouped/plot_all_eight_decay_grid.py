"""Overlay all eight datasets in one three-decay, four-metric figure.

Use --source proxy for the clearly recorded old-curve layout preview, or
--source measured after the 0.5/0.7/1.0 experiments have all completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from additional_datasets import STYLES
from period_plot_style import LINESTYLES
import plot_amazon_decay_grid as amazon_plot

DECAYS = (0.5, 0.7, 1.0)
PROXY_TAGS = ("030", "010", "001")
MEASURED_TAGS = ("050", "070", "100")
AMAZON = (
    ("All_Beauty", "All Beauty", "#0072B2", "o"),
    ("Baby_Products", "Baby Products", "#D55E00", "s"),
    ("Sports_and_Outdoors", "Sports & Outdoors", "#009E73", "^"),
    ("Toys_and_Games", "Toys & Games", "#CC79A7", "D"),
)
ADDITIONAL = ("Steam", "MovieLens1M", "Yelp19", "Yelp23")
SPECS = (
    ("item_influence", "Item influence", "Lag", "Margin change", "lag", "ci_low", "ci_high"),
    ("item_coverage", "Signal coverage", "Recent items", "Signal (%)", "recent_items", "q25", "q75"),
    ("period_influence", "Period influence", "Time period", "Margin change", "group", "ci_low", "ci_high"),
    ("period_coverage", "Period coverage", "Recent periods", "Signal (%)", "group", "q25", "q75"),
)


def load_sources(source):
    if source == "proxy":
        paths = (HERE / "outputs/amazon_decay_grid_030_010_001/plotted_data.json",
                 HERE / "outputs/additional_decay_grid_030_010_001/plotted_data.json")
        tags, measured_values = PROXY_TAGS, (0.3, 0.1, 0.01)
    else:
        paths = (HERE / "outputs/amazon_decay_grid/plotted_data.json",
                 HERE / "outputs/additional_decay_grid/plotted_data.json")
        tags, measured_values = MEASURED_TAGS, DECAYS
    amazon, additional = (json.loads(path.read_text()) for path in paths)
    expected = set(measured_values)
    for name, record in (("Amazon", amazon), ("Additional", additional)):
        observed = {entry["decay"] for entry in record["sources"]}
        if observed != expected or len(record["sources"]) != 12:
            raise ValueError(f"{name} source decays are {observed}; expected {expected}")
    if set(amazon["statistics"]) != {row[0] for row in AMAZON}:
        raise ValueError("Amazon source must contain four subsets")
    if set(additional["statistics"]) != set(tags):
        raise ValueError("Additional source must contain all three decay rows")
    return paths, tags, measured_values, amazon["statistics"], additional["statistics"]


def datasets(amazon, additional, tag, metric):
    for key, label, color, marker in AMAZON:
        yield label, color, marker, LINESTYLES[key], amazon[key][tag][metric]
    for key in ADDITIONAL:
        label, color, marker = STYLES[key]
        yield label, color, marker, LINESTYLES[key], additional[tag][key][metric]


def plot(args, tags, amazon, additional, values=DECAYS, parameter="decay",
         stem="all_eight_decay_item_period_grid"):
    amazon_plot.original_style.style(args.font_path)
    fig, axes = plt.subplots(3, 4, figsize=(9.6, 5.0))
    for row, (tag, value) in enumerate(zip(tags, values)):
        for col, (metric, title, xlabel, ylabel, xkey, low, high) in enumerate(SPECS):
            ax = axes[row, col]
            amazon_plot.original_style.decorate(
                ax, f"({chr(97 + 4 * row + col)}) {title} ({parameter}={value:.1f})",
                ylabel, xlabel if row == 2 else "", zero="coverage" not in metric)
            for label, color, marker, line, points in datasets(amazon, additional, tag, metric):
                points = [p for p in points if p[xkey] <= (100 if metric == "item_coverage" else 20)
                          and ("influence" not in metric or p["n_users"] >= args.min_users)]
                if not points:
                    continue
                x = [p[xkey] for p in points]
                y = [p["mean"] for p in points]
                band = [p for p in points if p.get(low) is not None and p.get(high) is not None
                        and np.isfinite(p[low]) and np.isfinite(p[high])]
                if band:
                    ax.fill_between([p[xkey] for p in band], [p[low] for p in band],
                                    [p[high] for p in band], color=color, alpha=.045,
                                    linewidth=0, zorder=1)
                ax.plot(x, y, color=color, linestyle=line, lw=1.35, marker=marker,
                        markersize=2.7, markeredgewidth=.55, markerfacecolor="white",
                        markevery=max(1, len(x) // 10), zorder=3)
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
    fig.subplots_adjust(left=.075, right=.96, top=.84, bottom=.065,
                        wspace=.29, hspace=.53)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.output / f"{stem}.{ext}", dpi=400,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("proxy", "measured"), default="proxy")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/all_eight_decay_grid")
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    paths, tags, measured_values, amazon, additional = load_sources(args.source)
    plot(args, tags, amazon, additional)
    provenance = {
        "figure_status": ("draft_proxy_not_measured_at_displayed_decay_values"
                          if args.source == "proxy" else "measured"),
        "display_to_source_decay": [
            {"display_decay": display, "measured_source_decay": source, "source_tag": tag}
            for tag, source, display in zip(tags, measured_values, DECAYS)],
        "source_files": [{"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in paths],
        "datasets": [row[1] for row in AMAZON] + [STYLES[key][0] for key in ADDITIONAL],
        "layout": "three decay rows by four metrics; all eight datasets overlaid in each panel",
        "min_users_for_influence": args.min_users,
    }
    (args.output / "plotted_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Saved", args.output / "all_eight_decay_item_period_grid.pdf", flush=True)


if __name__ == "__main__":
    main()
