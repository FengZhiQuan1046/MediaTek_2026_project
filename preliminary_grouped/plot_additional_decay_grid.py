"""Three-decay item/period comparison for Steam, MovieLens 1M, Yelp19, Yelp23.

Uses the completed preliminary_grouped/outputs_decay{050,070,100} runs. The
four dataset curve styles match outputs/additional_dated_20261023/aggregated/
figure0_acd_row. No models or semantic signals are recomputed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/additional-decay-grid-mpl")
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

STYLE_PATH = HERE.parent / "preliminary" / "aggregated" / "plot_aggregated.py"
STYLE_SPEC = importlib.util.spec_from_file_location("preliminary_figure_style", STYLE_PATH)
style = importlib.util.module_from_spec(STYLE_SPEC)
STYLE_SPEC.loader.exec_module(style)

DECAYS = (("030", .5), ("010", .7), ("001", 1.0))
DATASETS = ("Steam", "MovieLens1M", "Yelp19", "Yelp23")
ITEM_COVERAGE_MAX_X = 100


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values, repetitions, seed):
    """Match preliminary.experiment.bootstrap_mean with batched RNG draws."""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        raise ValueError("Empty item influence lag")
    mean = float(a.mean())
    if len(a) == 1:
        return mean, None, None
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 128):
        stop = min(start + 128, repetitions)
        index = rng.integers(len(a), size=(stop - start, len(a)))
        means[start:stop] = a[index].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return mean, float(low), float(high)


def item_influence(path, max_x, bootstrap, seed):
    values = defaultdict(list)
    unique = set()
    for row in read_rows(path):
        if row["split"] != "test":
            continue
        lag = int(row["lag"])
        if lag > max_x:
            continue
        key = (row["user"], lag)
        if key in unique:
            raise ValueError(f"Duplicate user/lag influence in {path}: {key}")
        unique.add(key)
        values[lag].append(float(row["margin_influence"]))
    return [dict(lag=lag, n_users=len(v),
                 **dict(zip(("mean", "ci_low", "ci_high"),
                            bootstrap_mean(v, bootstrap, seed + lag))))
            for lag, v in sorted(values.items())]


def collect(args, decays=DECAYS):
    data = {}
    sources = []
    for tag, decay in decays:
        root = HERE / f"outputs_decay{tag}"
        completion = json.loads((root / "completion.json").read_text())
        if (completion.get("status") != "complete"
                or completion.get("mamba_decay") != decay or completion.get("mamba_gate") != 0.5
                or set(completion.get("datasets", [])) != set(DATASETS)):
            raise ValueError(f"Incomplete decay output: {root}")
        data[tag] = {}
        for name in DATASETS:
            folder = root / name
            report = json.loads((folder / f"{name}.json").read_text())
            if report.get("mamba_decay") != decay or report.get("mamba_gate") != 0.5:
                raise ValueError(f"Gate mismatch in {folder}")
            job = json.loads((folder / "intermediate" / "job.json").read_text())
            influence = folder / "intermediate" / "training" / job["dataset"].replace(":", "_") / "history_item_influence.csv"
            coverage = folder / "intermediate" / "signals" / "information_coverage.csv"
            item_coverage = [dict(recent_items=int(r["recent_items"]),
                                  mean=float(r["mean_percent"]),
                                  q25=float(r["q25_percent"]),
                                  q75=float(r["q75_percent"]),
                                  n_users=int(r["n_users"]))
                             for r in read_rows(coverage)
                             if r["split"] == "test" and int(r["recent_items"]) <= ITEM_COVERAGE_MAX_X]
            period = report["five_figure_statistics"]
            if not item_coverage or not period["influence"] or not period["coverage"]:
                raise ValueError(f"Missing item or period statistics in {folder}")
            data[tag][name] = {
                "item_influence": item_influence(influence, args.max_x,
                                                 args.bootstrap, args.seed),
                "item_coverage": item_coverage,
                "period_influence": period["influence"],
                "period_coverage": period["coverage"],
            }
            sources.append(dict(dataset=name, decay=decay, report=str(folder / f"{name}.json"),
                                item_influence=str(influence), influence_sha256=sha256(influence),
                                item_coverage=str(coverage), coverage_sha256=sha256(coverage)))
        print(f"Collected decay={decay:g}: {', '.join(DATASETS)}", flush=True)
    return data, sources


def plot(data, args):
    style.style(args.font_path)
    specs = (
        ("item_influence", "Item influence", "Lag", "Margin change", "lag", "ci_low", "ci_high"),
        ("item_coverage", "Signal coverage", "Recent items", "Signal (%)", "recent_items", "q25", "q75"),
        ("period_influence", "Period influence", "Time period", "Margin change", "group", "ci_low", "ci_high"),
        ("period_coverage", "Period coverage", "Time period", "Signal (%)", "group", "q25", "q75"),
    )
    fig, axes = plt.subplots(3, 4, figsize=(8.8, 4.5))
    for row, (tag, decay) in enumerate(DECAYS):
        for col, (key, title, xlabel, ylabel, xkey, low, high) in enumerate(specs):
            ax = axes[row, col]
            style.decorate(ax, f"({chr(97 + 4 * row + col)}) {title} (decay={decay:.1f})",
                           ylabel, xlabel if row == len(DECAYS) - 1 else "",
                           zero="coverage" not in key)
            for name in DATASETS:
                label, color, marker = STYLES[name]
                max_x = ITEM_COVERAGE_MAX_X if key == "item_coverage" else args.max_x
                points = [p for p in data[tag][name][key] if p[xkey] <= max_x]
                if "influence" in key:
                    points = [p for p in points if p["n_users"] >= args.min_users]
                if not points:
                    continue
                x = [p[xkey] for p in points]
                y = [p["mean"] for p in points]
                shade = [p for p in points if p.get(low) is not None and p.get(high) is not None]
                ax.fill_between([p[xkey] for p in shade], [p[low] for p in shade],
                                [p[high] for p in shade], color=color,
                                alpha=.08 if "coverage" in key else .11,
                                linewidth=0, zorder=1)
                ax.plot(x, y, color=color, ls=LINESTYLES[name], lw=1.6,
                        marker=marker, markersize=3.5, markeredgewidth=.65,
                        markerfacecolor="white", markevery=max(1, len(x) // 9), zorder=3)
            if "coverage" in key:
                ax.set_ylim(0, 103)
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                ax.axhline(80, color="#777777", lw=.65, ls="--", zorder=1)
                if key == "item_coverage":
                    ax.set_xlim(.5, ITEM_COVERAGE_MAX_X + .5)
                    ax.set_xticks([1, 20, 40, 60, 80, 100])
                else:
                    ax.set_xlim(.5, args.max_x + .5)
                    ax.set_xticks([1, 10, 20])
            elif key == "item_influence":
                ax.set_xlim(.5, args.max_x + .5)
                ax.set_xticks([1, 10, 20])
            else:
                ax.set_xlim(.5, args.max_x + .5)
                ax.set_xticks([1, 5, 10, 15, 20])
            ax.title.set_fontsize(8)
            ax.xaxis.label.set_size(7.5)
            ax.yaxis.label.set_size(7.5)
            ax.tick_params(labelsize=6.5)
    handles = [Line2D([], [], color=STYLES[name][1], marker=STYLES[name][2],
                      linestyle=LINESTYLES[name], lw=1.6, markersize=4,
                      markerfacecolor="white", markeredgewidth=.65,
                      label=STYLES[name][0]) for name in DATASETS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .998),
               ncol=4, frameon=False, columnspacing=1.6, handlelength=2.5)
    fig.subplots_adjust(left=.075, right=.955, top=.87, bottom=.055,
                        wspace=.28, hspace=.52)
    args.output.mkdir(parents=True, exist_ok=True)
    style.save(fig, args.output, "additional_decay_item_period_grid",
               ("pdf", "png", "svg"), 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "additional_decay_grid")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--max-x", type=int, default=20)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    if min(args.bootstrap, args.min_users, args.max_x) < 1:
        parser.error("--bootstrap, --min-users, and --max-x must be positive")
    data, sources = collect(args)
    plot(data, args)
    (args.output / "plotted_data.json").write_text(json.dumps(
        {"statistics": data, "sources": sources, "min_users": args.min_users,
         "bootstrap": args.bootstrap, "bootstrap_seed": args.seed,
         "period_rule": "new period when consecutive observed timestamps differ by >30 days",
         "period_influence": "mean single-item deletion effects within user/period, then across users"},
        indent=2) + "\n")
    print("Saved", args.output / "additional_decay_item_period_grid.pdf", flush=True)


if __name__ == "__main__":
    main()
