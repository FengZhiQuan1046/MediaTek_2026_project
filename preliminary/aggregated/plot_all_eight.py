"""Replot the baseline preliminary item-level diagnostics for eight datasets.

Reads existing Amazon and additional-dataset summary CSVs. No model training
or metric recomputation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ADDITIONAL = PROJECT / "preliminary_grouped" / "outputs" / "additional_dated_20261023" / "item_level"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "preliminary_grouped"))

import plot_aggregated as style
from additional_datasets import STYLES
from period_plot_style import LINESTYLES

AMAZON = tuple(style.DATASETS)
EXTRAS = {
    "Steam": "steam",
    "MovieLens1M": "movielens-1m",
    "Yelp19": "yelp19-dated-matched",
    "Yelp23": "yelp23-dated-matched",
}
FILES = tuple(style.FILES)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tables():
    amazon_sources = json.loads((HERE / "sources.json").read_text())
    extra_sources = json.loads((ADDITIONAL / "sources.json").read_text())
    if {entry["dataset"] for entry in amazon_sources["sources"]} != {row[1] for row in AMAZON}:
        raise ValueError("Amazon aggregate does not contain the four expected subsets")
    if {entry["dataset"] for entry in extra_sources["sources"]} != set(EXTRAS):
        raise ValueError("Additional item-level aggregate does not contain four datasets")
    extra_settings = {
        name: (STYLES[name][0], STYLES[name][1], STYLES[name][2], LINESTYLES[name])
        for name in EXTRAS
    }
    datasets = list(AMAZON) + [
        (name, EXTRAS[name], *extra_settings[name]) for name in EXTRAS
    ]
    tables = {name: [] for name in FILES}
    inputs = [HERE / "sources.json", HERE / "plotted_data.csv", ADDITIONAL / "sources.json"]
    with (HERE / "plotted_data.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test" and row["source_table"] in tables:
                tables[row["source_table"]].append(row)
    for name in FILES:
        path = ADDITIONAL / name
        inputs.append(path)
        with path.open(newline="") as stream:
            tables[name].extend(row for row in csv.DictReader(stream) if row["split"] == "test")
        expected = {row[1] for row in datasets}
        if {row["dataset"] for row in tables[name]} != expected:
            raise ValueError(f"Missing dataset in {name}")
    return tables, datasets, inputs


def format_axis(ax, kind):
    if kind in ("alignment", "preference"):
        ax.set_xlim(.5, 100.5)
        ax.set_xticks([1, 24.5, 48.5, 82.5], labels=["1", "17–32", "33–64", "65–100"])
    elif kind == "influence":
        ax.set_xlim(.5, 100.5)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
    else:
        ax.set_xlim(.5, 100.5)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.title.set_fontsize(8)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    ax.tick_params(labelsize=6.5)


def render(tables, datasets, args):
    style.DATASETS = datasets
    style.style(args.font_path)
    specs = (
        ("alignment", "Semantic alignment", "Excess cosine (z)", "Lag bin"),
        ("influence", "Item influence", "Margin change", "Lag"),
        ("coverage", "Signal coverage", "Signal (%)", "Recent items"),
        ("preference", "Preference match", "Excess match (pp)", "Lag bin"),
    )
    for columns, stem, size in (
        ((0, 1, 2, 3), "figure0_abcd_row_all8", (9.6, 2.55)),
        ((0, 1, 2), "figure0_acd_row_all8", (8.8, 2.55)),
    ):
        fig, axes = style.plt.subplots(1, len(columns), figsize=size)
        for col, (ax, index) in enumerate(zip(axes, columns)):
            kind, title, ylabel, xlabel = specs[index]
            style.decorate(ax, f"({chr(97 + col)}) {title}", ylabel, xlabel,
                           zero=kind != "coverage")
            style.draw(ax, tables, kind, args.min_users, preference_count=32)
            format_axis(ax, kind)
        handles = [style.Line2D([], [], color=color, marker=marker, linestyle=line,
                                lw=1.6, markersize=4, markerfacecolor="white",
                                markeredgewidth=.65, label=label)
                   for _, _, label, color, marker, line in datasets]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .998),
                   ncol=4, frameon=False, fontsize=8, columnspacing=1.4,
                   handlelength=2.5)
        fig.subplots_adjust(left=.075, right=.96, top=.73, bottom=.19,
                            wspace=.28 if len(columns) == 4 else .24)
        style.save(fig, args.output, stem, ("pdf", "png", "svg"), 400)
        print("Saved", args.output / f"{stem}.pdf", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    if args.min_users < 1:
        parser.error("--min-users must be positive")
    tables, datasets, inputs = load_tables()
    args.output.mkdir(parents=True, exist_ok=True)
    render(tables, datasets, args)
    provenance = {
        "figure_status": "measured_baseline",
        "datasets": [{"label": row[2], "dataset_id": row[1]} for row in datasets],
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "split": "test",
        "min_users_for_influence_and_alignment": args.min_users,
        "metrics": list(FILES),
        "no_training_or_metric_recomputation": True,
    }
    (args.output / "sources_all8.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
