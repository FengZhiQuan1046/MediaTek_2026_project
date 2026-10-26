"""Draw the eight-dataset baseline temporal-period row from saved reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from additional_datasets import STYLES
from period_plot_style import LINESTYLES, MARKER

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
AMAZON = HERE / "outputs" / "grouped_preliminary"
ADDITIONAL = HERE / "outputs" / "additional_dated_20261023"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ADDITIONAL / "aggregated")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(PROJECT / "preliminary" / "aggregated"))
    import plot_aggregated as style

    amazon_names = ("All_Beauty", "Baby_Products", "Sports_and_Outdoors", "Toys_and_Games")
    datasets = [(name, label, color, marker, linestyle)
                for name, (_, _, label, color, marker, linestyle)
                in zip(amazon_names, style.DATASETS)]
    datasets.extend((name, *STYLES[name], LINESTYLES[name]) for name in STYLES)
    reports = {}
    sources = []
    for name, *_ in datasets:
        path = (AMAZON / f"{name}.json" if name not in STYLES
                else ADDITIONAL / name / f"{name}.json")
        report = json.loads(path.read_text())
        if report.get("split") != "test" or report["grouping"]["gap_days"] != 30:
            raise ValueError(f"Incompatible report: {path}")
        reports[name] = report["five_figure_statistics"]
        sources.append({"path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    style.style(None)
    fig, axes = style.plt.subplots(1, 3, figsize=(8.10, 2.25))
    specs = (("alignment", "(a) Semantic alignment", "Excess cosine (z)"),
             ("influence", "(b) Item influence", "Margin change"),
             ("coverage", "(c) Signal coverage", "Signal covered (%)"))
    for ax, (kind, title, ylabel) in zip(axes, specs):
        style.decorate(ax, title, ylabel, "Time period", kind != "coverage")
        for name, label, color, marker, linestyle in datasets:
            stats = reports[name][kind]
            rows = stats["all_available"] if kind == "alignment" else stats
            rows = [row for row in rows if kind == "coverage" or row["n_users"] >= 100]
            if not rows:
                continue
            ax.plot([row["group"] for row in rows], [row["mean"] for row in rows],
                    color=color, marker=marker, linestyle=linestyle,
                    markevery=max(1, len(rows) // 9), label=label, **MARKER)
            low, high = (("q25", "q75") if kind == "coverage"
                         else ("ci_low", "ci_high"))
            shade = [row for row in rows if row.get(low) is not None]
            ax.fill_between([row["group"] for row in shade],
                            [row[low] for row in shade],
                            [row[high] for row in shade],
                            color=color, alpha=.08 if kind == "coverage" else .11,
                            linewidth=0, zorder=1)
        if kind == "coverage":
            ax.set_xlim(.5, 20.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_ylim(0, 102)
            ax.axhline(80, color="#777777", lw=.65, ls="--", zorder=1)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        ax.tick_params(labelsize=6.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) != 8:
        raise ValueError(f"Expected eight plotted datasets, got {labels}")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .998),
               ncol=4, frameon=False, fontsize=8, columnspacing=1.4,
               handlelength=2.5)
    fig.subplots_adjust(left=.075, right=.97, top=.67, bottom=.20, wspace=.24)
    args.output.mkdir(parents=True, exist_ok=True)
    style.save(fig, args.output, "figure0_acd_row", ("pdf", "png", "svg"), 400)
    (args.output / "figure0_acd_row_sources.json").write_text(
        json.dumps({"datasets": [row[0] for row in datasets], "sources": sources,
                    "split": "test", "min_users": 100, "period_gap_days": 30}, indent=2) + "\n")
    print("Saved", args.output / "figure0_acd_row.pdf")


if __name__ == "__main__":
    main()
