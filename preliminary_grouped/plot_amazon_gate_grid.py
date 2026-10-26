"""Plot Amazon gate sweeps as four metrics per subset, without retraining.

Run with the project Python environment:
    python preliminary_grouped/plot_amazon_gate_grid.py

The 30-day periods use the same consecutive-gap rule and user weighting as
plot_grouped_preliminary.py. Coverage is reused from the completed period
analysis only after its per-position input is verified byte-for-byte identical
to each gate run's input.
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
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/amazon-gate-grid-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "ver4"))  # Needed to unpickle InteractionData.
import amazon_group_lengths as lengths
from plot_grouped_preliminary import group_lags

STYLE_PATH = PROJECT / "preliminary" / "aggregated" / "plot_aggregated.py"
STYLE_SPEC = importlib.util.spec_from_file_location("preliminary_figure_style", STYLE_PATH)
original_style = importlib.util.module_from_spec(STYLE_SPEC)
STYLE_SPEC.loader.exec_module(original_style)

SUBSETS = (
    ("amazon-all-beauty", "All_Beauty", "All Beauty"),
    ("amazon_Baby_Products", "Baby_Products", "Baby Products"),
    ("amazon-sports-and-outdoors", "Sports_and_Outdoors", "Sports & Outdoors"),
    ("amazon-toys-and-games", "Toys_and_Games", "Toys & Games"),
)
GATES = (("030", 0.3, "#0072B2", "o", "-"),
         ("070", 0.7, "#D55E00", "s", "--"),
         ("090", 0.9, "#009E73", "^", "-."))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_table():
    result = {}
    for tag, gate, _, _, _ in GATES:
        root = PROJECT / "preliminary" / f"outputs_gate{tag}" / "aggregated"
        sources = json.loads((root / "sources.json").read_text())["sources"]
        if {s["folder"] for s in sources} != {s[0] for s in SUBSETS}:
            raise ValueError(f"Missing Amazon subset in {root / 'sources.json'}")
        for source in sources:
            run = Path(source["source"])
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["status"] != "complete" or manifest["settings"].get("mamba_gate") != gate:
                raise ValueError(f"Incomplete or mismatched gate: {run}")
            result[tag, source["folder"]] = (source, manifest)
    return result


def influence_by_period(path, mappings, history_ids, test_target, seed, bootstrap):
    per_user = defaultdict(lambda: defaultdict(list))
    for row in rows(path):
        if row["split"] != "test":
            continue
        user, lag = int(row["user"]), int(row["lag"])
        if (int(row["item"]) != history_ids[user][-lag]
                or int(row["target"]) != test_target[user]
                or int(row["history_length"]) != len(history_ids[user])):
            raise ValueError(f"History/influence mismatch in {path}, user {user}")
        per_user[user][mappings[user][lag]].append(float(row["margin_influence"]))
    by_group = defaultdict(list)
    for user_groups in per_user.values():
        for group, values in user_groups.items():
            by_group[group].append(float(np.mean(values)))
    output = []
    for group, values in sorted(by_group.items()):
        a = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(seed + 2000 + group)
        samples = np.empty(bootstrap, dtype=np.float64)
        for start in range(0, bootstrap, 128):
            stop = min(start + 128, bootstrap)
            index = rng.integers(0, len(a), size=(stop - start, len(a)))
            samples[start:stop] = a[index].mean(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        output.append(dict(group=group, mean=float(a.mean()), ci_low=float(low),
                           ci_high=float(high), n_users=len(a)))
    return output


def collect(args):
    sources = source_table()
    base_sources = {s["folder"]: s for s in json.loads(
        (PROJECT / "preliminary" / "aggregated" / "sources.json").read_text())["sources"]}
    output = {}
    provenance = []
    for folder, category, label in SUBSETS:
        first, manifest = sources["030", folder]
        diagnostic = next(d for d in manifest["datasets"] if d["dataset"] == first["dataset"])
        cache = Path(diagnostic["cache"])
        if digest(cache) != diagnostic["cache_sha256"]:
            raise ValueError(f"Changed interaction cache: {cache}")
        with cache.open("rb") as stream:
            data = pickle.load(stream)
        influence_files = {}
        users = set()
        for tag, _, _, _, _ in GATES:
            source, run_manifest = sources[tag, folder]
            diag = next(d for d in run_manifest["datasets"] if d["dataset"] == source["dataset"])
            if diag["cache_sha256"] != diagnostic["cache_sha256"]:
                raise ValueError(f"Different history cache for {category}, gate {tag}")
            influence = Path(diag["model_item_influence_source"])
            influence_files[tag] = influence
            users.update(int(r["user"]) for r in rows(influence) if r["split"] == "test")
        restored, check = lengths.recover(
            {"category": category}, args.cache_dir, users, data)
        max_lag = manifest["settings"]["max_lag"]
        history_ids = {u: (list(data.train_by_user[u]) + [data.valid_target[u]])[-max_lag:]
                       for u in users}
        mappings = {u: group_lags(restored[u][:len(data.train_by_user[u]) + 1][-max_lag:])
                    for u in users}

        base = base_sources[folder]
        base_run = Path(base["source"])
        base_manifest = json.loads((base_run / "manifest.json").read_text())
        base_diag = next(d for d in base_manifest["datasets"] if d["dataset"] == base["dataset"])
        base_period = json.loads((HERE / "outputs" / "grouped_preliminary" /
                                  f"{category}.json").read_text())["five_figure_statistics"]
        base_influence = influence_by_period(
            Path(base_diag["model_item_influence_source"]), mappings, history_ids,
            data.test_target, args.seed, args.bootstrap)
        reference = base_period["influence"]
        if len(base_influence) != len(reference) or any(
            row["group"] != ref["group"] or row["n_users"] != ref["n_users"]
            or not np.isclose(row["mean"], ref["mean"], atol=1e-10)
            for row, ref in zip(base_influence, reference)
        ):
            raise ValueError(f"Period mapping does not reproduce baseline for {category}")

        baseline_position = base_run / base["dataset"].replace(":", "_") / "alignment_per_position.csv"
        baseline_coverage = base_run / "information_coverage.csv"
        output[category] = {}
        for tag, gate, _, _, _ in GATES:
            source, _ = sources[tag, folder]
            run = Path(source["source"])
            position = run / source["dataset"].replace(":", "_") / "alignment_per_position.csv"
            if digest(position) != digest(baseline_position) or digest(run / "information_coverage.csv") != digest(baseline_coverage):
                raise ValueError(f"Cannot reuse period coverage for {category}, gate {gate}")
            item = rows(PROJECT / "preliminary" / f"outputs_gate{tag}" / "aggregated" / "plotted_data.csv")
            influence = [dict(lag=int(r["lag"]), mean=float(r["mean_margin_influence"]),
                              ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                              n_users=int(r["n_users"])) for r in item
                         if r["dataset"] == source["dataset"] and r["split"] == "test"
                         and r["source_table"] == "model_item_influence_summary.csv"]
            coverage = [dict(recent_items=int(r["recent_items"]), mean=float(r["mean_percent"]),
                             q25=float(r["q25_percent"]), q75=float(r["q75_percent"]),
                             n_users=int(r["n_users"])) for r in item
                        if r["dataset"] == source["dataset"] and r["split"] == "test"
                        and r["source_table"] == "information_coverage.csv"]
            if not influence or not coverage:
                raise ValueError(f"Missing item metrics for {category}, gate {gate}")
            output[category][tag] = {
                "item_influence": influence,
                "item_coverage": coverage,
                "period_influence": influence_by_period(influence_files[tag], mappings,
                                                        history_ids, data.test_target,
                                                        args.seed, args.bootstrap),
                "period_coverage": base_period["coverage"],
            }
            provenance.append({"subset": category, "gate": gate, "source": str(run),
                               "influence_sha256": digest(influence_files[tag]),
                               "alignment_sha256": digest(position),
                               "coverage_sha256": digest(run / "information_coverage.csv")})
        print(f"Collected {label}: three gates, {len(users)} influence users", flush=True)
    return output, provenance


def plot(data, args):
    original_style.style(args.font_path)
    specs = (
        ("item_influence", "Item influence", "Lag", "Margin change", "lag", "ci_low", "ci_high"),
        ("item_coverage", "Signal coverage", "Recent items", "Signal (%)", "recent_items", "q25", "q75"),
        ("period_influence", "Period influence", "Period", "Margin change", "group", "ci_low", "ci_high"),
        ("period_coverage", "Period coverage", "Recent periods", "Signal (%)", "group", "q25", "q75"),
    )
    # Each row overlays the same four Amazon subsets as figure0_abcd_row;
    # the three rows hold the gate values.
    fig, axes = plt.subplots(len(GATES), 4, figsize=(8.8, 4.5))
    for row, (tag, gate, _, _, _) in enumerate(GATES):
        for col, (key, title, xlabel, ylabel, xkey, low, high) in enumerate(specs):
            ax = axes[row, col]
            original_style.decorate(ax, f"({chr(97 + 4 * row + col)}) {title} (gate={gate:.1f})",
                                    ylabel, xlabel if row == len(GATES) - 1 else "",
                                    zero="coverage" not in key)
            for (_, category, _), (_, _, label, color, marker, linestyle) in zip(
                    SUBSETS, original_style.DATASETS):
                points = data[category][tag][key]
                if "influence" in key:
                    points = [p for p in points if p["n_users"] >= args.min_users]
                points = [p for p in points if p[xkey] <= args.max_x]
                if not points:
                    continue
                x = [p[xkey] for p in points]
                y = [p["mean"] for p in points]
                ax.fill_between(x, [p[low] for p in points], [p[high] for p in points],
                                color=color, alpha=.08 if "coverage" in key else .11,
                                linewidth=0, zorder=1)
                ax.plot(x, y, color=color, ls=linestyle, lw=1.6, marker=marker,
                        markersize=3.5, markeredgewidth=.65, markerfacecolor="white",
                        markevery=max(1, len(x) // 9), zorder=3)
            if "coverage" in key:
                ax.set_ylim(0, 103)
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                ax.axhline(80, color="#777777", lw=.65, ls="--", zorder=1)
                ax.set_xlim(.5, args.max_x + .5)
                ax.set_xticks([1, 10, 20])
            elif key == "item_influence":
                ax.set_xlim(left=.5)
                ax.set_xticks([1, 8, 16])
            else:
                ax.set_xlim(left=.5)
                ax.set_xticks([1, 4, 8])
            ax.title.set_fontsize(8)
            ax.xaxis.label.set_size(7.5)
            ax.yaxis.label.set_size(7.5)
            ax.tick_params(labelsize=6.5)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                      lw=1.6, markersize=4, label=label)
               for _, _, label, color, marker, linestyle in original_style.DATASETS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .998),
               ncol=4, frameon=False, columnspacing=1.6, handlelength=2.5)
    fig.subplots_adjust(left=.075, right=.955, top=.87, bottom=.055,
                        wspace=.28, hspace=.52)
    args.output.mkdir(parents=True, exist_ok=True)
    original_style.save(fig, args.output, "amazon_gate_item_period_grid",
                        ("pdf", "png", "svg"), 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT.parent / "cache")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "amazon_gate_grid")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--max-x", type=int, default=20)
    parser.add_argument("--font-path", type=Path,
                        help="Times New Roman TTF/OTF, as in preliminary/aggregated/plot_aggregated.py")
    args = parser.parse_args()
    if min(args.bootstrap, args.min_users, args.max_x) < 1:
        parser.error("--bootstrap, --min-users, and --max-x must be positive")
    data, provenance = collect(args)
    plot(data, args)
    (args.output / "plotted_data.json").write_text(json.dumps(
        {"statistics": data, "sources": provenance, "period_rule": "gap > 30 days",
         "period_influence": "mean individual item deletion effects within user/period, then users",
         "coverage_note": "All gate curves are identical; source inputs verified by SHA-256"},
        indent=2) + "\n")
    print("Saved", args.output / "amazon_gate_item_period_grid.pdf", flush=True)


if __name__ == "__main__":
    main()
