"""Overlay the eight measured datasets in one three-gate, four-metric figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plot_all_eight_decay_grid import AMAZON, plot

HERE = Path(__file__).resolve().parent
GATES = (("030", 0.3), ("070", 0.7), ("090", 0.9))
DATASETS = {"Steam", "MovieLens1M", "Yelp19", "Yelp23"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/all_eight_gate_grid")
    parser.add_argument("--min-users", type=int, default=100)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    paths = (HERE / "outputs/amazon_gate_grid/plotted_data.json",
             HERE / "outputs/additional_gate_grid/plotted_data.json")
    amazon, additional = (json.loads(path.read_text()) for path in paths)
    gates = {value for _, value in GATES}
    for name, record in (("Amazon", amazon), ("Additional", additional)):
        observed = {entry["gate"] for entry in record["sources"]}
        if observed != gates or len(record["sources"]) != 12:
            raise ValueError(f"{name} source gates are {observed}; expected {gates}")
    if set(amazon["statistics"]) != {row[0] for row in AMAZON}:
        raise ValueError("Amazon source must contain four subsets")
    if set(additional["statistics"]) != {tag for tag, _ in GATES}:
        raise ValueError("Additional source must contain all gate rows")
    if any(set(additional["statistics"][tag]) != DATASETS for tag, _ in GATES):
        raise ValueError("Additional gate rows must each contain four datasets")

    stem = "all_eight_gate_item_period_grid"
    plot(args, [tag for tag, _ in GATES], amazon["statistics"], additional["statistics"],
         values=[value for _, value in GATES], parameter="gate", stem=stem)
    provenance = {
        "figure_status": "measured",
        "gate_values": [value for _, value in GATES],
        "source_files": [{"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in paths],
        "dataset_count": 8,
        "layout": "three gate rows by four metrics; all eight datasets overlaid in each panel",
        "min_users_for_influence": args.min_users,
    }
    (args.output / "plotted_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Saved", args.output / f"{stem}.pdf", flush=True)


if __name__ == "__main__":
    main()
