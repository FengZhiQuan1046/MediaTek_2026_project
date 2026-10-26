"""Render the 0.5/0.7/1.0 decay grids after both GPU sweeps finish."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DECAYS = (("050", 0.5), ("070", 0.7), ("100", 1.0))
SESSIONS = ("preliminary_decay_high_gpu0", "additional_decay_high_gpu1")


def readiness():
    status = {}
    for tag, decay in DECAYS:
        amazon = PROJECT / "preliminary" / f"outputs_decay{tag}" / "aggregated" / "sources.json"
        additional = HERE / f"outputs_decay{tag}" / "completion.json"
        for label, path in ((f"Amazon decay={decay:.1f}", amazon),
                            (f"Additional decay={decay:.1f}", additional)):
            if not path.exists():
                status[label] = "pending"
                continue
            try:
                record = json.loads(path.read_text())
                valid = (record.get("mamba_gate") == 0.5
                         and record.get("mamba_decay") == decay)
                if label.startswith("Amazon"):
                    valid &= len(record.get("sources", [])) == 4
                else:
                    valid &= record.get("status") == "complete"
                    valid &= set(record.get("datasets", [])) == {
                        "Steam", "MovieLens1M", "Yelp19", "Yelp23"}
                status[label] = "ready" if valid else "invalid metadata"
            except (OSError, ValueError) as error:
                status[label] = f"unreadable: {error}"
    return status


def session_running(name):
    return subprocess.run(("tmux", "has-session", "-t", name),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        status = readiness()
        if status != previous:
            print(json.dumps(status, indent=2), flush=True)
            previous = status
        if all(value == "ready" for value in status.values()):
            break
        if args.check_only:
            return 1
        if not any(session_running(name) for name in SESSIONS):
            raise RuntimeError("Both GPU sweeps ended before all six result sets completed")
        time.sleep(60)

    for script in ("plot_amazon_decay_grid.py", "plot_additional_decay_grid.py"):
        subprocess.run((sys.executable, str(HERE / script)), cwd=PROJECT, check=True)
    subprocess.run((sys.executable, str(HERE / "plot_all_eight_decay_grid.py"),
                    "--source", "measured"), cwd=PROJECT, check=True)
    subprocess.run((sys.executable, str(HERE / "plot_all_eight_gate_decay_grid.py"),
                    "--decay-source", "measured"), cwd=PROJECT, check=True)
    for label in ("amazon", "additional"):
        output = HERE / "outputs" / f"{label}_decay_grid"
        plotted = json.loads((output / "plotted_data.json").read_text())
        assert {row["decay"] for row in plotted["sources"]} == {0.5, 0.7, 1.0}
        assert len(plotted["sources"]) == 12
        assert (output / f"{label}_decay_item_period_grid.pdf").exists()
    combined = HERE / "outputs/all_eight_decay_grid"
    combined_source = json.loads((combined / "plotted_data.json").read_text())
    assert combined_source["figure_status"] == "measured"
    assert (combined / "all_eight_decay_item_period_grid.pdf").exists()
    gate_decay = HERE / "outputs/all_eight_gate_decay_grid"
    gate_decay_source = json.loads((gate_decay / "plotted_data.json").read_text())
    assert gate_decay_source["figure_status"] == "all_measured"
    assert (gate_decay / "all_eight_gate_decay_item_period_grid.pdf").exists()
    print("COMPLETE: all decay grids and the combined gate/decay grid use measured data", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
