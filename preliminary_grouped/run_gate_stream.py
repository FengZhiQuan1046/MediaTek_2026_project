"""Group each gate/dataset on GPU1 immediately after its preliminary analysis completes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "ver4"))
import plot_grouped_preliminary as grouped  # noqa: E402
DATASETS = [
    ("amazon-all-beauty", "amazon-all-beauty", "All_Beauty"),
    ("amazon_Baby_Products", "amazon:Baby_Products", "Baby_Products"),
    ("amazon-sports-and-outdoors", "amazon-sports-and-outdoors", "Sports_and_Outdoors"),
    ("amazon-toys-and-games", "amazon-toys-and-games", "Toys_and_Games"),
]
GATES = (("030", 0.3), ("070", 0.7), ("090", 0.9))


def complete_run(folder: Path) -> Path | None:
    for run in sorted(folder.glob("run_*"), reverse=True):
        manifest = run / "manifest.json"
        try:
            if json.loads(manifest.read_text()).get("status") == "complete":
                return run
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
    return None


def save_status(path: Path, **values):
    path.write_text(json.dumps(values, indent=2) + "\n")


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Launch this streaming grouped queue with CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU1 isolation failed: exactly one logical CUDA device is required")
    torch.cuda.set_device(0)
    for tag, gate in GATES:
        analysis = PROJECT / "preliminary" / f"outputs_gate{tag}" / "analysis"
        output = HERE / f"outputs_gate{tag}"
        output.mkdir(parents=True, exist_ok=True)
        marker = output / ".streaming"
        marker.touch()
        status = output / "stream_status.json"
        reports = {}
        try:
            for folder, dataset, category in DATASETS:
                save_status(status, status="waiting", gate=gate, physical_gpu=1,
                            completed=list(reports), waiting_for=dataset)
                run = None
                while run is None:
                    run = complete_run(analysis / folder)
                    if run is None:
                        time.sleep(10)
                save_status(status, status="running", gate=gate, physical_gpu=1,
                            completed=list(reports), dataset=dataset, source=str(run))
                args = SimpleNamespace(
                    cache_dir=PROJECT.parent / "cache", output=output, seed=25252,
                    bootstrap=2000, min_users=100, device="cuda:0", physical_gpu=1,
                )
                reports[category] = grouped.calculate(
                    {"category": category, "dataset": dataset, "source": str(run)}, args, HERE
                )
            grouped.plot(reports, args)
            save_status(output / "run_status.json", status="complete", gate=gate,
                        physical_gpu=1, figures=5, formats=["png", "pdf", "svg"],
                        subsets=list(reports), bootstrap=args.bootstrap, streaming=True)
            save_status(status, status="complete", gate=gate, physical_gpu=1,
                        completed=list(reports))
        except Exception as error:
            save_status(status, status="failed", gate=gate, physical_gpu=1,
                        completed=list(reports), error=repr(error))
            raise
        finally:
            marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
