#!/usr/bin/env python3
"""Run one GPU-1-only command and sample its GPU process memory with nvidia-smi."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.PIPE).strip()


def live_pids(root):
    """Linux descendants of the launched command; no dependency on psutil."""
    parent = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            parent[int(proc.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    descendants = {root}
    while True:
        expanded = descendants | {pid for pid, ppid in parent.items() if ppid in descendants}
        if expanded == descendants:
            return descendants
        descendants = expanded


def process_memory(uuid):
    output = smi("-i", uuid, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits")
    result = {}
    for line in output.splitlines():
        try:
            pid, mib = (part.strip() for part in line.split(",", 1))
            result[int(pid)] = int(mib)
        except ValueError:
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for the profiled Python command")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="Extra child environment variable")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command or args.interval <= 0:
        parser.error("provide -- COMMAND and a positive --interval")
    if not Path(args.command[0]).name.startswith("python"):
        parser.error("use a direct Python command; shell launchers may reset CUDA_VISIBLE_DEVICES")
    if not shutil.which("nvidia-smi"):
        parser.error("nvidia-smi is unavailable; no GPU measurements were made")
    try:
        uuid = smi("-i", "1", "--query-gpu=uuid", "--format=csv,noheader").splitlines()[0].strip()
    except (subprocess.CalledProcessError, IndexError) as exc:
        parser.error(f"physical GPU 1 is unavailable: {exc}")
    if not uuid.startswith("GPU-"):
        parser.error(f"unexpected GPU UUID: {uuid!r}")
    env = os.environ.copy()
    for assignment in args.env:
        if "=" not in assignment:
            parser.error(f"invalid --env assignment: {assignment!r}")
        key, value = assignment.split("=", 1)
        if key in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES"}:
            parser.error(f"--env cannot override GPU isolation: {key}")
        env[key] = value
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.output.mkdir(parents=True, exist_ok=True)
    child_cwd = args.cwd.resolve()
    if not child_cwd.is_dir():
        parser.error(f"working directory does not exist: {child_cwd}")
    start = time.monotonic()
    proc = subprocess.Popen(args.command, env=env, cwd=child_cwd, start_new_session=True)
    samples = []
    error = None
    try:
        while True:
            pids = live_pids(proc.pid)
            try:
                gpu_memory = process_memory(uuid)
                matched = {pid: mib for pid, mib in gpu_memory.items() if pid in pids}
                samples.append((time.monotonic() - start, sum(matched.values()), len(matched)))
            except subprocess.CalledProcessError as exc:
                error = str(exc)
            if proc.poll() is not None:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
    elapsed = time.monotonic() - start
    with (args.output / "gpu1_samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("elapsed_seconds", "tracked_process_memory_mib", "matched_gpu_processes"))
        writer.writerows(samples)
    peak = max((mib for _, mib, count in samples if count), default=None)
    result = dict(name=args.name, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  physical_gpu_index=1, gpu_uuid=uuid, command=args.command,
                  working_directory=str(child_cwd),
                  exit_code=proc.returncode, wall_seconds=elapsed,
                  sampled_process_peak_mib=peak, samples=len(samples),
                  sampling_interval_seconds=args.interval, sampling_error=error,
                  caveat="NVML process memory is sampled, not CUDA allocator peak; brief spikes may be missed.")
    (args.output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
