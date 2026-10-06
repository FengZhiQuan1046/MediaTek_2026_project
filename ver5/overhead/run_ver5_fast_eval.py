#!/usr/bin/env python3
"""Run ver5 with the audited fast evaluator, keeping all changes in overhead/.

Pass the same arguments accepted by `python -m src.train_mamba_rl`.
Physical GPU 1 must be bound through CUDA_VISIBLE_DEVICES using its UUID.
"""
import sys

from benchmark_ours_throughput import main

if __name__ == "__main__":
    sys.argv.insert(1, "--fast-run")
    main()
