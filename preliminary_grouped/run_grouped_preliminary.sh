#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$WORKSPACE/cache/matplotlib"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"
exec "$WORKSPACE/miniconda3/envs/py31014/bin/python" -u "$HERE/plot_grouped_preliminary.py" "$@"
