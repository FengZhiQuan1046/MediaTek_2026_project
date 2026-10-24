#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
GPU="${1:?GPU required}"
RUN="${2:?Run label required}"
shift 2
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export EXPECTED_VISIBLE_GPUS=1 REQUESTED_GPU_IDS="$GPU"
export HF_HOME="$WORKSPACE/cache" HF_HUB_CACHE="$WORKSPACE/cache/hub"
export HF_DATASETS_CACHE="$WORKSPACE/cache/datasets" TRANSFORMERS_CACHE="$WORKSPACE/cache/transformers"
export MPLCONFIGDIR="$WORKSPACE/cache/matplotlib" TMPDIR="$WORKSPACE/cache/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$WORKSPACE/cache/training_runs/$RUN"
for dataset in "$@"; do
  "$WORKSPACE/miniconda3/envs/py31014/bin/python" -u "$HERE/additional_datasets.py" "$dataset" \
    --gpu "$GPU" --epochs 3 --output "$HERE/outputs/$RUN/$dataset" \
    > "$WORKSPACE/cache/training_runs/$RUN/${dataset}.log" 2>&1
  result=$?
  echo "$dataset exit=$result"
done
