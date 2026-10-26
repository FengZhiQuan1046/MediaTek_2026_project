#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="$WORKSPACE/miniconda3/envs/py31014/bin/python"
DATASETS=(Yelp19 Yelp23 Steam MovieLens1M)
DECAYS=(1.0 0.7 0.5)

export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export EXPECTED_VISIBLE_GPUS=1 REQUESTED_GPU_IDS=1
export HF_HOME="$WORKSPACE/cache" HF_HUB_CACHE="$WORKSPACE/cache/hub"
export HF_DATASETS_CACHE="$WORKSPACE/cache/datasets" TRANSFORMERS_CACHE="$WORKSPACE/cache/transformers"
export MPLCONFIGDIR="$WORKSPACE/cache/matplotlib" TMPDIR="$WORKSPACE/cache/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$WORKSPACE/cache/training_runs/decay_high_sweep_gpu1"

for decay in "${DECAYS[@]}"; do
  case "$decay" in
    1.0) tag=100 ;;
    0.7) tag=070 ;;
    0.5) tag=050 ;;
  esac
  output="$HERE/outputs_decay$tag"
  mkdir -p "$output"
  for dataset in "${DATASETS[@]}"; do
    status_file="$output/$dataset/status.json"
    if [[ -f "$status_file" ]] && "$PYTHON_BIN" - "$status_file" "$decay" <<'PY'
import json,sys
status=json.load(open(sys.argv[1]))
raise SystemExit(0 if status.get('status')=='complete' and status.get('mamba_gate')==0.5
                 and status.get('mamba_decay')==float(sys.argv[2]) else 1)
PY
    then
      echo "REUSE complete dataset: $dataset gate=0.5 decay=$decay"
      continue
    fi
    echo "========== GPU1 $dataset: gate=0.5 recurrent_decay=$decay =========="
    "$PYTHON_BIN" -u "$HERE/additional_datasets.py" "$dataset" \
      --gpu 1 --epochs 2 --mamba-gate 0.5 --mamba-decay "$decay" \
      --output "$output/$dataset" \
      2>&1 | tee "$WORKSPACE/cache/training_runs/decay_high_sweep_gpu1/decay${tag}_${dataset}.log"
  done
  "$PYTHON_BIN" "$HERE/count_additional_groups.py" "$output"
  "$PYTHON_BIN" "$HERE/finalize_additional.py" "$output" \
    --mamba-gate 0.5 --mamba-decay "$decay"
done
