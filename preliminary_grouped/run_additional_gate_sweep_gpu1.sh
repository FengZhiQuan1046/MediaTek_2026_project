#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="$WORKSPACE/miniconda3/envs/py31014/bin/python"
DATASETS=(Yelp19 Yelp23 Steam MovieLens1M)
GATES=(0.3 0.7 0.9)

export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export EXPECTED_VISIBLE_GPUS=1 REQUESTED_GPU_IDS=1
export HF_HOME="$WORKSPACE/cache" HF_HUB_CACHE="$WORKSPACE/cache/hub"
export HF_DATASETS_CACHE="$WORKSPACE/cache/datasets" TRANSFORMERS_CACHE="$WORKSPACE/cache/transformers"
export MPLCONFIGDIR="$WORKSPACE/cache/matplotlib" TMPDIR="$WORKSPACE/cache/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$WORKSPACE/cache/training_runs/gate_sweep_gpu1"

for gate in "${GATES[@]}"; do
  tag="$(printf '%03d' "$(awk -v value="$gate" 'BEGIN { print value * 100 }')")"
  output="$HERE/outputs_gate$tag"
  mkdir -p "$output"
  touch "$output/.additional_owner"
  for dataset in "${DATASETS[@]}"; do
    echo "========== GPU1 $dataset fixed Mamba gate $gate =========="
    status_file="$output/$dataset/status.json"
    if [[ -f "$status_file" ]] && "$PYTHON_BIN" - "$status_file" "$gate" <<'PY'
import json,sys
status=json.load(open(sys.argv[1]))
raise SystemExit(0 if status.get('status')=='complete' and status.get('mamba_gate')==float(sys.argv[2]) else 1)
PY
    then
      echo "REUSE complete dataset: $dataset gate=$gate"
      continue
    fi
    "$PYTHON_BIN" -u "$HERE/additional_datasets.py" "$dataset" \
      --gpu 1 --epochs 3 --mamba-gate "$gate" --output "$output/$dataset" \
      2>&1 | tee "$WORKSPACE/cache/training_runs/gate_sweep_gpu1/gate${tag}_${dataset}.log"
  done
  "$PYTHON_BIN" "$HERE/count_additional_groups.py" "$output"
  "$PYTHON_BIN" "$HERE/finalize_additional.py" "$output" --mamba-gate "$gate"
done
