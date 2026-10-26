#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/outputs}"
DATASETS="${DATASETS:-amazon-all-beauty,amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games}"
# Physical CUDA device(s), for example GPU_IDS=0 or GPU_IDS=1,2.
GPU_IDS="${GPU_IDS:-1}"
# The GPU1 decay=0.01 sweep owns this output once it has claimed it.  The
# already-running GPU0 sweep calls this script later; wait for its result
# instead of launching a second training run into the same directory.
if [[ "$GPU_IDS" == 0 && "${MAMBA_GATE:-}" == 0.5 && "${MAMBA_DECAY:-}" == 0.01 && -f "$OUTPUT_DIR/.gpu1_claimed" ]]; then
  echo "GPU1 owns Amazon gate=0.5 decay=0.01; waiting for its result."
  while [[ -f "$OUTPUT_DIR/.gpu1_active" ]]; do sleep 30; done
  if [[ ! -f "$OUTPUT_DIR/.gpu1_complete" ]]; then
    echo "GPU1 decay=0.01 run ended without completing; see $HERE/decay001_gpu1.log" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$OUTPUT_DIR/aggregated/sources.json" <<'PY'
import json, sys
source = json.load(open(sys.argv[1]))
assert source['mamba_gate'] == 0.5 and source['mamba_decay'] == 0.01, source
PY
  echo "GPU1 completed Amazon gate=0.5 decay=0.01; skipping duplicate GPU0 training."
  exit 0
fi
# amazon-all-beauty,amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games,
# amazon:Musical_Instruments,amazon:Video_Games,amazon:Software
# Training settings: edit these defaults here, or override them as environment variables.
LORA_RANK="${LORA_RANK:-16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "GPU_IDS must contain one or two physical GPU numbers, for example 1 or 1,3" >&2
  exit 2
fi
if [[ "$GPU_IDS" == *,* && "${GPU_IDS%%,*}" == "${GPU_IDS##*,}" ]]; then
  echo "GPU_IDS must not contain duplicate GPUs" >&2
  exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export REQUESTED_GPU_IDS="$GPU_IDS"
if [[ "$GPU_IDS" == *,* ]]; then
  export EXPECTED_VISIBLE_GPUS=2
else
  export EXPECTED_VISIBLE_GPUS=1
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

GATE_ARGS=()
if [[ -n "${MAMBA_GATE:-}" ]]; then
  GATE_ARGS=(--mamba-gate "$MAMBA_GATE")
fi
DECAY_ARGS=()
if [[ -n "${MAMBA_DECAY:-}" ]]; then
  DECAY_ARGS=(--mamba-decay "$MAMBA_DECAY")
fi

IFS="," read -r -a DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  [[ -n "$dataset" ]] || continue
  safe_dataset="${dataset//:/_}"
  safe_dataset="${safe_dataset//\//_}"

  echo "========== preliminary subset: $dataset (training) =========="
  "$PYTHON_BIN" "$HERE/prepare_missing.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR/training" \
    --datasets "$dataset" \
    --gpu-ids "$GPU_IDS" \
    --python-bin "$PYTHON_BIN" \
    --mamba-model-id "${MAMBA_MODEL_ID:-state-spaces/mamba-130m-hf}" \
    --lora-rank "$LORA_RANK" \
    --epochs "$TRAIN_EPOCHS" \
    --max-history "${MAX_LAG:-100}" \
    --influence-user-limit "${MAX_USERS:-2000}" \
    --validation-user-limit "${TRAIN_VALIDATION_USER_LIMIT:-0}" \
    --seed "${SEED:-25252}" \
    "${GATE_ARGS[@]}" "${DECAY_ARGS[@]}"

  echo "========== preliminary subset: $dataset (statistics and figures) =========="
  "$PYTHON_BIN" "$HERE/experiment.py" \
    --cache-dir "$CACHE_DIR" \
    --training-output-dir "$OUTPUT_DIR/training" \
    --output-dir "$OUTPUT_DIR/analysis/$safe_dataset" \
    --datasets "$dataset" \
    --max-users "${MAX_USERS:-2000}" \
    --max-lag "${MAX_LAG:-100}" \
    --short-window "${SHORT_WINDOW:-10}" \
    --random-matches "${RANDOM_MATCHES:-20}" \
    --reference-items "${REFERENCE_ITEMS:-20000}" \
    --normalization-pairs "${NORMALIZATION_PAIRS:-100000}" \
    --cluster-items "${CLUSTER_ITEMS:-10000}" \
    --clusters "${CLUSTERS:-32}" \
    --bootstrap "${BOOTSTRAP:-2000}" \
    --seed "${SEED:-25252}" \
    "${GATE_ARGS[@]}" "${DECAY_ARGS[@]}" "$@"
done
