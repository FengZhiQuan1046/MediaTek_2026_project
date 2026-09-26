#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/outputs}"
DATASETS="${DATASETS:-amazon:Musical_Instruments,amazon:Video_Games,amazon:Software}"
# Physical CUDA device(s), for example GPU_IDS=0 or GPU_IDS=1,2.
GPU_IDS="${GPU_IDS:-1}"
# amazon-all-beauty,amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games,
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
    --seed "${SEED:-25252}"

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
    --seed "${SEED:-25252}" "$@"
done
