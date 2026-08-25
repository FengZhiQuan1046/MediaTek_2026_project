#!/usr/bin/env bash
# DL baseline launcher. Usage: bash run_dl.sh [CUDA_IDS]
set -euo pipefail

CACHE_DIR="/workspace/P78123011/cache"
SEED="25252"
BATCH_SIZE="128"
CUDA_IDS="${1:-0}"

if [[ $# -gt 1 || ! "$CUDA_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Usage: bash run_dl.sh [CUDA_IDS], e.g. bash run_dl.sh 0,1" >&2
  exit 2
fi

export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
IFS=',' read -ra GPU_IDS <<< "$CUDA_IDS"
GPU_COUNT="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"

TRAIN_ARGS=(
  --cache-dir "$CACHE_DIR"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --subset raw_review_All_Beauty
  --epochs 20
  --device cuda
)

if [[ "$GPU_COUNT" -eq 1 ]]; then
  python -m src.train "${TRAIN_ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="$GPU_COUNT" -m src.train "${TRAIN_ARGS[@]}" --distributed
fi
