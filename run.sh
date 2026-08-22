#!/usr/bin/env bash
# Run with: bash run.sh
set -euo pipefail

CACHE_DIR="/home/P78123011/cache"
SEED="25252"

# These environment variables also cover any Hugging Face subcomponents.
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"

python -m src.train \
  --cache-dir "$CACHE_DIR" \
  --seed "$SEED" \
  --subset raw_review_All_Beauty \
  --max-events 12000 \
  --epochs 5 \
  --device cuda
