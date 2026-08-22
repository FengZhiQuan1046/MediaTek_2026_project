#!/usr/bin/env bash
# Usage: ./run.sh <cache-dir> <seed> [additional src.train arguments]
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <cache-dir> <seed> [additional src.train arguments]" >&2
  exit 2
fi

CACHE_DIR="$1"
SEED="$2"
shift 2

# These environment variables also cover any Hugging Face subcomponents.
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"

python -m src.train --cache-dir "$CACHE_DIR" --seed "$SEED" "$@"
