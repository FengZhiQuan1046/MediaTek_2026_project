#!/usr/bin/env bash
# Same exact ver5 cached splits; outputs stay in SIGMA/outputs/<dataset>/<run>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_ORDER="${DATASET_ORDER:-steam yelp19 yelp23}"
export SIGMA_MAXLEN=0
export EPOCHS="${EPOCHS:-20}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec bash "$HERE/../run_yelp23yelp19steam.sh" SIGMA "${1:-1}" "${2:-sigma_$(TZ=Asia/Taipei date '+%Y%m%d_%H%M%S')}"
