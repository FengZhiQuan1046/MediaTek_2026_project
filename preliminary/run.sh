#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataspace/P78123011/miniconda3/envs/py31014/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" "$HERE/experiment.py" \
  --cache-dir "${CACHE_DIR:-$WORKSPACE/cache}" \
  --output-dir "${OUTPUT_DIR:-$HERE/results}" \
  --datasets "${DATASETS:-amazon-all-beauty,amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games}" \
  --max-users "${MAX_USERS:-2000}" \
  --max-lag "${MAX_LAG:-100}" \
  --short-window "${SHORT_WINDOW:-10}" \
  --random-matches "${RANDOM_MATCHES:-20}" \
  --reference-items "${REFERENCE_ITEMS:-20000}" \
  --cluster-items "${CLUSTER_ITEMS:-10000}" \
  --clusters "${CLUSTERS:-32}" \
  --bootstrap "${BOOTSTRAP:-2000}" \
  --seed "${SEED:-25252}" "$@"
