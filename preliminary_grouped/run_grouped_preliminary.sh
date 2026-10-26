#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one physical GPU number" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MPLCONFIGDIR="$WORKSPACE/cache/matplotlib"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

# A GPU1 streaming queue may already be grouping each dataset as soon as its
# matching preliminary analysis finishes. Avoid recomputing the same gate when
# the outer preliminary launcher reaches its legacy all-at-once grouped step.
OUTPUT_TARGET=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--output" ]]; then
    OUTPUT_TARGET="$argument"
    break
  fi
  previous="$argument"
done
if [[ -n "$OUTPUT_TARGET" && -e "$OUTPUT_TARGET/.streaming" ]]; then
  echo "GPU1 streaming grouped job owns $OUTPUT_TARGET; waiting for its completion."
  while [[ -e "$OUTPUT_TARGET/.streaming" ]]; do sleep 10; done
fi
if [[ -n "$OUTPUT_TARGET" && -e "$OUTPUT_TARGET/.additional_owner" ]]; then
  echo "Skipping Amazon grouped pass: GPU1 additional-dataset queue owns $OUTPUT_TARGET"
  exit 0
fi
if [[ -n "$OUTPUT_TARGET" && -f "$OUTPUT_TARGET/run_status.json" ]] && \
   grep -q '"status"[[:space:]]*:[[:space:]]*"complete"' "$OUTPUT_TARGET/run_status.json"; then
  echo "Grouped output already complete: $OUTPUT_TARGET"
  exit 0
fi
exec "$WORKSPACE/miniconda3/envs/py31014/bin/python" -u "$HERE/plot_grouped_preliminary.py" \
  --physical-gpu "$GPU_ID" "$@"
