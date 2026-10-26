#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
OUTPUT_DIR="$HERE/outputs_decay001"
mkdir -p "$OUTPUT_DIR"
if [[ -e "$OUTPUT_DIR/.gpu1_claimed" ]]; then
  echo "GPU1 has already claimed $OUTPUT_DIR" >&2
  exit 1
fi
touch "$OUTPUT_DIR/.gpu1_active" "$OUTPUT_DIR/.gpu1_claimed"
trap 'rm -f "$OUTPUT_DIR/.gpu1_active"' EXIT

echo "========== GPU1 Amazon: gate=0.5 recurrent_decay=0.01 =========="
GPU_IDS=1 MAMBA_GATE=0.5 MAMBA_DECAY=0.01 OUTPUT_DIR="$OUTPUT_DIR" bash "$HERE/run.sh"
"$PYTHON_BIN" "$HERE/aggregated/plot_aggregated.py" \
  --analysis-root "$OUTPUT_DIR/analysis" --output-dir "$OUTPUT_DIR/aggregated" \
  --abc-three-panel
"$PYTHON_BIN" - "$OUTPUT_DIR/aggregated/sources.json" <<'PY'
import json, sys
source = json.load(open(sys.argv[1]))
assert source['mamba_gate'] == 0.5 and source['mamba_decay'] == 0.01, source
print('COMPLETE Amazon gate=0.5 decay=0.01 on GPU1', flush=True)
PY
touch "$OUTPUT_DIR/.gpu1_complete"
