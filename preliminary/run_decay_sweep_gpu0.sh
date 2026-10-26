#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
DECAYS=(0.3 0.1 0.01)

export GPU_IDS=0
for decay in "${DECAYS[@]}"; do
  case "$decay" in
    0.3) tag=030 ;;
    0.1) tag=010 ;;
    0.01) tag=001 ;;
  esac
  output="$HERE/outputs_decay$tag"
  echo "========== GPU0 Amazon: gate=0.5 recurrent_decay=$decay =========="
  MAMBA_GATE=0.5 MAMBA_DECAY="$decay" OUTPUT_DIR="$output" bash "$HERE/run.sh"
  "$PYTHON_BIN" "$HERE/aggregated/plot_aggregated.py" \
    --analysis-root "$output/analysis" --output-dir "$output/aggregated" \
    --abc-three-panel
  "$PYTHON_BIN" - "$output/aggregated/sources.json" "$decay" <<'PY'
import json,sys
source=json.load(open(sys.argv[1]))
assert source['mamba_gate']==0.5 and source['mamba_decay']==float(sys.argv[2]),source
print('COMPLETE Amazon gate=0.5 decay=',sys.argv[2],flush=True)
PY
done
