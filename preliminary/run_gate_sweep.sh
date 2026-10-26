#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
GATES=(0.3 0.7 0.9)

export GPU_IDS=0
for gate in "${GATES[@]}"; do
  tag="$(printf '%03d' "$(awk -v value="$gate" 'BEGIN { print value * 100 }')")"
  preliminary_output="$HERE/outputs_gate$tag"
  grouped_output="$PROJECT/preliminary_grouped/outputs_gate$tag"

  echo "========== fixed Mamba gate $gate: preliminary =========="
  MAMBA_GATE="$gate" OUTPUT_DIR="$preliminary_output" bash "$HERE/run.sh"

  echo "========== fixed Mamba gate $gate: preliminary abc figure =========="
  "$PYTHON_BIN" "$HERE/aggregated/plot_aggregated.py" \
    --analysis-root "$preliminary_output/analysis" \
    --output-dir "$preliminary_output/aggregated" \
    --abc-three-panel

  echo "========== fixed Mamba gate $gate: grouped abc figure =========="
  bash "$PROJECT/preliminary_grouped/run_grouped_preliminary.sh" \
    --sources "$preliminary_output/aggregated/sources.json" \
    --output "$grouped_output"
  "$PYTHON_BIN" "$PROJECT/preliminary_grouped/plot_grouped_row.py" \
    --output "$grouped_output"
done
