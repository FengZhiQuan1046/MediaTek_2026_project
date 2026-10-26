#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
DECAYS=(1.0 0.7 0.5)

export GPU_IDS=0
for decay in "${DECAYS[@]}"; do
  case "$decay" in
    1.0) tag=100 ;;
    0.7) tag=070 ;;
    0.5) tag=050 ;;
  esac
  output="$HERE/outputs_decay$tag"
  if [[ "$decay" == 1.0 ]]; then
    # All Beauty finished before the epoch correction; keep that completed run.
    datasets="amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games"
    "$PYTHON_BIN" - "$output" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'training/preparation_manifest.json').read_text())
assert any(row['dataset'] == 'amazon-all-beauty' and row['status'] == 'trained'
           and row['training_epochs'] == 3 for row in manifest)
runs = sorted((root / 'analysis/amazon-all-beauty').glob('run_*/manifest.json'))
assert runs and any(json.loads(path.read_text())['status'] == 'complete' for path in runs)
PY
  else
    datasets="amazon-all-beauty,amazon:Baby_Products,amazon-sports-and-outdoors,amazon-toys-and-games"
  fi
  echo "========== GPU0 Amazon: gate=0.5 recurrent_decay=$decay =========="
  DATASETS="$datasets" TRAIN_EPOCHS=2 MAMBA_GATE=0.5 MAMBA_DECAY="$decay" OUTPUT_DIR="$output" bash "$HERE/run.sh"
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
