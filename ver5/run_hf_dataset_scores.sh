#!/usr/bin/env bash
# Run one normalized HF dataset. Scores/logs in outputs_mamba_rl; large artifacts in cache.
# Usage: bash run_hf_dataset_scores.sh GPU_ID DATASET DATA_PATH RUN_ROOT
set -euo pipefail
if [[ $# != 4 ]]; then echo 'Usage: GPU_ID DATASET DATA_PATH RUN_ROOT' >&2; exit 2; fi
GPU_ID="$1"
DATASET="$2"
DATA_PATH="$3"
RUN_ROOT="$4"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
CACHE_DIR="$WORKSPACE_ROOT/cache"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$DATASET" =~ ^(yelp23|yelp19|steam)$ ]] || exit 2
[[ -f "$DATA_PATH" ]] || { echo "Missing input: $DATA_PATH" >&2; exit 2; }
RUN_DIR="$PROJECT_ROOT/outputs_mamba_rl/$DATASET/$(basename "$RUN_ROOT")"
CONSOLE_DIR="$RUN_ROOT/$DATASET"
[[ ! -e "$RUN_DIR/config.json" ]] || { echo "Run already started: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR" "$CONSOLE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export TMPDIR="$CACHE_DIR/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
printf 'status=running\nstarted=%s\n' "$(date -Is)" > "$RUN_DIR/status.txt"
printf 'START %s dataset=%s physical_gpu=%s\n' "$(date -Is)" "$DATASET" "$GPU_ID"
set +e
"$PYTHON_BIN" -u -m src.train_yelp_steam_scores \
  --dataset "$DATASET" --data-path "$DATA_PATH" --cache-dir "$CACHE_DIR" \
  --device cuda:0 --graph-device cuda:0 \
  --specialist-epochs "${SPECIALIST_EPOCHS:-3}" --joint-epochs "${JOINT_EPOCHS:-7}" \
  --rl-coef 0.1 --rl-topk 10 --batch-size "${BATCH_SIZE:-128}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}" --max-transitions 500000 \
  --candidates 64 --short-window 4 --max-history "${MAX_HISTORY:-0}" --preference-count 32 \
  --mamba-model-id state-spaces/mamba-130m-hf --mamba-encode-batch-size 32 \
  --mamba-max-tokens 32 --validate-every-steps 1000 \
  --validation-user-limit 0 --periodic-test-user-limit 0 \
  --seed 25252 --no-save-model-weights --no-generate-reasons \
  --output-dir "$PROJECT_ROOT/outputs_mamba_rl/$DATASET" --output-run-dir "$RUN_DIR" --score-file "$RUN_DIR/${DATASET}_scores.json" \
  --experiment-note 'HF full release; existing ver5 preprocessing; Yelp order=source-row index (timestamps unavailable); Steam order=review date; weights disabled' \
  2>&1 | tee "$CONSOLE_DIR/console.log"
status=${PIPESTATUS[0]}
printf 'exit_code=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$RUN_DIR/status.txt"
exit "$status"
