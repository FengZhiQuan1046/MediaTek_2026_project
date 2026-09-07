#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/P78123011/miniforge3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_IDS="${1:-${GPU_IDS:-0,1}}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU IDs: $GPU_IDS" >&2; exit 2; }

SUBSETS="${SUBSETS:-all}"
REPEATS="${REPEATS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
# True: the same state-spaces/mamba-2.8b-hf checkpoint used by ver4.
# False: authors' small Mamba4Rec.
USE_LARGE_MAMBA="${USE_LARGE_MAMBA:-False}"
LARGE_MODEL_ID="${LARGE_MODEL_ID:-state-spaces/mamba-2.8b-hf}"
LARGE_MODEL_DTYPE="${LARGE_MODEL_DTYPE:-bfloat16}"
case "$USE_LARGE_MAMBA" in
  True)
    LARGE_MAMBA_OPTION=--use-large-mamba
    DEFAULT_BATCH_SIZE=1; DEFAULT_EVAL_BATCH_SIZE=1; DEFAULT_LEARNING_RATE=1e-5
    if [[ "$GPU_IDS" == *,* ]]; then
      LARGE_DEVICE_MAP="${LARGE_DEVICE_MAP:-balanced}"
    else
      LARGE_DEVICE_MAP="${LARGE_DEVICE_MAP:-none}"
    fi
    ;;
  False)
    LARGE_MAMBA_OPTION=--no-use-large-mamba
    DEFAULT_BATCH_SIZE=128; DEFAULT_EVAL_BATCH_SIZE=64; DEFAULT_LEARNING_RATE=1e-3
    LARGE_DEVICE_MAP=none
    ;;
  *) echo "USE_LARGE_MAMBA must be True or False" >&2; exit 2 ;;
esac
EPOCHS="${EPOCHS:-15}"; BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"; EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEFAULT_EVAL_BATCH_SIZE}"
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"; WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAXLEN="${MAXLEN:-100}"; HIDDEN_SIZE="${HIDDEN_SIZE:-64}"; NUM_LAYERS="${NUM_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.2}"; D_STATE="${D_STATE:-32}"; D_CONV="${D_CONV:-4}"; EXPAND="${EXPAND:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"; MONITOR_METRIC="${MONITOR_METRIC:-ndcg@10}"
SEED="${SEED:-25252}"; MAX_EVENTS="${MAX_EVENTS:-}"; MAX_BATCHES_PER_EPOCH="${MAX_BATCHES_PER_EPOCH:-0}"
USE_AMP="${USE_AMP:-True}"
case "$USE_AMP" in True) AMP_OPTION=--amp ;; False) AMP_OPTION=--no-amp ;; *) echo "USE_AMP must be True or False" >&2; exit 2 ;; esac

should_run() { [[ "$SUBSETS" == all || ",$SUBSETS," == *",$1,"* ]]; }
run_subset() {
  local subset_name="$1" dataset="$2" repeat_number="$3" timestamp run_dir
  should_run "$subset_name" || return 0
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/mamba4rec_${timestamp}_r${repeat_number}"
  echo "Running Mamba4Rec subset=$subset_name dataset=$dataset GPUs=$GPU_IDS output=$run_dir"
  args=(--dataset "$dataset" --cache-dir "$CACHE_DIR" --output-dir "$run_dir" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE" --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY" --maxlen "$MAXLEN" --hidden-size "$HIDDEN_SIZE" --num-layers "$NUM_LAYERS"
    --dropout "$DROPOUT" --d-state "$D_STATE" --d-conv "$D_CONV" --expand "$EXPAND"
    "$LARGE_MAMBA_OPTION" --large-model-id "$LARGE_MODEL_ID" --large-model-dtype "$LARGE_MODEL_DTYPE"
    --large-device-map "$LARGE_DEVICE_MAP"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" --monitor-metric "$MONITOR_METRIC"
    --max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH" --seed "$SEED" --device cuda "$AMP_OPTION")
  [[ -z "$MAX_EVENTS" ]] || args+=(--max-events "$MAX_EVENTS")
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}"
}

mkdir -p "$OUTPUT_ROOT"
for ((repeat_number=1; repeat_number<=REPEATS; repeat_number++)); do
  run_subset "Full_Beauty" "amazon-all-beauty" "$repeat_number"
  run_subset "Baby_Products" "amazon:Baby_Products" "$repeat_number"
  run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" "$repeat_number"
  run_subset "Toys_and_Games" "amazon-toys-and-games" "$repeat_number"
done
