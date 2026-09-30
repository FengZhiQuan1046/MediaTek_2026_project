#!/usr/bin/env bash
# Usage:
#   bash run.sh 0
#   bash run.sh 0,1
#   GPU_IDS=1 SUBSETS=Full_Beauty,Video_Games bash run.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python"
  [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
fi
command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
# Shared with the other Reproduce baselines (workspace/cache).
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
GPU_IDS="${1:-${GPU_IDS:-1}}"

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU IDs must look like 0, 1, or 0,1; got: $GPU_IDS" >&2
  exit 2
fi

# all 跑與 ver4/run_amazons_full_rl.sh 相同的八個 subset；也可用逗號選擇。
SUBSETS="${SUBSETS:-all}"
REPEATS="${REPEATS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"

# BERT4Rec defaults, all overridable from the environment.
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAXLEN="${MAXLEN:-100}"
HIDDEN_UNITS="${HIDDEN_UNITS:-128}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-2}"
DROPOUT_RATE="${DROPOUT_RATE:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MASK_PROB="${MASK_PROB:-0.2}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-128}"
DATASET="${DATASET:-}"
DATA_PATH="${DATA_PATH:-}"
MIN_RATING="${MIN_RATING:-4.0}"
DEVICE="${DEVICE:-cuda}"
export HF_HOME="$CACHE_DIR" HF_DATASETS_CACHE="$CACHE_DIR/datasets" TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
MONITOR_METRIC="${MONITOR_METRIC:-ndcg@10}"
SEED="${SEED:-25252}"
MAX_EVENTS="${MAX_EVENTS:-}"
MAX_BATCHES_PER_EPOCH="${MAX_BATCHES_PER_EPOCH:-0}"
USE_AMP="${USE_AMP:-True}"

boolean_option() {
  local value="$1"
  local true_option="$2"
  local false_option="$3"
  case "$value" in
    True) echo "$true_option" ;;
    False) echo "$false_option" ;;
    *)
      echo "Boolean settings must be True or False, got: $value" >&2
      return 2
      ;;
  esac
}

AMP_OPTION="$(boolean_option "$USE_AMP" --amp --no-amp)"

should_run() {
  local subset_name="$1"
  [[ "$SUBSETS" == "all" || ",$SUBSETS," == *",$subset_name,"* ]]
}

run_subset() {
  local subset_name="$1"
  local dataset="$2"
  local repeat_number="$3"
  local timestamp run_dir
  [[ -n "$DATASET" ]] || should_run "$subset_name" || return 0

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/bert4rec_${timestamp}_r${repeat_number}"
  echo "Running BERT4Rec subset=$subset_name dataset=$dataset GPUs=$GPU_IDS output=$run_dir"

  args=(
    --dataset "$dataset"
    --cache-dir "$CACHE_DIR"
    --output-dir "$run_dir"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --maxlen "$MAXLEN"
    --hidden-units "$HIDDEN_UNITS"
    --num-blocks "$NUM_BLOCKS"
    --num-heads "$NUM_HEADS"
    --dropout-rate "$DROPOUT_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --mask-prob "$MASK_PROB"
    --loss-chunk-size "$LOSS_CHUNK_SIZE"
    --min-rating "$MIN_RATING"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --monitor-metric "$MONITOR_METRIC"
    --max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH"
    --seed "$SEED"
    --device "$DEVICE"
    "$AMP_OPTION"
  )
  if [[ -n "$MAX_EVENTS" ]]; then
    args+=(--max-events "$MAX_EVENTS")
  fi
  [[ -z "$DATA_PATH" ]] || args+=(--data-path "$DATA_PATH")
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}"
}

if [[ -z "$DATASET" && "$SUBSETS" != all ]]; then
  IFS=, read -r -a selected_subsets <<< "$SUBSETS"
  for subset in "${selected_subsets[@]}"; do
    case "$subset" in
      Full_Beauty|Beauty_and_Personal_Care|Baby_Products|Sports_and_Outdoors|Books|Toys_and_Games|Video_Games|Clothing_Shoes_and_Jewelry) ;;
      *) echo "Unknown subset: $subset" >&2; exit 2 ;;
    esac
  done
  [[ ${#selected_subsets[@]} -gt 0 ]] || { echo "SUBSETS cannot be empty" >&2; exit 2; }
fi
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"
for ((repeat_number = 1; repeat_number <= REPEATS; repeat_number++)); do
  if [[ -n "$DATASET" ]]; then
    subset_label="${DATASET//[:\/]/_}"
    run_subset "$subset_label" "$DATASET" "$repeat_number"
    continue
  fi
  # Keep this list synchronized with ver4/run_amazons_full_rl.sh.
  run_subset "Full_Beauty" "amazon-all-beauty" "$repeat_number"
  # run_subset "Beauty_and_Personal_Care" "amazon:Beauty_and_Personal_Care" "$repeat_number"
  run_subset "Baby_Products" "amazon:Baby_Products" "$repeat_number"
  run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" "$repeat_number"
  # run_subset "Books" "amazon-books" "$repeat_number"
  run_subset "Toys_and_Games" "amazon-toys-and-games" "$repeat_number"
  # run_subset "Video_Games" "amazon-video-games" "$repeat_number"
  # run_subset "Clothing_Shoes_and_Jewelry" "amazon-clothing-shoes-and-jewelry" "$repeat_number"
done
