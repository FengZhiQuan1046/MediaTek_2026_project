#!/usr/bin/env bash
# Usage:
#   bash run.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
# ================= 手動調整 GPU =================
# 可填 "0"、"1"、"0,1" 或 "1,0"。
GPU_IDS="0"

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU IDs must look like 0, 1, or 0,1; got: $GPU_IDS" >&2
  exit 2
fi

# Same eight Amazon aliases as ver4/run_amazons_full_rl.sh. Select a comma-
# separated subset list when only part of the suite is needed.
SUBSETS="${SUBSETS:-all}"
REPEATS="${REPEATS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"

# ReSID defaults. Both learned stages use about eight epochs as requested.
FAMAE_EPOCHS="${FAMAE_EPOCHS:-8}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
FAMAE_LEARNING_RATE="${FAMAE_LEARNING_RATE:-1e-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAXLEN="${MAXLEN:-100}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
FAMAE_LAYERS="${FAMAE_LAYERS:-2}"
RECOMMENDER_LAYERS="${RECOMMENDER_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
TEXT_FIELDS="${TEXT_FIELDS:-4}"
TEXT_BUCKETS="${TEXT_BUCKETS:-4096}"
CODEBOOK1_SIZE="${CODEBOOK1_SIZE:-64}"
CODEBOOK2_SIZE="${CODEBOOK2_SIZE:-64}"
FAMAE_ITEM_CANDIDATES="${FAMAE_ITEM_CANDIDATES:-4096}"
SCORE_PAIR_CHUNK_SIZE="${SCORE_PAIR_CHUNK_SIZE:-512}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
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
  local timestamp run_dir
  should_run "$subset_name" || return 0

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/resid_${timestamp}"
  echo "Running ReSID subset=$subset_name dataset=$dataset GPUs=$GPU_IDS output=$run_dir"

  args=(
    --dataset "$dataset"
    --cache-dir "$CACHE_DIR"
    --output-dir "$run_dir"
    --famae-epochs "$FAMAE_EPOCHS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --famae-learning-rate "$FAMAE_LEARNING_RATE"
    --learning-rate "$LEARNING_RATE"
    --maxlen "$MAXLEN"
    --hidden-size "$HIDDEN_SIZE"
    --num-heads "$NUM_HEADS"
    --famae-layers "$FAMAE_LAYERS"
    --recommender-layers "$RECOMMENDER_LAYERS"
    --dropout "$DROPOUT"
    --text-fields "$TEXT_FIELDS"
    --text-buckets "$TEXT_BUCKETS"
    --codebook1-size "$CODEBOOK1_SIZE"
    --codebook2-size "$CODEBOOK2_SIZE"
    --famae-item-candidates "$FAMAE_ITEM_CANDIDATES"
    --score-pair-chunk-size "$SCORE_PAIR_CHUNK_SIZE"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --monitor-metric "$MONITOR_METRIC"
    --max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH"
    --seed "$SEED"
    --device cuda
    "$AMP_OPTION"
  )
  if [[ -n "$MAX_EVENTS" ]]; then
    args+=(--max-events "$MAX_EVENTS")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}"
}

mkdir -p "$OUTPUT_ROOT"
for ((repeat_number = 1; repeat_number <= REPEATS; repeat_number++)); do
  # run_subset "Full_Beauty" "amazon-all-beauty"
  # run_subset "Baby_Products" "amazon:Baby_Products"
  # run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors"
  # run_subset "Books" "amazon-books"
  run_subset "Toys_and_Games" "amazon-toys-and-games"
  # run_subset "Video_Games" "amazon-video-games"
  # run_subset "Clothing_Shoes_and_Jewelry" "amazon-clothing-shoes-and-jewelry"
  # run_subset "Beauty_and_Personal_Care" "amazon:Beauty_and_Personal_Care"
done
