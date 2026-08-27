#!/usr/bin/env bash
# Run from anywhere: bash /workspace/P78123011/MediaTek_2026_project/run_amazons_full_rl.sh
# Each invocation below uses the same architecture/hyperparameters. Only the
# validation/test sample sizes change with dataset size. Periodic test is
# monitoring-only; the final test evaluates every eligible user.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/outputs_mamba_rl/amazons_full"
PYTHON_BIN="${PYTHON_BIN:-/home/P78123011/miniforge3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-/workspace/P78123011/cache}"

COMMON_ENV=(
  "PYTHON_BIN=$PYTHON_BIN"
  "CACHE_DIR=$CACHE_DIR"
  "BATCH_SIZE=256"
  "EVAL_BATCH_SIZE=64"
  "MAX_TRANSITIONS=500000"
  "MIN_TRANSITIONS_PER_USER=2"
  "SPECIALIST_EPOCHS=5"
  "COORDINATOR_EPOCHS=3"
  "RL_EPOCHS=12"
  "MONITOR_METRIC=ndcg@10"
  "EARLY_STOPPING_PATIENCE=4"
  "LR_PATIENCE=2"
  "CANDIDATES=256"
  "DIM=128"
  "ID_BUCKETS=262144"
  "LORA_RANK=8"
  "LORA_ALPHA=16.0"
  "LORA_DROPOUT=0.05"
  "SHORT_WINDOW=3"
  "MAX_HISTORY=100"
  "SPECIALIST_LR=2e-4"
  "COORDINATOR_LR=2e-4"
  "JOINT_LR=5e-5"
  "ENTROPY_COEF=0.001"
  "RL_COEF=0.05"
  "SUPERVISED_COEF=1.0"
  "HARD_NEGATIVE_COEF=0.5"
  "SPECIALIZATION_COEF=0.001"
  "POPULARITY_ALPHA=0.0"
  "TRANSITION_BETA=0.0"
  "MAMBA_ENCODE_BATCH_SIZE=32"
  "MAMBA_MAX_TOKENS=32"
  "GENERATE_REASONS=0"
  "SAVE_MODEL_WEIGHTS=0"
)

run_subset() {
  local subset_name="$1"
  local dataset="$2"
  local validate_every_steps="$3"
  local validation_user_limit="$4"
  local periodic_test_user_limit="$5"
  local run_seed="$6"
  local timestamp run_dir

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/rl_$timestamp"

  env "${COMMON_ENV[@]}" bash "$PROJECT_ROOT/run_mamba_rl.sh" "$dataset" 0 \
    --output-run-dir "$run_dir" \
    --score-file "$run_dir/${subset_name}_scores.json" \
    --validate-every-steps "$validate_every_steps" \
    --validation-user-limit "$validation_user_limit" \
    --periodic-test-user-limit "$periodic_test_user_limit" \
    --seed "$run_seed" \
    --experiment-note "amazons_full subset=$subset_name; common architecture; periodic validation/test"
}

# One command per requested subset. Set REPEATS=5 to repeat the whole suite five times.
REPEATS="${REPEATS:-1}"
for ((run_number = 1; run_number <= REPEATS; run_number++)); do
  run_seed=$((25251 + run_number))
  run_subset "Beauty" "amazon-all-beauty" 0 0 0 "$run_seed"
  run_subset "Sports" "amazon-sports-and-outdoors" 0 20000 10000 "$run_seed"
  run_subset "Games" "amazon-games" 0 15000 7500 "$run_seed"
  run_subset "Books" "amazon-books" 0 30000 15000 "$run_seed"
  run_subset "Toys" "amazon-toys" 0 25000 12500 "$run_seed"
  run_subset "Video_Games" "amazon-video-games" 0 15000 7500 "$run_seed"
  run_subset "Clothing" "amazon-clothing-shoes-and-jewelry" 0 30000 15000 "$run_seed"
done
