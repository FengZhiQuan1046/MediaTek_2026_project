#!/usr/bin/env bash
# Usage: bash run_long_short.sh [TRAINING_OPTIONS...]
# Same suite settings, with opt-in length metrics and separate output directories.
# Every subset uses the same model architecture. Training-only hyperparameters
# scale with catalog size; every validation and test evaluates all eligible users.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
# 手動設定要使用的實體 GPU："0"、"1"、"0,1" 或 "1,0"
# 使用兩張卡時，第一張供主模型使用，第二張供 graph/preference 模型使用。
GPU_IDS="${GPU_IDS:-1}"
EVAL_LENGTH_THRESHOLD="${EVAL_LENGTH_THRESHOLD:-10}"
TRAINING_OPTIONS=("$@")
if [[ ! "$EVAL_LENGTH_THRESHOLD" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_LENGTH_THRESHOLD must be a positive integer" >&2
  exit 2
fi

# Hugging Face Mamba 規模：130m、370m、790m、1.4b、2.8b。
# 也可直接覆寫完整 repo，例如：
# MAMBA_MODEL_ID="state-spaces/mamba-1.4b-hf"
MAMBA_MODEL_SIZE="${MAMBA_MODEL_SIZE:-130m}"
MAMBA_MODEL_ID="${MAMBA_MODEL_ID:-state-spaces/mamba-${MAMBA_MODEL_SIZE}-hf}"

# Project preprocessing always applies the one-pass user/item threshold from
# src/data.py before the chronological leave-two-out split.

# ================= 手動調整：訓練模式與三階段超參數 =================
# 1: LoRA；0: full-rank finetuning（推薦模型內的 dense adaptation）
ENABLE_LORA=1
# 指定要存到 outputs_mamba_rl/ 底下的資料夾名稱。
# 例如：OUTPUT_FOLDER_NAME="my_experiment"

SPECIALISTS_EPOCH=3
COORDINATOR_EPOCH=3
JOINT_EPOCH=7
SPECIALISTS_LR=1e-4
COORDINATOR_LR=1e-4
JOINT_LR=5e-5

# Agent ablations: 1 = enabled, 0 = disabled (also overridable via environment).
# USE_LONG=0: all sequence agents (including preference) use only SHORT_WINDOW.
USE_LONG="${USE_LONG:-0}"
USE_SHORT="${USE_SHORT:-1}"
USE_PREFERENCE="${USE_PREFERENCE:-1}"
USE_GCN="${USE_GCN:-1}"


SHORT_WINDOWs=(1)

for SHORT_WINDOW in "${SHORT_WINDOWs[@]}"; do
  OUTPUT_FOLDER_NAME="amazons_long_short"
  # 1: 自動在資料夾名後面追加 L{USE_LONG}_S{USE_SHORT}_P{USE_PREFERENCE}_G{USE_GCN}
  # 0: 完全使用上面的 OUTPUT_FOLDER_NAME
  OUTPUT_APPEND_ABLATION_TAG="${OUTPUT_APPEND_ABLATION_TAG:-1}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs_mamba_rl/$OUTPUT_FOLDER_NAME}"
  echo "Running with SHORT_WINDOW=$SHORT_WINDOW" 

  for switch in "$USE_LONG" "$USE_SHORT" "$USE_PREFERENCE" "$USE_GCN"; do
    if [[ "$switch" != 0 && "$switch" != 1 ]]; then
      echo "USE_LONG/USE_SHORT/USE_PREFERENCE/USE_GCN must be 0 or 1" >&2
      exit 2
    fi
  done
  if (( USE_LONG + USE_SHORT + USE_PREFERENCE + USE_GCN == 0 )); then
    echo "All agents disabled requires USE_GCN=1" >&2
    exit 2
  fi
  GCN_OPTION="--no-use-graph-embeddings"
  if [[ "$USE_GCN" == 1 ]]; then GCN_OPTION="--use-graph-embeddings"; fi
  ABLATION_TAG="L${USE_LONG}_S${USE_SHORT}_P${USE_PREFERENCE}_G${USE_GCN}_SW${SHORT_WINDOW}"
  if [[ "$OUTPUT_APPEND_ABLATION_TAG" == 1 ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT}_${ABLATION_TAG}"
  fi

  COMMON_ENV=(
    "PYTHON_BIN=$PYTHON_BIN"
    "CACHE_DIR=$CACHE_DIR"
    "ENABLE_LORA=$ENABLE_LORA"
    "SPECIALIST_EPOCHS=$SPECIALISTS_EPOCH"
    "COORDINATOR_EPOCHS=$COORDINATOR_EPOCH"
    "JOINT_EPOCHS=$JOINT_EPOCH"
    "SPECIALIST_LR=$SPECIALISTS_LR"
    "COORDINATOR_LR=$COORDINATOR_LR"
    "JOINT_LR=$JOINT_LR"
    "BATCH_SIZE=128"
    "EVAL_BATCH_SIZE=64"
    "MONITOR_METRIC=ndcg@10"
    "EARLY_STOPPING_PATIENCE=6"
    "LR_PATIENCE=2"
    "DIM=128"
    "LORA_RANK=16"
    "LORA_ALPHA=32.0"
    "LORA_DROPOUT=0.05"
    "SHORT_WINDOW=$SHORT_WINDOW"
    "PREFERENCE_COUNT=32"
    "PREFERENCE_HIDDEN=128"
    "PREFERENCE_TEMPERATURE=0.2"
    "PREFERENCE_SCORE_WEIGHT=0.2"
    "PREFERENCE_COEF=0.2"
    "PREFERENCE_TRANSITION_COEF=0.1"
    "PREFERENCE_BALANCE_COEF=0.01"
    "PREFERENCE_SEPARATION_COEF=0.01"
    "PREFERENCE_SHARPNESS_COEF=0.05"
    "FUTURE_HORIZON=3"
    "FUTURE_DECAY=0.5"
    "HARD_NEGATIVE_POOL_MULTIPLIER=12"
    "HARD_NEGATIVE_FRACTION=0.75"
    "HARD_NEGATIVE_WARMUP_EPOCHS=2"
    "USE_IN_BATCH_NEGATIVES=1"
    "PREFERENCE_CONTRASTIVE_COEF=0.05"
    "AGENT_DIVERSITY_COEF=0.02"
    "USE_GRAPH_EMBEDDINGS=$USE_GCN"
    "MAX_HISTORY=100"
    "MAMBA_ENCODE_BATCH_SIZE=32"
    "MAMBA_MAX_TOKENS=32"
    "MAMBA_MODEL_ID=$MAMBA_MODEL_ID"
    "ITEM_PROMPT_PREFIX=Preference-aware product representation: "
    "GENERATE_REASONS=0"
    "SAVE_MODEL_WEIGHTS=0"
  )

  run_subset() {
    local subset_name="$1"
    local dataset="$2"
    local validate_every_steps="$3"
    local max_transitions="$4"
    local candidates="$5"
    local popularity_alpha="$6"
    local transition_beta="$7"
    local timestamp run_dir

    timestamp="$(date '+%Y%m%d_%H%M%S')"
    run_dir="$OUTPUT_ROOT/$subset_name/rl_${timestamp}_r${run_number}_$$"

    env "${COMMON_ENV[@]}" bash "$PROJECT_ROOT/run_mamba_rl.sh" "$dataset" "$GPU_IDS" \
      --output-run-dir "$run_dir" \
      --score-file "$run_dir/${subset_name}_scores.json" \
      --validate-every-steps "$validate_every_steps" \
      --validation-user-limit 0 \
      --periodic-test-user-limit 0 \
      --max-transitions "$max_transitions" \
      --candidates "$candidates" \
      --specialist-epochs "$SPECIALISTS_EPOCH" \
      --coordinator-epochs "$COORDINATOR_EPOCH" \
      --joint-epochs "$JOINT_EPOCH" \
      --specialist-lr "$SPECIALISTS_LR" \
      --coordinator-lr "$COORDINATOR_LR" \
      --joint-lr "$JOINT_LR" \
      --popularity-alpha "$popularity_alpha" \
      --transition-beta "$transition_beta" \
      --eval-length-threshold "$EVAL_LENGTH_THRESHOLD" \
      --eval-length-basis filtered_full_sequence \
      --experiment-note "$OUTPUT_FOLDER_NAME subset=$subset_name; short filtered-full-sequence <= $EVAL_LENGTH_THRESHOLD; long > $EVAL_LENGTH_THRESHOLD; same model and full catalog" \
      "${TRAINING_OPTIONS[@]}" \
      --use-long "$USE_LONG" --use-short "$USE_SHORT" --use-preference "$USE_PREFERENCE" \
      "$GCN_OPTION"
  }

  # One command per requested subset. Set REPEATS=5 to repeat the whole suite five times.
  REPEATS="${REPEATS:-1}"
  for ((run_number = 1; run_number <= REPEATS; run_number++)); do
    # name dataset validation_steps max_samples candidates popularity transition
    run_subset "Full_Beauty" "amazon-all-beauty" 250 500000 64 -0.25 4.0
    run_subset "Baby_Products" "amazon:Baby_Products" 6000 1500000 192 0.30 0.5
    run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" 8000 1000000 256 0.35 0.5
    # run_subset "Books" "amazon-books" 12000 2000000 256 0.35 0.5
    run_subset "Toys_and_Games" "amazon-toys-and-games" 6000 1500000 192 0.30 0.5
    # run_subset "Video_Games" "amazon-video-games" 4000 1000000 128 0.20 0.5
    # run_subset "Clothing_Shoes_and_Jewelry" "amazon-clothing-shoes-and-jewelry" 12000 2000000 256 0.35 0.5
    # run_subset "Beauty_and_Personal_Care" "amazon:Beauty_and_Personal_Care" 12000 2000000 256 0.35 0.5
  done ;
done