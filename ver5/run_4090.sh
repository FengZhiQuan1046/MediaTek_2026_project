#!/usr/bin/env bash
# Usage: bash run_long_short.sh [TRAINING_OPTIONS...]
# Same suite settings, with opt-in length metrics and separate output directories.
# Every subset uses the same model architecture. Training-only hyperparameters
# scale with catalog size; every validation and test evaluates all eligible users.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python" ]]; then
    PYTHON_BIN="$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python"
  elif [[ -x /dataspace/P78123011/miniconda3/envs/py31014/bin/python ]]; then
    PYTHON_BIN=/dataspace/P78123011/miniconda3/envs/py31014/bin/python
  else
    PYTHON_BIN=python
  fi
fi
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
# 手動設定要使用的實體 GPU："0"、"1"、"0,1" 或 "1,0"
# 使用兩張卡時，第一張供主模型使用，第二張供 graph/preference 模型使用。
GPU_IDS="${GPU_IDS:-0}"
EVAL_LENGTH_THRESHOLD="${EVAL_LENGTH_THRESHOLD:-10}"
TRAINING_OPTIONS=("$@")
if ! bash -n "$PROJECT_ROOT/run_mamba_rl.sh"; then
  echo "run_mamba_rl.sh has a shell syntax error; stopping before the dataset suite." >&2
  exit 2
fi
# Resume an interrupted suite without repeating completed subsets.
START_FROM="${START_FROM:-}"
case "$START_FROM" in
  ""|Full_Beauty|Baby_Products|Sports_and_Outdoors|Toys_and_Games) ;;
  *) echo "Unsupported START_FROM: $START_FROM" >&2; exit 2 ;;
esac
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

# ================= 訓練模式：JOINT_EPOCH=0 時全部模組單階段 DL =================
# 1: LoRA；0: full-rank finetuning（推薦模型內的 dense adaptation）
ENABLE_LORA=1
# 指定要存到 outputs_mamba_rl/ 底下的資料夾名稱。
# 例如：OUTPUT_FOLDER_NAME="my_experiment"

SPECIALISTS_EPOCH="${SPECIALISTS_EPOCH:-4}"
# 設為 0：short、preference、coordinator、GCN 同時 DL 訓練 SPECIALISTS_EPOCH 次。
# 大於 0：先 specialists DL，再 joint DL+RL。
JOINT_EPOCH="${JOINT_EPOCH:-0}"
SPECIALISTS_LR="${SPECIALISTS_LR:-1e-4}"
JOINT_LR="${JOINT_LR:-5e-5}"
RL_COEF="${RL_COEF:-0.1}"
RL_TOPK="${RL_TOPK:-10}"

# Agent ablations: 1 = enabled, 0 = disabled (also overridable via environment).
# Preference uses MAX_HISTORY whenever enabled; only short uses SHORT_WINDOW.
USE_SHORT="${USE_SHORT:-1}"
USE_PREFERENCE="${USE_PREFERENCE:-1}"
USE_GCN="${USE_GCN:-1}"
USE_COORDINATOR="${USE_COORDINATOR:-1}"
PREFERENCE_COUNT="${PREFERENCE_COUNT:-128}"
for ((option_index = 0; option_index < ${#TRAINING_OPTIONS[@]}; option_index++)); do
  case "${TRAINING_OPTIONS[option_index]}" in
    --preference-count)
      if (( option_index + 1 >= ${#TRAINING_OPTIONS[@]} )); then
        echo "--preference-count requires a value" >&2
        exit 2
      fi
      PREFERENCE_COUNT="${TRAINING_OPTIONS[++option_index]}"
      ;;
    --preference-count=*)
      PREFERENCE_COUNT="${TRAINING_OPTIONS[option_index]#*=}"
      ;;
  esac
done


SHORT_WINDOWs=(4)

for SHORT_WINDOW in "${SHORT_WINDOWs[@]}"; do
  OUTPUT_FOLDER_NAME="amazons_long_short"
  # 1: 在資料夾名後追加模組開關、SW{SHORT_WINDOW}_P{PREFERENCE_COUNT}
  # 0: 完全使用上面的 OUTPUT_FOLDER_NAME
  OUTPUT_APPEND_ABLATION_TAG="${OUTPUT_APPEND_ABLATION_TAG:-1}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs_mamba_rl/$OUTPUT_FOLDER_NAME}"
  echo "Running with SHORT_WINDOW=$SHORT_WINDOW" 

  for switch in "$USE_SHORT" "$USE_PREFERENCE" "$USE_GCN" "$USE_COORDINATOR"; do
    if [[ "$switch" != 0 && "$switch" != 1 ]]; then
      echo "USE_SHORT/USE_PREFERENCE/USE_GCN/USE_COORDINATOR must be 0 or 1" >&2
      exit 2
    fi
  done
  if (( USE_SHORT + USE_PREFERENCE + USE_GCN == 0 )); then
    echo "All agents disabled requires USE_GCN=1" >&2
    exit 2
  fi
  if [[ ! "$PREFERENCE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREFERENCE_COUNT must be a positive integer" >&2
    exit 2
  fi
  GCN_OPTION="--no-use-graph-embeddings"
  if [[ "$USE_GCN" == 1 ]]; then GCN_OPTION="--use-graph-embeddings"; fi
  EFFECTIVE_COORDINATOR=0
  if (( USE_SHORT == 1 && USE_PREFERENCE == 1 )); then
    EFFECTIVE_COORDINATOR="$USE_COORDINATOR"
  fi
  COORDINATOR_OPTION="--no-use-coordinator"
  if [[ "$EFFECTIVE_COORDINATOR" == 1 ]]; then COORDINATOR_OPTION="--use-coordinator"; fi
  ABLATION_TAG="S${USE_SHORT}_P${USE_PREFERENCE}_G${USE_GCN}_C${EFFECTIVE_COORDINATOR}_SW${SHORT_WINDOW}_P${PREFERENCE_COUNT}"
  if [[ "$OUTPUT_APPEND_ABLATION_TAG" == 1 ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT}_${ABLATION_TAG}"
  fi

  COMMON_ENV=(
    "PYTHON_BIN=$PYTHON_BIN"
    "CACHE_DIR=$CACHE_DIR"
    "ENABLE_LORA=$ENABLE_LORA"
    "SPECIALIST_EPOCHS=$SPECIALISTS_EPOCH"
    "JOINT_EPOCHS=$JOINT_EPOCH"
    "RL_COEF=$RL_COEF"
    "RL_TOPK=$RL_TOPK"
    "SPECIALIST_LR=$SPECIALISTS_LR"
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
    "PREFERENCE_COUNT=$PREFERENCE_COUNT"
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
    "USE_COORDINATOR=$EFFECTIVE_COORDINATOR"
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

    if [[ "$SKIP_UNTIL_START" == 1 ]]; then
      if [[ "$subset_name" != "$START_FROM" ]]; then
        echo "Skipping completed subset: $subset_name"
        return 0
      fi
      SKIP_UNTIL_START=0
    fi

    # Check again before every subset in case the launcher was edited mid-run.
    if ! bash -n "$PROJECT_ROOT/run_mamba_rl.sh"; then
      echo "run_mamba_rl.sh has a shell syntax error; stopping before $subset_name." >&2
      return 2
    fi

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
      --joint-epochs "$JOINT_EPOCH" \
      --rl-coef "$RL_COEF" \
      --rl-topk "$RL_TOPK" \
      --specialist-lr "$SPECIALISTS_LR" \
      --joint-lr "$JOINT_LR" \
      --popularity-alpha "$popularity_alpha" \
      --transition-beta "$transition_beta" \
      --eval-length-threshold "$EVAL_LENGTH_THRESHOLD" \
      --eval-length-basis filtered_full_sequence \
      --experiment-note "$OUTPUT_FOLDER_NAME subset=$subset_name; short filtered-full-sequence <= $EVAL_LENGTH_THRESHOLD; long > $EVAL_LENGTH_THRESHOLD; same model and full catalog" \
      "${TRAINING_OPTIONS[@]}" \
      --use-short "$USE_SHORT" --use-preference "$USE_PREFERENCE" \
      "$GCN_OPTION" "$COORDINATOR_OPTION"
  }

  # One command per requested subset. Set REPEATS=5 to repeat the whole suite five times.
  REPEATS="${REPEATS:-1}"
  for ((run_number = 1; run_number <= REPEATS; run_number++)); do
    SKIP_UNTIL_START=0
    if [[ -n "$START_FROM" ]]; then SKIP_UNTIL_START=1; fi
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
