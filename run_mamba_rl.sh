#!/usr/bin/env bash
# Usage: bash run_mamba_rl.sh DATASET [CUDA_ID] [DATA_PATH]
# Examples:
#   bash run_mamba_rl.sh movielens-1m 0
#   bash run_mamba_rl.sh amazon-all-beauty 0
#   bash run_mamba_rl.sh yelp19 0 /data/yelp-2019
#   bash run_mamba_rl.sh yelp23 0 /data/yelp-2023/yelp_academic_dataset_review.json
set -euo pipefail

DATASET="${1:-movielens-1m}"
CUDA_ID="${2:-0}"
DATA_PATH="${3:-}"
CACHE_DIR="${CACHE_DIR:-/workspace/P78123011/cache}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SPECIALIST_EPOCHS="${SPECIALIST_EPOCHS:-3}"
COORDINATOR_EPOCHS="${COORDINATOR_EPOCHS:-2}"
RL_EPOCHS="${RL_EPOCHS:-20}"
VALIDATE_EVERY_STEPS="${VALIDATE_EVERY_STEPS:-1000}"
MONITOR_METRIC="${MONITOR_METRIC:-ndcg@10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
LR_PATIENCE="${LR_PATIENCE:-3}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-500000}"
SEED="${SEED:-25252}"
POPULARITY_ALPHA="${POPULARITY_ALPHA:-0.0}"
TRANSITION_BETA="${TRANSITION_BETA:-0.0}"
TARGET_RECALL_AT_10="${TARGET_RECALL_AT_10:-0.15}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ $# -gt 3 || ! "$CUDA_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash run_mamba_rl.sh DATASET [CUDA_ID] [DATA_PATH]" >&2
  exit 2
fi

case "$DATASET" in
  movielens-100k|movielens-1m|movielens-20m|movielens-25m|movielens-32m|movielens-latest-small|amazon-*|amazon:*|yelp19|yelp-2019|yelp23|yelp-2023|synthetic) ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    echo "Use movielens-{100k,1m,20m,25m,32m,latest-small}, amazon-CATEGORY, yelp19, or yelp23." >&2
    exit 2
    ;;
esac
if [[ "$DATASET" == yelp* && -z "$DATA_PATH" ]]; then
  echo "Yelp snapshots require DATA_PATH pointing to the licensed local dataset." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_ID"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"

ARGS=(
  --dataset "$DATASET"
  --cache-dir "$CACHE_DIR"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --specialist-epochs "$SPECIALIST_EPOCHS"
  --coordinator-epochs "$COORDINATOR_EPOCHS"
  --rl-epochs "$RL_EPOCHS"
  --validate-every-steps "$VALIDATE_EVERY_STEPS"
  --monitor-metric "$MONITOR_METRIC"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --lr-patience "$LR_PATIENCE"
  --max-transitions "$MAX_TRANSITIONS"
  --seed "$SEED"
  --popularity-alpha "$POPULARITY_ALPHA"
  --transition-beta "$TRANSITION_BETA"
  --target-recall-at-10 "$TARGET_RECALL_AT_10"
  --experiment-note "$EXPERIMENT_NOTE"
  --device cuda
  --generate-reasons
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "$DATA_PATH" ]]; then
  ARGS+=(--data-path "$DATA_PATH")
fi

python -m src.train_mamba_rl "${ARGS[@]}"
