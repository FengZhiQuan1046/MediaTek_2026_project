#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
GPU_IDS="${GPU_IDS:-1}"
DISTRIBUTED="${DISTRIBUTED:-0}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-64}"
SUBSETS="${SUBSETS:-all}"; REPEATS="${REPEATS:-1}"
DIN_STEPS="${DIN_STEPS:-10000}"; TRAIN_STEPS="${TRAIN_STEPS:-20000}"
DIN_BATCH_SIZE="${DIN_BATCH_SIZE:-128}"; BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"; EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-300}"
MAXLEN="${MAXLEN:-19}"; MIN_HISTORY="${MIN_HISTORY:-4}"; MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
BRANCH_FACTOR="${BRANCH_FACTOR:-0}"; D_MODEL="${D_MODEL:-96}"; NUM_HEADS="${NUM_HEADS:-4}"
SEMANTIC_MODEL_ID="${SEMANTIC_MODEL_ID:-google-t5/t5-base}"
SEMANTIC_BACKEND="${SEMANTIC_BACKEND:-t5}"; SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-32}"
SEMANTIC_MAX_TOKENS="${SEMANTIC_MAX_TOKENS:-64}"; EVAL_USER_LIMIT="${EVAL_USER_LIMIT:-0}"
PREDICT_CANDIDATES="${PREDICT_CANDIDATES:-50}"; SEED="${SEED:-2024}"; MAX_EVENTS="${MAX_EVENTS:-}"
export HF_HOME="$CACHE_DIR" HF_DATASETS_CACHE="$CACHE_DIR/datasets" TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS="," read -r -a GPU_ARRAY <<< "$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -ge 1 ]] || { echo "At least one GPU is required" >&2; exit 2; }

should_run() { [[ "$SUBSETS" == all || ",$SUBSETS," == *",$1,"* ]]; }
run_subset() {
  local subset="$1" dataset="$2" repeat="$3" gpu="$4" timestamp run_dir
  should_run "$subset" || return 0
  timestamp="$(date "+%Y%m%d_%H%M%S")"
  run_dir="$OUTPUT_ROOT/$subset/eager_${timestamp}_r${repeat}"
  mkdir -p "$run_dir"
  echo "Running EAGER subset=$subset dataset=$dataset GPU=$gpu output=$run_dir"
  args=(--dataset "$dataset" --cache-dir "$CACHE_DIR" --output-dir "$run_dir"
    --din-steps "$DIN_STEPS" --train-steps "$TRAIN_STEPS" --din-batch-size "$DIN_BATCH_SIZE"
    --rerank-batch-size "$RERANK_BATCH_SIZE" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE" --eval-every-steps "$EVAL_EVERY_STEPS"
    --maxlen "$MAXLEN" --min-history "$MIN_HISTORY" --max-train-samples "$MAX_TRAIN_SAMPLES"
    --branch-factor "$BRANCH_FACTOR" --d-model "$D_MODEL" --num-heads "$NUM_HEADS"
    --semantic-model-id "$SEMANTIC_MODEL_ID" --semantic-backend "$SEMANTIC_BACKEND"
    --semantic-batch-size "$SEMANTIC_BATCH_SIZE" --semantic-max-tokens "$SEMANTIC_MAX_TOKENS"
    --eval-user-limit "$EVAL_USER_LIMIT" --predict-candidates "$PREDICT_CANDIDATES"
    --seed "$SEED" --device cuda)
  [[ -z "$MAX_EVENTS" ]] || args+=(--max-events "$MAX_EVENTS")
  if [[ "$DISTRIBUTED" == 1 ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
      --standalone --nproc_per_node="${#GPU_ARRAY[@]}" "$PROJECT_ROOT/train.py" "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}"
  fi
}
# "Full_Beauty|amazon-all-beauty" "Baby_Products|amazon:Baby_Products" \
              
mkdir -p "$OUTPUT_ROOT"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  jobs=0
  pids=()
  for spec in "Sports_and_Outdoors|amazon-sports-and-outdoors" "Toys_and_Games|amazon-toys-and-games"; do
    IFS="|" read -r subset dataset <<< "$spec"
    should_run "$subset" || continue
    gpu="${GPU_ARRAY[$((jobs % ${#GPU_ARRAY[@]}))]}"
    if [[ "$DISTRIBUTED" == 1 ]]; then
      run_subset "$subset" "$dataset" "$repeat" "$GPU_IDS"
      continue
    fi
    run_subset "$subset" "$dataset" "$repeat" "$gpu" &
    pids+=("$!")
    ((jobs+=1))
    if (( jobs % ${#GPU_ARRAY[@]} == 0 )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
