#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/P78123011/miniforge3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
DEVICES="${#GPU_ARRAY[@]}"
SUBSETS="${SUBSETS:-all}"; REPEATS="${REPEATS:-1}"
LLAMA_CACHE_ROOT="$CACHE_DIR/models--meta-llama--Llama-2-7b-hf"
if [[ -z "${LLM_PATH:-}" && -f "$LLAMA_CACHE_ROOT/refs/main" ]]; then
  LLAMA_REVISION="$(<"$LLAMA_CACHE_ROOT/refs/main")"
  LLM_PATH="$LLAMA_CACHE_ROOT/snapshots/$LLAMA_REVISION"
else
  LLM_PATH="${LLM_PATH:-meta-llama/Llama-2-7b-hf}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"; ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"; MAXLEN="${MAXLEN:-10}"; CANS_NUM="${CANS_NUM:-10}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"; EVAL_USER_LIMIT="${EVAL_USER_LIMIT:-1000}"
REC_EPOCHS="${REC_EPOCHS:-10}"; REC_BATCH_SIZE="${REC_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"; SEED="${SEED:-1234}"; MAX_EVENTS="${MAX_EVENTS:-}"
export HF_HOME="$CACHE_DIR" HF_DATASETS_CACHE="$CACHE_DIR/datasets" TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

should_run() { [[ "$SUBSETS" == all || ",$SUBSETS," == *",$1,"* ]]; }
run_subset() {
  local subset="$1" dataset="$2" repeat="$3" timestamp run_dir log_path
  should_run "$subset" || return 0
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset/llara_${timestamp}_r${repeat}"
  mkdir -p "$run_dir"; log_path="$run_dir/train_${timestamp}.log"
  echo "Running LLaRA subset=$subset dataset=$dataset GPUs=$GPU_IDS output=$run_dir"
  args=(--dataset "$dataset" --cache-dir "$CACHE_DIR" --output-dir "$run_dir"
    --llm-path "$LLM_PATH" --devices "$DEVICES" --batch-size "$BATCH_SIZE"
    --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES" --max-epochs "$MAX_EPOCHS"
    --maxlen "$MAXLEN" --cans-num "$CANS_NUM" --max-train-samples "$MAX_TRAIN_SAMPLES"
    --eval-user-limit "$EVAL_USER_LIMIT" --rec-epochs "$REC_EPOCHS"
    --rec-batch-size "$REC_BATCH_SIZE" --num-workers "$NUM_WORKERS" --seed "$SEED")
  [[ -z "$MAX_EVENTS" ]] || args+=(--max-events "$MAX_EVENTS")
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}" 2>&1 | tee "$log_path"
}

mkdir -p "$OUTPUT_ROOT"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  run_subset "Full_Beauty" "amazon-all-beauty" "$repeat"
  run_subset "Baby_Products" "amazon:Baby_Products" "$repeat"
  run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" "$repeat"
  run_subset "Toys_and_Games" "amazon-toys-and-games" "$repeat"
done
