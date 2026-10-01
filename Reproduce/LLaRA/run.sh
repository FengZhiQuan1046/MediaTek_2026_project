#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
LLARA_UPSTREAM_DIR="${LLARA_UPSTREAM_DIR:-$WORKSPACE_ROOT/LLaRA}"
if [[ ! -f "$LLARA_UPSTREAM_DIR/model/model_interface.py" ]]; then
  echo "LLaRA upstream source is incomplete or missing: $LLARA_UPSTREAM_DIR" >&2
  echo "Expected file: $LLARA_UPSTREAM_DIR/model/model_interface.py" >&2
  exit 2
fi
export LLARA_UPSTREAM_DIR
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python"
  [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
fi
command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
PYTHON_SITE="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
CUSPARSELT_LIB="$PYTHON_SITE/nvidia/cusparselt/lib"
if [[ -f "$CUSPARSELT_LIB/libcusparseLt.so.0" ]]; then
  export LD_LIBRARY_PATH="$CUSPARSELT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"' || {
  echo "PyTorch/CUDA preflight failed for: $PYTHON_BIN" >&2
  exit 2
}
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
GPU_IDS="${1:-${GPU_IDS:-1}}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU IDs: $GPU_IDS" >&2; exit 2; }
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
DEVICES="${#GPU_ARRAY[@]}"
SUBSETS="${SUBSETS:-all}"; REPEATS="${REPEATS:-1}"
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
if [[ -z "${LLM_PATH:-}" ]]; then
  for llama_cache_root in \
      "$CACHE_DIR/models--meta-llama--Llama-2-7b-hf" \
      "$CACHE_DIR/hub/models--meta-llama--Llama-2-7b-hf" \
      "$CACHE_DIR/transformers/models--meta-llama--Llama-2-7b-hf"; do
    if [[ -f "$llama_cache_root/refs/main" ]]; then
      llama_revision="$(<"$llama_cache_root/refs/main")"
      LLM_PATH="$llama_cache_root/snapshots/$llama_revision"
      break
    fi
  done
fi
LLM_PATH="${LLM_PATH:-meta-llama/Llama-2-7b-hf}"
BATCH_SIZE="${BATCH_SIZE:-4}"; ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
MAX_EPOCHS="${EPOCHS:-${MAX_EPOCHS:-5}}"; MAXLEN="${MAXLEN:-10}"; CANS_NUM="${CANS_NUM:-10}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"; EVAL_USER_LIMIT="${EVAL_USER_LIMIT:-1000}"
REC_EPOCHS="${REC_EPOCHS:-10}"; REC_BATCH_SIZE="${REC_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"; SEED="${SEED:-1234}"; MAX_EVENTS="${MAX_EVENTS:-}"
LR="${LR:-8e-4}"; EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
export HF_HOME="$CACHE_DIR" HF_DATASETS_CACHE="$CACHE_DIR/datasets" TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    --rec-batch-size "$REC_BATCH_SIZE" --num-workers "$NUM_WORKERS" --seed "$SEED"
    --lr "$LR" --early-stopping-patience "$EARLY_STOPPING_PATIENCE")
  [[ -z "$MAX_EVENTS" ]] || args+=(--max-events "$MAX_EVENTS")
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}" 2>&1 |
    "$PYTHON_BIN" -u "$PROJECT_ROOT/filter_progress.py" "$log_path"
}

if [[ "$SUBSETS" != all ]]; then
  IFS=, read -r -a chosen <<< "$SUBSETS"
  [[ ${#chosen[@]} -gt 0 ]] || { echo "SUBSETS cannot be empty" >&2; exit 2; }
  for subset in "${chosen[@]}"; do
    case "$subset" in
      Full_Beauty|Beauty_and_Personal_Care|Baby_Products|Sports_and_Outdoors|Books|Toys_and_Games|Video_Games|Clothing_Shoes_and_Jewelry) ;;
      *) echo "Unknown subset: $subset" >&2; exit 2 ;;
    esac
  done
fi
mkdir -p "$OUTPUT_ROOT"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  run_subset "Full_Beauty" "amazon-all-beauty" "$repeat"
  # run_subset "Beauty_and_Personal_Care" "amazon:Beauty_and_Personal_Care" "$repeat"
  run_subset "Baby_Products" "amazon:Baby_Products" "$repeat"
  run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" "$repeat"
  # run_subset "Books" "amazon-books" "$repeat"
  run_subset "Toys_and_Games" "amazon-toys-and-games" "$repeat"
  # run_subset "Video_Games" "amazon-video-games" "$repeat"
  # run_subset "Clothing_Shoes_and_Jewelry" "amazon-clothing-shoes-and-jewelry" "$repeat"
done
