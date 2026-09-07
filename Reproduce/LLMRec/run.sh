#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIATEK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$MEDIATEK_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/P78123011/miniforge3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
GPU_IDS="${GPU_IDS:-0}"
SUBSETS="${SUBSETS:-all}"
REPEATS="${REPEATS:-1}"
EPOCHS="${EPOCHS:-15}"; BATCH_SIZE="${BATCH_SIZE:-1024}"
EMBED_SIZE="${EMBED_SIZE:-64}"; FEATURE_DIM="${FEATURE_DIM:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"; EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-7}"
SEED="${SEED:-2022}"; MAX_EVENTS="${MAX_EVENTS:-}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -ge 1 ]] || { echo "At least one GPU is required" >&2; exit 2; }

should_run() { [[ "$SUBSETS" == all || ",$SUBSETS," == *",$1,"* ]]; }
run_subset() {
  local subset="$1" dataset="$2" repeat="$3" gpu="$4" timestamp run_dir
  should_run "$subset" || return 0
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset/llmrec_${timestamp}_r${repeat}"
  echo "Running LLMRec subset=$subset dataset=$dataset GPU=$gpu output=$run_dir"
  args=(--dataset "$dataset" --cache-dir "$CACHE_DIR" --output-dir "$run_dir"
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --embed-size "$EMBED_SIZE"
    --feature-dim "$FEATURE_DIM" --learning-rate "$LEARNING_RATE"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" --seed "$SEED" --device cuda)
  [[ -z "$MAX_EVENTS" ]] || args+=(--max-events "$MAX_EVENTS")
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${args[@]}"
}

mkdir -p "$OUTPUT_ROOT"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  jobs=0
  for spec in \
    "Full_Beauty|amazon-all-beauty" \
    "Baby_Products|amazon:Baby_Products" \
    "Sports_and_Outdoors|amazon-sports-and-outdoors" \
    "Toys_and_Games|amazon-toys-and-games"; do
    IFS='|' read -r subset dataset <<< "$spec"
    should_run "$subset" || continue
    gpu="${GPU_ARRAY[$((jobs % ${#GPU_ARRAY[@]}))]}"
    run_subset "$subset" "$dataset" "$repeat" "$gpu" &
    ((jobs+=1))
    if (( jobs % ${#GPU_ARRAY[@]} == 0 )); then wait; fi
  done
  wait
done
