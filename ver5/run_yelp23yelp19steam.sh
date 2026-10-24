#!/usr/bin/env bash
# Train independently and evaluate Yelp23, Yelp19, then Steam.
# Usage:
# YELP23_DATA_PATH=/data/yelp23 YELP19_DATA_PATH=/data/yelp19 \
# STEAM_DATA_PATH=/data/steam/reviews.json.gz bash run_yelp23yelp19steam.sh
# Optional: GPU_IDS=0 PYTHON_BIN=/path/to/python DRY_RUN=1
# Additional trainer arguments are forwarded to each run (e.g. --skip-mamba).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  sed -n '2,7p' "$PROJECT_ROOT/run_yelp23yelp19steam.sh"
  exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Set PYTHON_BIN to a Python environment with ver5 dependencies." >&2
  exit 2
fi
GPU_IDS="${GPU_IDS:-0}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo 'GPU_IDS must contain one or two GPU indices, e.g. 0 or 0,1' >&2
  exit 2
fi
paths=("${YELP23_DATA_PATH:-}" "${YELP19_DATA_PATH:-}" "${STEAM_DATA_PATH:-}")
names=(yelp23 yelp19 steam)
for i in "${!names[@]}"; do
  if [[ -z "${paths[$i]}" || ! -e "${paths[$i]}" ]]; then
    echo "Set ${names[$i]^^}_DATA_PATH to an existing local dataset." >&2
    exit 2
  fi
  paths[$i]="$(realpath "${paths[$i]}")"
done
if [[ ! -f "${paths[2]}" ]]; then
  echo 'STEAM_DATA_PATH must be a CSV/JSONL file, optionally .gz.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
GRAPH_DEVICE=cuda:0
if [[ "$GPU_IDS" == *,* ]]; then GRAPH_DEVICE=cuda:1; fi
# A unique suite directory prevents overwriting previous scores or data.
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs_mamba_rl/yelp23_yelp19_steam}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  SUITE_DIR="$OUTPUT_ROOT/dry_run"
else
  mkdir -p "$OUTPUT_ROOT"
  SUITE_DIR="$(mktemp -d "$OUTPUT_ROOT/run_$(date '+%Y%m%d_%H%M%S')_XXXXXX")"
fi
cd "$PROJECT_ROOT"
for i in "${!names[@]}"; do
  dataset="${names[$i]}"
  run_dir="$SUITE_DIR/$dataset"
  args=(
    -m src.train_yelp_steam
    --dataset "$dataset" --data-path "${paths[$i]}" --cache-dir "$CACHE_DIR"
    --device cuda:0 --graph-device "$GRAPH_DEVICE"
    --specialist-epochs "${SPECIALIST_EPOCHS:-3}"
    --joint-epochs "${JOINT_EPOCHS:-7}" --rl-coef "${RL_COEF:-0.1}"
    --batch-size "${BATCH_SIZE:-128}" --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
    --max-transitions "${MAX_TRANSITIONS:-500000}" --candidates "${CANDIDATES:-64}"
    --short-window "${SHORT_WINDOW:-auto}" --max-history "${MAX_HISTORY:-0}"
    --preference-count "${PREFERENCE_COUNT:-32}"
    --mamba-model-id "${MAMBA_MODEL_ID:-state-spaces/mamba-130m-hf}"
    --mamba-encode-batch-size "${MAMBA_ENCODE_BATCH_SIZE:-32}"
    --mamba-max-tokens "${MAMBA_MAX_TOKENS:-32}"
    --validate-every-steps "${VALIDATE_EVERY_STEPS:-1000}"
    --validation-user-limit 0 --periodic-test-user-limit 0
    --seed "${SEED:-25252}" --no-generate-reasons --no-save-model-weights
    --output-run-dir "$run_dir" --score-file "$run_dir/${dataset}_scores.json"
  )
  if [[ -n "${MAX_EVENTS:-}" ]]; then args+=(--max-events "$MAX_EVENTS"); fi
  args+=("$@")
  echo "=== $dataset: $run_dir ==="
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$PYTHON_BIN" "${args[@]}"
    printf '\n'
  else
    mkdir -p "$run_dir"
    "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$run_dir/console.log"
  fi
done
echo "Scores: $SUITE_DIR/{yelp23,yelp19,steam}/*_scores.json"
