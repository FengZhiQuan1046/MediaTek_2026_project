#!/usr/bin/env bash
# RQ5: uncertainty-stratified preference-transition evaluation; fixed SW4/P32; other training defaults match run_4090.sh.
# Usage: bash run_rq5.sh [RQ5_PYTHON_OPTIONS...]
# Preview: DRY_RUN=1 bash run_rq5.sh
# One subset: SUBSETS=Full_Beauty GPU_IDS=0 bash run_rq5.sh
# Outputs: one rq5.json + rq5_summary.json per subset, without model checkpoints.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python; fi
GPU_IDS="${GPU_IDS:-1}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+$ ]]; then
  echo "GPU_IDS must select one physical GPU" >&2; exit 2
fi
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2; exit 2
fi
read -r -a selected <<< "${SUBSETS:-Sports_and_Outdoors Toys_and_Games}"
for subset in "${selected[@]}"; do
  case "$subset" in
    Full_Beauty|Baby_Products|Sports_and_Outdoors|Toys_and_Games) ;;
    *) echo "Unknown subset: $subset" >&2; exit 2 ;;
  esac
done
# Full_Beauty Baby_Products 
RUN_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs_mamba_rl/rq5}/rq5_$(date '+%Y%m%d_%H%M%S')_$$"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# Keep imported source trees read-only, including avoiding new __pycache__ files.
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ribs-rq5-mpl-$$"
windows=(4)
preferences=(32)
MAMBA_MODEL_ID="${MAMBA_MODEL_ID:-state-spaces/mamba-${MAMBA_MODEL_SIZE:-130m}-hf}"
cd "$PROJECT_ROOT"
for preference in "${preferences[@]}"; do
for window in "${windows[@]}"; do
for subset in "${selected[@]}"; do
  command=("$PYTHON_BIN" -B -m src.rq5
    --subset "$subset" --output-dir "$RUN_ROOT/SW${window}_P${preference}/$subset"
    --cache-dir "${CACHE_DIR:-$WORKSPACE_ROOT/cache}" --device cuda:0
    --epochs "${SPECIALISTS_EPOCH:-4}"
    --short-window "$window" --preference-count "$preference" --mamba-model-id "$MAMBA_MODEL_ID"
    --joint-epochs "${JOINT_EPOCH:-0}"
    --specialist-lr "${SPECIALISTS_LR:-1e-4}" --joint-lr "${JOINT_LR:-5e-5}"
    --rl-coef "${RL_COEF:-0.1}" --rl-topk "${RL_TOPK:-10}"
    --batch-size "${BATCH_SIZE:-128}" --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
    --encode-batch-size "${MAMBA_ENCODE_BATCH_SIZE:-32}"
    --shuffle-repeats "${SHUFFLE_REPEATS:-5}" --bootstrap "${BOOTSTRAP_REPEATS:-1000}"
    --seed "${SEED:-25252}" "$@")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done
done
done
