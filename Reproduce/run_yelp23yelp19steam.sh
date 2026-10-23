#!/usr/bin/env bash
# Usage: bash run_yelp23yelp19steam.sh Mamba4Rec 0 [RUN_ID]
#        bash run_yelp23yelp19steam.sh ReSID 1 [RUN_ID]
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
MODEL="${1:?Specify Mamba4Rec or ReSID}"
GPU_ID="${2:?Specify GPU index}"
RUN_ID="${3:-hf_$(TZ=Asia/Taipei date '+%Y%m%d_%H%M%S')}"
case "$MODEL" in Mamba4Rec|ReSID|SASRec|BERT4Rec|SIGMA) ;; *) exit 2 ;; esac
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
CACHE_DIR="$WORKSPACE_ROOT/cache"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/miniconda3/envs/py31014/bin/python}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$CACHE_DIR" HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets" TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export TORCH_HOME="$CACHE_DIR/torch" XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TMPDIR="$CACHE_DIR/tmp" MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$REPRO_ROOT/$MODEL"
failed=0
for dataset in ${DATASET_ORDER:-steam yelp23 yelp19}; do
  case "$dataset" in
    steam) data_path="$CACHE_DIR/normalized/steam.csv" ;;
    *) data_path="$CACHE_DIR/normalized/${dataset}_unique.jsonl" ;;
  esac
  run_dir="$REPRO_ROOT/$MODEL/outputs/$dataset/$RUN_ID"
  console_dir="$CACHE_DIR/training_runs/$RUN_ID/$MODEL/$dataset"
  [[ ! -e "$run_dir" ]] || { echo "Output exists: $run_dir" >&2; exit 2; }
  mkdir -p "$run_dir" "$console_dir"
  args=(--reproduction "$MODEL" --dataset "$dataset" --data-path "$data_path"
    --cache-dir "$CACHE_DIR" --output-dir "$run_dir" --device cuda:0
    --seed 25252 --maxlen 100 --batch-size "${BATCH_SIZE:-128}"
    --eval-batch-size "${EVAL_BATCH_SIZE:-64}" --amp)
  if [[ "$MODEL" == Mamba4Rec ]]; then
    args+=(--no-use-large-mamba --epochs "${EPOCHS:-15}" --learning-rate 1e-3
      --hidden-size 64 --num-layers 1 --early-stopping-patience 10)
  elif [[ "$MODEL" == ReSID ]]; then
    args+=(--famae-epochs "${FAMAE_EPOCHS:-8}" --epochs "${EPOCHS:-25}"
      --learning-rate 1e-3 --hidden-size 128 --num-heads 4
      --early-stopping-patience 8)
  elif [[ "$MODEL" == SIGMA ]]; then
    args+=(--no-use-large-mamba --epochs "${EPOCHS:-200}" --learning-rate 1e-3
      --hidden-size 64 --num-layers 1 --early-stopping-patience 10
      --maxlen "${SIGMA_MAXLEN:-0}")
  elif [[ "$MODEL" == SASRec ]]; then
    args+=(--epochs "${EPOCHS:-7}" --learning-rate 1e-3 --hidden-units 128
      --num-blocks 2 --num-heads 1 --dropout-rate 0.2 --early-stopping-patience 20)
  else
    args+=(--epochs "${EPOCHS:-50}" --learning-rate 1e-3 --hidden-units 128
      --num-blocks 2 --num-heads 2 --dropout-rate 0.2 --mask-prob 0.2
      --loss-chunk-size 128 --early-stopping-patience 20)
  fi
  printf 'status=running\nphysical_gpu=%s\nstarted=%s\n' "$GPU_ID" "$(date -Is)" > "$run_dir/status.txt"
  set +e
  "$PYTHON_BIN" -u "$REPRO_ROOT/train_shared_dataset.py" "${args[@]}" 2>&1 | tee "$console_dir/console.log"
  result=${PIPESTATUS[0]}
  set -e
  printf 'exit_code=%s\nfinished=%s\n' "$result" "$(date -Is)" >> "$run_dir/status.txt"
  if [[ "$result" != 0 ]]; then failed=1; fi
 done
exit "$failed"
