#!/usr/bin/env bash
# One model per tmux; four datasets, GPU 0 only, <=8 total epochs per dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
MODEL="${1:?Specify LLM-SRec or LLMRec}"
GPU="${2:-0}"
RUN_ID="${3:-llm_yelp_$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S)}"
case "$MODEL" in LLM-SRec|LLMRec) ;; *) exit 2;; esac
[[ "$GPU" == 0 && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$WORKSPACE/cache" HF_HUB_CACHE="$WORKSPACE/cache"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON_BIN:-$WORKSPACE/miniconda3/envs/py31014/bin/python}"
failed=0
for dataset in ${DATASET_ORDER:-yelp19 yelp23 movielens-1m steam}; do
    out="$ROOT/$MODEL/outputs/$dataset/$RUN_ID"
    [[ ! -e "$out" ]] || { echo "Output exists: $out"; exit 2; }
    mkdir -p "$out"
    printf 'status=preparing\nphysical_gpu=%s\nstarted=%s\n' "$GPU" "$(date -Is)" > "$out/status.txt"
    set +e
    "$PYTHON" -u "$ROOT/prepare_hf_four.py" --cache-dir "$WORKSPACE/cache" --datasets "$dataset" > "$out/prepare.log" 2>&1
    result=$?
    if [[ "$result" == 0 ]]; then
        case "$dataset" in
            yelp19|yelp23) data_path="$WORKSPACE/cache/normalized/${dataset}_unique.jsonl" ;;
            movielens-1m) data_path="$WORKSPACE/cache/hf_datasets/movielens-1m" ;;
            steam) data_path="$WORKSPACE/cache/normalized/steam.csv" ;;
            *) echo "Unsupported dataset: $dataset"; exit 2 ;;
        esac
        args=(--dataset "$dataset" --data-path "$data_path"
              --cache-dir "$WORKSPACE/cache" --output-dir "$out" --device cuda:0 --seed 25252)
        if [[ "$MODEL" == LLM-SRec ]]; then
            args+=(--epochs "${EPOCHS:-6}" --teacher-epochs "${TEACHER_EPOCHS:-2}"
                   --batch-size "${BATCH_SIZE:-8}" --eval-batch-size "${EVAL_BATCH_SIZE:-16}")
        else
            args+=(--epochs "${EPOCHS:-8}" --batch-size "${BATCH_SIZE:-1024}" --early-stopping-patience 10)
        fi
        printf 'status=running\n' >> "$out/status.txt"
        "$PYTHON" -u "$ROOT/$MODEL/train.py" "${args[@]}" 2>&1 | tee "$out/console.log"
        result=${PIPESTATUS[0]}
    fi
    set -e
    if [[ "$result" == 0 ]]; then status=completed; else status=failed;failed=1; fi
    printf 'status=%s\nexit_code=%s\nfinished=%s\n' "$status" "$result" "$(date -Is)" >> "$out/status.txt"
done
exit "$failed"
