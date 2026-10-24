#!/usr/bin/env bash
# Two sequential datasets per GPU, with the dataset-specific temporal-group windows.
# Usage: bash run_group_window_queue.sh GPU RUN_LABEL
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
GPU="${1:?GPU required}"
RUN="${2:?run label required}"
case "$GPU" in
  0) DATASETS=(steam movielens-1m); WINDOWS=(5 79)
     PATHS=("$WORKSPACE/cache/normalized/steam.csv" "$WORKSPACE/cache/raw/dated_analysis/movielens-1m") ;;
  1) DATASETS=(yelp23 yelp19); WINDOWS=(6 7)
     PATHS=("$WORKSPACE/cache/normalized/yelp23_unique.jsonl" "$WORKSPACE/cache/normalized/yelp19_unique.jsonl") ;;
  *) echo 'Expected GPU 0 or 1' >&2; exit 2 ;;
esac
RUN_ROOT="$WORKSPACE/cache/training_runs/$RUN"
mkdir -p "$RUN_ROOT"
export MAX_HISTORY=0
export SPECIALIST_EPOCHS=4 JOINT_EPOCHS=0 RL_COEF=0
for preference_count in 32; do
for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  echo "START dataset=$dataset physical_gpu=$GPU short_window=${WINDOWS[$i]}"
  PREFERENCE_COUNT="$preference_count" SHORT_WINDOW="${WINDOWS[$i]}" bash "$HERE/run_hf_dataset_scores.sh" "$GPU" "$dataset" "${PATHS[$i]}" "${RUN_ROOT}_P${preference_count}"
  result=$?
  echo "FINISH dataset=$dataset exit=$result"
  if [[ "$result" != 0 ]]; then
    echo "Dataset failed; inspect $RUN_ROOT/$dataset/console.log" >&2
  fi
done
done
