# LLaRA reproduction with the ver4 data and evaluation protocol

This directory contains the LLaRA model interface, LoRA projector, and SASRec
implementation. It uses the installed `py31014` conda environment and does not
need a separate upstream LLaRA checkout or a new virtual environment.

## Data and evaluation

`data_adapter.py` calls ver4's own `load_recommendation_data_cached` and
`build_transitions`. The same 5-core filtering, `min_rating=4.0`, chronological
split, cache identity, seed (`25252`), per-subset transition limit, and training
prefix selection apply. Validation and test use the same eligible users,
sorted IDs, 100-event history, and test history that includes the validation
item. `EVAL_USER_LIMIT=0` evaluates all eligible users. A positive limit uses
ver4's evenly spaced pilot-user selection and is labeled as a pilot.

LLaRA now uses a history-only prompt. Each item in the **entire catalog** is
scored by the mean token log probability of its name continuation. It then
applies the same ver4 popularity and transition priors, masks previously seen
items except the target, and counts ties with `scores >= target_score`.
Recall, Hit, NDCG, and the filtered-full-sequence short/long breakdown use
ver4's formulas. The best LoRA/projector state is chosen by validation
`NDCG@10`, restored, and evaluated on validation and test. Each run records
user counts and protocol details in `metrics.json`.

The model and scoring function remain LLaRA-specific. The priors are applied
with the numerical coefficients used by `ver4/run_long_short.sh` for each
subset; those coefficients can be overridden through `POPULARITY_ALPHA` and
`TRANSITION_BETA`. This provides the same evaluation postprocessing, although
the priors can have a different effect because LLaRA's raw scores have a
different scale from ver4's model scores.

Full-catalog generative evaluation is computationally expensive: Full Beauty
has 4,090 products and 1,457 eligible test users. Every formal validation
check and final test scores all products for all selected users. `EVAL_START`,
`EVAL_PROGRESS` (with ETA), and `EVAL_DONE` are written to the terminal and
`train_*.log` every roughly 30 seconds during ranking. The scorer reuses the
prompt KV cache and groups items by token length, but one 7B full-catalog
validation can still take many hours. Set
`EVAL_USER_LIMIT` to a positive value for a pilot, then set it back to `0`
for a comparable formal result. Old sampled-candidate `metrics.json` files
remain unchanged and are not comparable to new full-catalog results.

## Run

```bash
SUBSETS=Full_Beauty GPU_IDS=0 \
  bash /workspace/P78123011/MediaTek_2026_project/Reproduce/LLaRA/run.sh
```

The script finds Llama-2-7B-hf in the shared Hugging Face cache or accepts
`LLM_PATH`. It uses `/dataspace/<user>/miniconda3/envs/py31014/bin/python` by
default; set `PYTHON_BIN` to another existing Python executable if needed.
`CACHE_DIR` defaults to `/workspace/P78123011/cache`, and outputs go to
`Reproduce/LLaRA/outputs/<subset>/llara_<timestamp>_r<repeat>`.

Key overrides: `GPU_IDS`, `EPOCHS`, `BATCH_SIZE`, `ACCUMULATE_GRAD_BATCHES`,
`MAXLEN` (default 100), `MAX_TRAIN_SAMPLES` (default matches ver4 per subset),
`EVAL_USER_LIMIT` (default 0), `CATALOG_CHUNK_SIZE` (default 8), `SEED`
(default 25252), `POPULARITY_ALPHA`, `TRANSITION_BETA`, `REC_EPOCHS`,
`REC_BATCH_SIZE`, `NUM_WORKERS`, `CACHE_DIR`, `OUTPUT_ROOT`, and `LLM_PATH`.
The script currently runs Full Beauty, Baby Products, Sports and Outdoors, and
Toys and Games in sequence when `SUBSETS=all`; additional subset commands are
present but commented out. `REPEATS` controls repeated runs.

To check the code without loading the 7B model:

```bash
bash -n run.sh
PYTHONDONTWRITEBYTECODE=1 \
  /dataspace/$(id -un)/miniconda3/envs/py31014/bin/python \
  -m unittest discover -s tests -v
```

The 100-event history is kept for every user. Product display text in the
prompt is shortened to 64 characters so Full Beauty histories fit Llama-2's
context; the loaded events and item IDs remain unchanged. The frozen 7B
weights are read from safetensors, and only the best trainable LoRA and
projector weights are saved to `best_adapter.pt`.
