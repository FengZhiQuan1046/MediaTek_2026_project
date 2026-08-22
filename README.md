# Mamba + Bipartite-Graph Recommender Demo

This is a small, reproducible recommendation-system baseline for the
`McAuley-Lab/Amazon-Reviews-2023` dataset.  It combines three signals:

1. a LightGCN-style user--item bipartite graph,
2. product-text vectors produced by `state-spaces/mamba-2.8b`, and
3. a short interaction-history encoder.

The final score is a learned fusion of graph and Mamba/text-history scores.
The 2.8B model is loaded frozen by default; training it in full is not a
practical demo workload.  The trainable recommender head stays compact.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`McAuley-Lab/Amazon-Reviews-2023` currently uses a dataset loading script.
The project therefore pins `datasets==3.6.0`; do not upgrade `datasets` to 4.x,
which rejects these scripts.

The Mamba tokenizer also requires `protobuf`, which is installed through the
project requirements.

For a first smoke test (no network or GPU required):

```powershell
bash run.sh
```

## Amazon Reviews 2023 run

The command below loads the All_Beauty review configuration and its matching
item metadata configuration from Hugging Face, using the configured cache
directory. Change `--subset` to any available raw review
configuration (for example `raw_review_Movies_and_TV`).

```powershell
bash run.sh
```

`run.sh` is configured with cache directory `/home/P78123011/cache` and random
seed `25252`. It passes the cache directory explicitly to both the dataset and
Mamba loaders. The script loads `raw_review_All_Beauty` and joins it with
`raw_meta_All_Beauty` using `parent_asin`; the metadata title, features, and
description become the item text encoded by Mamba. Every run writes a timestamped log under `log/`, a loss
curve PNG under `fig_outputs/`, and up to five held-out recommendation examples
as JSON under `json_outputs/`.

By default, Mamba item vectors are cached under the `--cache-dir` directory.
Use `--skip-mamba` during CPU-only graph development; this deliberately replaces
the Mamba vectors with a trainable item vector, so it is not the target model.

## Evaluation

The data split is chronological per user: each user's newest interaction is the
test target, and the preceding one is validation when possible.  The script
reports Recall@10 and NDCG@10 using sampled negatives.  This is an offline
implicit-feedback evaluation, not a claim of production performance.

## Layout

- `src/data.py`: Amazon loader, input schema normalisation, chronological split.
- `src/model.py`: graph propagation, Mamba text encoder, and fused ranker.
- `src/train.py`: end-to-end command-line training and testing.
