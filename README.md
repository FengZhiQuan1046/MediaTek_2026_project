# Mamba + Bipartite-Graph Recommender Demo

This is a small, reproducible recommendation-system baseline for the
`McAuley-Lab/Amazon-Reviews-2023` dataset. It combines three signals:

1. a LightGCN-style user--item bipartite graph,
2. product-text vectors produced by `state-spaces/mamba-2.8b-hf`, and
3. a short interaction-history encoder.

The final score is a learned fusion of graph and Mamba/text-history scores.
`state-spaces/mamba-2.8b-hf` is the official Transformers-compatible companion
to `state-spaces/mamba-2.8b`; it retains the checkpoint and provides the
required tokenizer/config files. The 2.8B model is loaded frozen by default; training it in full is not a
practical demo workload.  The trainable recommender head stays compact.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the configured NVIDIA L40S environment, the requirements install
`torch==2.7.1+cu128`. This CUDA 12.8 wheel is compatible with a driver reporting
CUDA API 12.9. If a different Torch build is already installed, replace it with:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
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

## GPU selection and multi-GPU training

```bash
bash run.sh          # single GPU: CUDA 0
bash run.sh 3        # single GPU: CUDA 3
bash run.sh 0,1      # two GPUs with DistributedDataParallel
bash run.sh 0,2,3    # three GPUs with DistributedDataParallel
```

For multi-GPU runs, `run.sh` sets `CUDA_VISIBLE_DEVICES`, then launches one
NCCL process per selected GPU with PyTorch `torchrun`. Only rank 0 loads Mamba
to create item vectors and writes logs, figures, and JSON outputs.

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
