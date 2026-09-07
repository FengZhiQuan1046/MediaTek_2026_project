# Mamba4Rec reproduction for MediaTek 2026

This is a standalone adaptation of the official Mamba4Rec architecture. It
uses `ver4/src/data_mamba_rl.py` directly, so Amazon loading, one-pass 5-core
filtering, ID mapping, chronological leave-two-out splitting, cache identity,
full-catalog evaluation, seen-item masking, and tie handling match ver4.

Run the requested four datasets:

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce/Mamba4Rec
PYTHON_BIN=/home/P78123011/miniforge3/envs/py31014/bin/python bash install_requirements.sh
bash run.sh 0
```

`USE_LARGE_MAMBA=True` is the default and loads the same pretrained
`state-spaces/mamba-2.8b-hf` checkpoint used by ver4. It uses batch size 1, evaluation batch size
1, bfloat16, and learning rate 1e-5 by default. To reproduce the authors'
original small Mamba4Rec instead:

```bash
USE_LARGE_MAMBA=False bash run.sh 0
```

The checkpoint and Hugging Face caches use the same workspace-level
`CACHE_DIR` as ver4 (`/workspace/P78123011/cache` by default). Both modes use
the same item-level CE objective, ver4 protocol, and outputs layout. The large
mode does not save a local model checkpoint in the experiment output folder.

For the default 2.8B mode, `run.sh` defaults to `GPU_IDS=0,1` and
`LARGE_DEVICE_MAP=balanced`. Accelerate partitions the backbone layers across
both visible GPUs; this is model parallelism, not DDP, so the model is not
duplicated on both cards. Override the physical cards when needed:

```bash
GPU_IDS=0,1 USE_LARGE_MAMBA=True bash run.sh
```

The shared environment uses a newer PyTorch/CUDA ABI than the CUDA extension
versions in the original 2024 environment. The reproduction therefore uses
the compatible pure-PyTorch `mambapy` backend by default. If a compatible
`mamba_ssm` is already installed, it is detected and used automatically.

Use `SUBSETS=Full_Beauty,Toys_and_Games`, `REPEATS=5`, or the hyperparameter
environment variables in `run.sh` to override defaults. Each run writes
`config.json`, `metrics.json`, and `train_{time}.log` beneath
`outputs/{subset}/mamba4rec_{time}_r{repeat}/`, matching ReSID's layout.

The model retains the upstream CE objective and Mamba block defaults:
hidden size 64, one layer, `d_state=32`, `d_conv=4`, expansion 2, and dropout
0.2. Real catalog IDs remain zero-based at the ver4 boundary and are shifted
only inside padded model inputs.
