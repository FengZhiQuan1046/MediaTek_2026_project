# ReSID reproduction for MediaTek 2026

This directory adapts the official ReSID pipeline to the project's common data
and evaluation boundary. It implements the three ReSID steps:

1. field-aware masked auto-encoding (FAMAE),
2. globally aligned orthogonal quantization (GAOQ), and
3. autoregressive semantic-ID next-item recommendation.

The original implementation used its own Spark-produced Parquet splits. This
adaptation deliberately does **not** use those splits: `data_adapter.py` calls
`ver4/src/data_mamba_rl.py::load_recommendation_data` directly and uses the same
cache identity as every other reproduction.

## Exact test protocol

- Amazon source/config aliases are identical to `ver4/run_amazons_full_rl.sh`.
- The project always applies its one-pass raw user/item threshold
  (`MIN_INTERACTIONS = 5`), followed by chronological leave-two-out.
- Validation input is `train`; its target is `valid_target`.
- Test input is exactly `train + valid_target`; its target is `test_target`.
- Every eligible user and every catalog item are evaluated.
- Previously seen items are masked, except when the seen item is the target.
- Ranks use the same conservative tie rule: `score >= target_score`.
- Recall/Hit/NDCG at 5 and 10 use the common project formulas.

## Run

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce/ReSID
bash run.sh
```

By default, all eight Amazon subsets from `ver4/run_amazons_full_rl.sh` run with
8 FAMAE epochs and 8 recommender epochs. Set `GPU_IDS="0"` near the top of
`run.sh` manually. Common overrides:

```bash
SUBSETS=Full_Beauty,Video_Games FAMAE_EPOCHS=8 EPOCHS=8 bash run.sh
REPEATS=5 bash run.sh
```

For a fast local end-to-end check without downloads:

```bash
python train.py \
  --dataset synthetic --cache-dir /tmp/resid-cache --output-dir /tmp/resid-smoke \
  --device cpu --no-amp --famae-epochs 1 --epochs 1 \
  --max-batches-per-epoch 1 --hidden-size 16 --num-heads 2 \
  --codebook1-size 4 --codebook2-size 4 --text-buckets 32
```

Each run directory contains only
`config.json`, `metrics.json`, and `train_{time}.log` by default. They are saved
under `outputs/{subset_name}/resid_{time}/`.

The ReSID method is based on the upstream project at
<https://github.com/yu-liang/ReSID> and paper arXiv:2602.02338. The adaptation
uses only metadata text already normalized by ver4; deterministic hashed token
fields provide FAMAE's categorical side-information without introducing a
second Amazon preprocessing or split path.
