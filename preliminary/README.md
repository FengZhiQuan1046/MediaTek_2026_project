# ver4 preliminary experiment

Run from this directory with:

```bash
GPU_IDS=1 bash run.sh
```

`GPU_IDS` accepts one physical GPU (for example `0`) or two GPUs (for example
`0,1`; the first runs the single Mamba branch, LoRA, and training batches, while the
second runs LightGCN and its graph tensors). The launcher masks CUDA before any
Python process starts and verifies that PyTorch sees only those devices. With
one GPU, every CUDA component uses that GPU. Raw dataset objects and disk caches
remain on CPU/storage by design. `DATASETS` can select a comma-separated subset list. Analysis and
training statistics are written below `preliminary/outputs/`:

- `outputs/training/`: cumulative per-subset training manifest and ver4
  score/log artifacts.
- `outputs/analysis/<subset>/run_YYYYMMDD_HHMMSS/`: that subset's CSV
  statistics, diagnostics, report, manifest, and complete set of PDF figures.

Subsets run as independent pipelines rather than one combined job. For each
selected subset, `run.sh` completes training, immediately calculates its
statistics, and writes Figure 1–4 before starting the next subset. The default
seven subsets therefore produce seven separate figure sets; curves from different
subsets are no longer combined in one PDF.

Every `bash run.sh` invocation first retrains a single full-history
Mamba+LoRA+LightGCN model for every selected subset, even when its exact schema-v2 data
cache already exists. Existing exact caches only avoid repeating data download
and preprocessing; they never skip model training. Only the full-history
Selective-Mamba branch is active (`Long=1, Short=0, Preference=0, Graph=1`).
There is no coordinator and no short or preference agent; LightGCN is fused
with the one sequence branch. Training uses one complete validation and one
complete test at the end of every epoch, followed by ver4's final best-model
evaluation block. `--no-save-model-weights` remains enabled, so logs, scores,
metrics, data caches, and statistical artifacts are retained without writing a
model checkpoint or weight file.

The three additional default Amazon Reviews 2023 subsets are `Musical_Instruments`,
`Video_Games`, and `Software`. The [official dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
reports 3.0M, 4.6M, and 4.9M raw ratings respectively, versus 6.0M for
the existing `Baby_Products` subset. They use at most 500,000 training
transitions and 64 evaluation candidates each to keep the single-GPU
training settings conservative. The loader still downloads each entire
raw category and its metadata on first use; these limits do not cap
preprocessing or full validation/test time. A complete run of these new
categories has not yet been measured on this machine, so GPU memory use
is an estimate based on dataset size and settings rather than a verified peak.
To run one added subset at a time, set e.g.
`GPU_IDS=1 DATASETS=amazon:Musical_Instruments bash run.sh`.

The LoRA rank and epoch count are grouped near the top of `run.sh` as
`LORA_RANK` and `TRAIN_EPOCHS`. Their defaults can also be overridden with the
corresponding environment variables. Only the joint-ranking stage is used;
specialist and coordinator stages are set to zero.

The default 2,000 users per subset is a pilot. Set `MAX_USERS=0` for all eligible
users. For example:

```bash
GPU_IDS=0 DATASETS=amazon-sports-and-outdoors MAX_USERS=0 bash run.sh
```

## Statistics

The semantic representation is a deterministic 1,024-dimensional word/bigram
hashing vector of ver4's `item_texts`. Popularity matching, cosine
normalization, and K-means preference-proxy fitting use training items only.
The cache does not retain timestamps, so distance is the number of interactions,
not elapsed time.

Figures 1, 2, and 4 are model-free dataset diagnostics. Figure 3 is different:
it is computed from the restored best single Mamba+LoRA+LightGCN model before
the training process releases it. No checkpoint or model weights are saved.

For every test user and every history position from 1 to `MAX_LAG`, Figure 3
scores the same true next item and 64 fixed sampled unseen negatives twice:
once with the complete history and once after deleting exactly that history
item. Its statistic is
`margin(full history) - margin(history without item)`, where margin is the true
item score minus the mean negative score. Positive values mean that keeping the
item helped; negative values mean it hurt. The PDF plots the mean and a 95%
user-bootstrap interval at every exact position. Per-user rows are saved as
`model_item_influence_per_position.csv`, and the plotted values are saved as
`model_item_influence_summary.csv`.

Cosine values are normalized separately for every Amazon subset, using random
distinct usable item pairs sampled uniformly from that subset's complete item
catalog rather than from a capped reference-item pool. Let `mu_D` and `sigma_D`
be their mean and standard deviation. Then:

```text
cosine_z_D(a,b) = (cosine(a,b) - mu_D) / sigma_D
excess_cosine_z(u,k) = cosine_z_D(history[-k], y)
                       - mean_B cosine_z_D(matched_random, y)
```

The sampled pair count, `mu_D`, and `sigma_D` are saved in each subset's
`diagnostics.json`; both raw and normalized cosine columns are preserved in
`alignment_per_position.csv`. Random controls are sampled from the same subset
and training-popularity quintile. Figure 1 and the primary paired contrast use
`excess_cosine_z`. All interaction-lag plots use a linear x-axis.

Figure 4 turns the positive part of this score into an easy-to-read cumulative
percentage. For each user, every history item contributes
`max(excess_cosine_z, 0)`. The value at X is the sum from the most recent X
items divided by the sum across that user's full observed history (capped by
`MAX_LAG`). Users with no positive signal are excluded because a percentage is
undefined for them. The line is the mean across users and the shaded region is
the 25th–75th percentile range. The underlying values are saved in
`information_coverage.csv`; `information_coverage_per_user.csv` also records
how many recent items each user needs to reach 50%, 80%, and 90% of the signal.

Curves include an all-available cohort and fixed cohorts with histories of at
least K=8,16,32. `paired_decay.csv` compares lag 1 against the within-user mean
of lags 9–16 among the same history>=16 users. Intervals are user-bootstrap 95%
CIs; a CI crossing zero is not evidence of equivalence or a proven noise cutoff.

The second statistic is a training-only coarse preference proxy:

```text
excess_cluster_match(u,k) = 1[cluster(history[-k]) = cluster(y)]
                            - mean_B 1[cluster(random) = cluster(y)]
```

Old-history item affinity and preference-cluster readouts evaluate the same
held-out target and popularity-matched negatives. They are diagnostic encodings,
not the learned ver4 preference agent. The report therefore does not claim that
Mamba cannot perform multiple functions, and unfavorable results are retained.
