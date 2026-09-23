# EAGER reproduction

This wrapper runs the upstream [`yewzz/EAGER`](https://github.com/yewzz/EAGER) implementation on the exact
ver4 Amazon-2023 preprocessing and chronological leave-two-out split.

The upstream source must also be present; `lib` is supplied by that repository.
From this wrapper directory, set up the default workspace layout with:

```bash
git clone https://github.com/yewzz/EAGER.git ../../../EAGER
pip install -r requirements.txt
bash run.sh
```

If upstream is checked out elsewhere, set `EAGER_UPSTREAM_DIR` to its inner
`EAGER` directory containing `lib/` and `optimizers/`, for example
`EAGER_UPSTREAM_DIR=/path/to/EAGER/EAGER bash run.sh`.

`GPU_IDS=0,1` runs two subsets concurrently, one per GPU. With
`DISTRIBUTED=1`, one subset uses all listed GPUs through DDP. The launcher sets
`CUDA_VISIBLE_DEVICES` before Python starts and verifies the visible-device
count inside every process, so models, k-means, semantic encoding, and CUDA
batches cannot use an unlisted physical GPU. Raw dataset objects, DataLoader
storage, and disk caches remain on CPU/storage by design. Select subsets with
`SUBSETS=Full_Beauty,Toys_and_Games`. Intermediate DIN weights, T5 vectors and
semantic-ID trees are cached under the shared `cache/eager` directory. Run
outputs contain only the training log, `config.json`, `metrics.json`, and the
top-10 score file; no model checkpoint is written into `outputs`.

Defaults preserve the upstream EAGER settings: a 19-event history, DIN with 60
sampled negatives, 30k DIN steps, 60k two-stream steps, a one-layer encoder,
two-layer decoders, 96 hidden units, and disabled optional contrastive/guide
ablations. The branch factor is the smallest even integer at least
`ceil(sqrt(num_items))`, so the upstream two-level decoder and its four-row
k-means kernel remain valid for every Amazon-2023 subset.
