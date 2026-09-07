# EAGER reproduction

This wrapper runs the upstream `P78123011/EAGER` implementation on the exact
ver4 Amazon-2023 preprocessing and chronological leave-two-out split.

```bash
pip install -r requirements.txt
bash run.sh
```

`GPU_IDS=0,1` runs two subsets concurrently, one per GPU. Select subsets with
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
