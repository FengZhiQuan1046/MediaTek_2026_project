# Additional preliminary datasets

GPU0 / tmux `ver5_gpu0`: Steam → Yelp19. GPU1 / tmux `ver5_gpu1`: MovieLens1M → Yelp23.
Each dataset trains the original preliminary single-long Mamba+LoRA+LightGCN
for at most3 joint epochs, then exports its individual-item deletion signals.
No model weights are saved. Every GPU runs only one dataset job at a time.

Original Amazon plots/data remain unchanged. The same test-history definition
(train + observed validation, test target excluded),100-item cap, seed25252,
maximum2000 sampled users, and strict consecutive gap >30 days are used.

Yelp19/23 labels refer to **dated, uniquely matched subsets** of their original
MAPLE positive user-item-rating interactions. The historic Yelp2018 HF mirror
recovers84.43% of Yelp19 positive interactions; the recent Yelp mirror
recovers96.77% of Yelp23. Unmatched and ambiguous dates are excluded, not
imputed. Chronological splits are newly built from actual dates. These differ
from the older full MAPLE source-index splits; see each provenance.json.

Each dataset folder stores:
- `group_count_summary.json`: average groups/user for all eligible users and
  the sampled plotting cohort, with capped and full histories distinguished.
- `intermediate/per_user_group_counts.csv.gz`: reusable counts per original user.
- `intermediate/job.json`: exact prepared cache path and checksum.
- `intermediate/training/`: training configuration, metrics, logs, model item
  deletion CSV, and completion marker. A completed run is reused.
- `intermediate/signals/`: original per-position semantic/cluster signals,
  coverage per user, summary CSVs, diagnostics, and manifest. Reused on rerun.
- `intermediate/timestamps.json.gz` and `time_group_map.json.gz`: dated histories
  and recent-first group assignment per original lag, to regroup without training.
- `<dataset>.json`, plotted_data.csv, five standalone plots and the (a,b,c)
  compact row after analysis finishes. PNG/PDF/SVG are provided.

The `aggregated/` folder is automatically generated after all four jobs finish,
including the four-dataset row and a mean-groups-per-user chart. Raw downloads,
SQL databases, normalized tables, tensors and pretrained model caches stay in
`/workspace/P78123011/cache`. HF source versions are in
`cache/raw/dated_analysis/sources.json`.

To repeat statistics/plots, run the same queue command with the same run label;
completed training and per-position signals are reused. A changed raw-data
version or train split must use a new run/cache folder. Changing temporal gap
rules requires regrouping real timestamps, not training the recommender again.
