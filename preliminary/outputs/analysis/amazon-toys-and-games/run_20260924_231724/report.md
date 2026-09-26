# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.

## Coverage
- amazon-toys-and-games: 2000/564666 users; 333160 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-toys-and-games

- Fixed-K16 normalized item excess cosine, lag 1: 1.3878 [0.9147, 1.9423], n=147.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.5312 [0.3637, 0.7218], n=147.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 0.8565 [0.4056, 1.3558], n=147.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0863 [0.0533, 0.1226], n=147.
- Old-history item-level pairwise accuracy: 0.6551 [0.6219, 0.6873], n=322.
- Old-history preference-proxy pairwise accuracy: 0.5843 [0.5563, 0.6109], n=322.
- Paired preference-minus-item accuracy: -0.0708 [-0.1071, -0.0332], n=322.
- Fraction of capped history in old low-alignment positions: 0.1482 [0.1347, 0.1625], n=322 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0085 [-0.0292, 0.0148], n=245.
- Full-minus-recent mean-pooling margin: -0.0513 [-0.1049, -0.0020], n=322.


## How to read the results

`alignment_summary.csv` shows whether recent and older items look more like the next item than matched random items do.
If a confidence interval crosses zero, the experiment cannot say whether that history range helps.
`information_coverage.csv` shows what percentage of all positive next-item signal is already contained in the most recent X items.
`old_proxy_summary.csv` compares a direct old-item match with a broader preference summary, using the same targets and negatives.
For Figure 3, positive influence means keeping that exact item raises the trained model's true-target margin over the same sampled negatives; negative influence means it lowers the margin.
The preparatory run writes these influence statistics but no checkpoint or weights.

## Figures

- figures/figure1_history_alignment.pdf
- figures/figure2_preference_proxy.pdf
- figures/figure3_old_encoding_proxy.pdf
- figures/figure4_recent_information_coverage.pdf
