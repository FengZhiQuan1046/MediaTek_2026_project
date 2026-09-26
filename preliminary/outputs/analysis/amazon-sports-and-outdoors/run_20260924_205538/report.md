# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.

## Coverage
- amazon-sports-and-outdoors: 2000/625563 users; 401572 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-sports-and-outdoors

- Fixed-K16 normalized item excess cosine, lag 1: 2.0380 [1.3276, 2.8375], n=114.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.2675 [0.1693, 0.3683], n=114.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 1.7706 [1.1129, 2.5375], n=114.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0603 [0.0308, 0.0899], n=114.
- Old-history item-level pairwise accuracy: 0.6406 [0.6068, 0.6752], n=266.
- Old-history preference-proxy pairwise accuracy: 0.5463 [0.5154, 0.5765], n=266.
- Paired preference-minus-item accuracy: -0.0943 [-0.1352, -0.0533], n=266.
- Fraction of capped history in old low-alignment positions: 0.1621 [0.1472, 0.1775], n=266 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0101 [-0.0316, 0.0125], n=215.
- Full-minus-recent mean-pooling margin: -0.1019 [-0.1642, -0.0410], n=266.


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
