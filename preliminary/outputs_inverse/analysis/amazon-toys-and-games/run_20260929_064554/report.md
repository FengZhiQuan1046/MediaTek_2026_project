# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out; model input: **reversed**.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.
In inverse mode, lag 1 is nearest the end of the reversed input, which is the oldest item in the truncated chronological prefix.

## Coverage
- amazon-toys-and-games: 2000/564666 users; 333160 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-toys-and-games

- Fixed-K16 normalized item excess cosine, lag 1: 0.2680 [0.0841, 0.4664], n=147.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.5339 [0.3777, 0.7085], n=147.
- Paired lag-1 minus lag-9:16 normalized excess cosine: -0.2660 [-0.4285, -0.0946], n=147.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0811 [0.0480, 0.1177], n=147.
- Old-history item-level pairwise accuracy: 0.7129 [0.6815, 0.7469], n=322.
- Old-history preference-proxy pairwise accuracy: 0.6309 [0.6038, 0.6576], n=322.
- Paired preference-minus-item accuracy: -0.0820 [-0.1151, -0.0473], n=322.
- Fraction of capped history in old low-alignment positions: 0.1300 [0.1168, 0.1437], n=322 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0029 [-0.0253, 0.0217], n=229.
- Full-minus-recent mean-pooling margin: 0.2983 [0.2261, 0.3808], n=322.


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
