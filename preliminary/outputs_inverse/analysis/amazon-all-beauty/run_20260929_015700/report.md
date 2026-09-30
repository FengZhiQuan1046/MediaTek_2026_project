# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out; model input: **reversed**.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.
In inverse mode, lag 1 is nearest the end of the reversed input, which is the oldest item in the truncated chronological prefix.

## Coverage
- amazon-all-beauty: 1457/1457 users; 4090 items; text coverage 1.000; 3161 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-all-beauty

- Fixed-K16 normalized item excess cosine, lag 1: 0.1972 [-0.0490, 0.4519], n=105.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.1791 [0.0417, 0.3368], n=105.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 0.0181 [-0.2685, 0.3092], n=105.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0096 [-0.0120, 0.0342], n=105.
- Old-history item-level pairwise accuracy: 0.6300 [0.5845, 0.6746], n=194.
- Old-history preference-proxy pairwise accuracy: 0.5733 [0.5367, 0.6119], n=194.
- Paired preference-minus-item accuracy: -0.0567 [-0.1019, -0.0093], n=194.
- Fraction of capped history in old low-alignment positions: 0.2190 [0.1971, 0.2406], n=194 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0291 [-0.0489, -0.0050], n=168.
- Full-minus-recent mean-pooling margin: 0.1614 [0.0655, 0.2667], n=194.


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
