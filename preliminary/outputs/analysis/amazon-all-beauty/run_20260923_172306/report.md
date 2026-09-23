# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.

## Coverage
- amazon-all-beauty: 1457/1457 users; 4090 items; text coverage 1.000; 3161 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-all-beauty

- Fixed-K16 normalized item excess cosine, lag 1: 1.3640 [0.7334, 2.0584], n=105.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.2236 [0.0950, 0.3721], n=105.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 1.1403 [0.5234, 1.8631], n=105.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0251 [0.0024, 0.0490], n=105.
- Old-history item-level pairwise accuracy: 0.5964 [0.5510, 0.6376], n=194.
- Old-history preference-proxy pairwise accuracy: 0.5737 [0.5370, 0.6090], n=194.
- Paired preference-minus-item accuracy: -0.0227 [-0.0697, 0.0249], n=194.
- Fraction of capped history in old low-alignment positions: 0.2215 [0.1993, 0.2449], n=194 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0360 [-0.0535, -0.0160], n=169.
- Full-minus-recent mean-pooling margin: 0.0131 [-0.0895, 0.1095], n=194.


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
