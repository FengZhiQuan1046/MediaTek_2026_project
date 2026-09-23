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

- Fixed-K16 normalized item excess cosine, lag 1: 1.3867 [0.7473, 2.0866], n=105.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.2268 [0.0956, 0.3818], n=105.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 1.1599 [0.5363, 1.9157], n=105.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0436 [0.0140, 0.0773], n=105.
- Old-history item-level pairwise accuracy: 0.5957 [0.5514, 0.6387], n=194.
- Old-history preference-proxy pairwise accuracy: 0.5675 [0.5311, 0.6017], n=194.
- Paired preference-minus-item accuracy: -0.0282 [-0.0754, 0.0224], n=194.
- Fraction of capped history in old low-alignment positions: 0.2209 [0.1985, 0.2439], n=194 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0306 [-0.0532, -0.0034], n=171.
- Full-minus-recent mean-pooling margin: -0.0043 [-0.1053, 0.0944], n=194.


## How to read the results

`alignment_summary.csv` shows whether recent and older items look more like the next item than matched random items do.
If a confidence interval crosses zero, the experiment cannot say whether that history range helps.
`information_coverage.csv` shows what percentage of all positive next-item signal is already contained in the most recent X items.
`old_proxy_summary.csv` compares a direct old-item match with a broader preference summary, using the same targets and negatives.
A negative full-minus-recent value means that adding older history made this simple score worse; it does not by itself prove that Mamba fails.
The preparatory ver4 run writes statistics but no checkpoint or weights. The text-based diagnostic does not reuse the trained state.

## Figures

- figures/figure1_history_alignment.pdf
- figures/figure2_preference_proxy.pdf
- figures/figure3_old_encoding_proxy.pdf
- figures/figure4_recent_information_coverage.pdf
