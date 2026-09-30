# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out; model input: **reversed**.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.
In inverse mode, lag 1 is nearest the end of the reversed input, which is the oldest item in the truncated chronological prefix.

## Coverage
- amazon:Baby_Products: 2000/184584 users; 77324 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon:Baby_Products

- Fixed-K16 normalized item excess cosine, lag 1: -0.0700 [-0.2413, 0.1076], n=84.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.1639 [0.0882, 0.2503], n=84.
- Paired lag-1 minus lag-9:16 normalized excess cosine: -0.2339 [-0.4037, -0.0367], n=84.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0091 [-0.0093, 0.0291], n=84.
- Old-history item-level pairwise accuracy: 0.6065 [0.5679, 0.6439], n=246.
- Old-history preference-proxy pairwise accuracy: 0.5563 [0.5271, 0.5873], n=246.
- Paired preference-minus-item accuracy: -0.0502 [-0.0890, -0.0100], n=246.
- Fraction of capped history in old low-alignment positions: 0.1462 [0.1304, 0.1621], n=246 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0255 [-0.0445, -0.0039], n=188.
- Full-minus-recent mean-pooling margin: 0.1303 [0.0713, 0.1943], n=246.


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
