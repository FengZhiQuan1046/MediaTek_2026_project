# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.

## Coverage
- amazon:Baby_Products: 2000/184584 users; 77324 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon:Baby_Products

- Fixed-K16 normalized item excess cosine, lag 1: 0.5533 [0.2374, 0.9234], n=84.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.1748 [0.0783, 0.2864], n=84.
- Paired lag-1 minus lag-9:16 normalized excess cosine: 0.3785 [0.0529, 0.7827], n=84.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0045 [-0.0156, 0.0280], n=84.
- Old-history item-level pairwise accuracy: 0.5734 [0.5380, 0.6102], n=246.
- Old-history preference-proxy pairwise accuracy: 0.5200 [0.4923, 0.5479], n=246.
- Paired preference-minus-item accuracy: -0.0534 [-0.0936, -0.0138], n=246.
- Fraction of capped history in old low-alignment positions: 0.1489 [0.1333, 0.1646], n=246 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0245 [-0.0468, 0.0004], n=195.
- Full-minus-recent mean-pooling margin: -0.0133 [-0.0559, 0.0295], n=246.


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
