# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out; model input: **reversed**.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.
In inverse mode, lag 1 is nearest the end of the reversed input, which is the oldest item in the truncated chronological prefix.

## Coverage
- amazon-sports-and-outdoors: 2000/625563 users; 401572 items; text coverage 1.000; 20000 train-reference items.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-sports-and-outdoors

- Fixed-K16 normalized item excess cosine, lag 1: 0.3067 [0.0991, 0.5404], n=114.
- Fixed-K16 normalized item excess cosine, lags 9-16: 0.3823 [0.2367, 0.5388], n=114.
- Paired lag-1 minus lag-9:16 normalized excess cosine: -0.0756 [-0.3461, 0.1944], n=114.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0763 [0.0431, 0.1086], n=114.
- Old-history item-level pairwise accuracy: 0.6908 [0.6541, 0.7265], n=266.
- Old-history preference-proxy pairwise accuracy: 0.6116 [0.5790, 0.6431], n=266.
- Paired preference-minus-item accuracy: -0.0792 [-0.1173, -0.0399], n=266.
- Fraction of capped history in old low-alignment positions: 0.1329 [0.1185, 0.1471], n=266 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: 0.0208 [-0.0061, 0.0509], n=205.
- Full-minus-recent mean-pooling margin: 0.2678 [0.1757, 0.3544], n=266.


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
