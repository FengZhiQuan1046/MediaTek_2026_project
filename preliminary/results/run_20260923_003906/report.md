# Preliminary ver4 diagnostic

Mode: **pilot**; split: chronological leave-two-out.

These are hypothesis tests, not proofs of a noise cutoff or Mamba failure.
Text-hash features are independent of recommendation training but item-text provenance is unknown.
The preference cluster is a train-only coarse proxy, not the learned ver4 preference agent.
No timestamps survive the ver4 cached split; all lag results are interaction-based.

## Coverage
- amazon-sports-and-outdoors: 2000/625563 users; 401572 items; text coverage 1.000; 20000 train-reference items.
- amazon-toys-and-games: 2000/564666 users; 333160 items; text coverage 1.000; 20000 train-reference items.
- amazon-all-beauty: skipped; exact ver4 cache unavailable at /workspace/P78123011/cache/mamba_multi_agent_data/amazon-all-beauty_90e3e920169a.pkl.
- amazon:Baby_Products: skipped; exact ver4 cache unavailable at /workspace/P78123011/cache/mamba_multi_agent_data/amazon_Baby_Products_e6e228b87df8.pkl.

## Test-split estimates (95% user-bootstrap intervals)

These are descriptive pilot results. The fixed-K16 contrast uses the *same* users for lags 1 through 16; later bins are omitted.

### amazon-sports-and-outdoors

- Fixed-K16 item excess cosine, lag 1: 0.1121 [0.0714, 0.1569], n=114.
- Fixed-K16 item excess cosine, lags 9-16: 0.0148 [0.0092, 0.0207], n=114.
- Paired lag-1 minus lag-9:16 excess cosine: 0.0973 [0.0605, 0.1410], n=114.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0520 [0.0248, 0.0820], n=114.
- Old-history item-level pairwise accuracy: 0.6363 [0.6007, 0.6703], n=266.
- Old-history preference-proxy pairwise accuracy: 0.5643 [0.5346, 0.5931], n=266.
- Paired preference-minus-item accuracy: -0.0720 [-0.1128, -0.0306], n=266.
- Fraction of capped history in old low-alignment positions: 0.1622 [0.1479, 0.1765], n=266 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0007 [-0.0224, 0.0242], n=221.
- Full-minus-recent mean-pooling margin: -0.0060 [-0.0095, -0.0026], n=266.

### amazon-toys-and-games

- Fixed-K16 item excess cosine, lag 1: 0.0812 [0.0542, 0.1130], n=147.
- Fixed-K16 item excess cosine, lags 9-16: 0.0315 [0.0217, 0.0428], n=147.
- Paired lag-1 minus lag-9:16 excess cosine: 0.0498 [0.0251, 0.0781], n=147.
- Fixed-K16 coarse preference-cluster excess, lags 9-16: 0.0942 [0.0602, 0.1302], n=147.
- Old-history item-level pairwise accuracy: 0.6624 [0.6301, 0.6935], n=322.
- Old-history preference-proxy pairwise accuracy: 0.5761 [0.5484, 0.6035], n=322.
- Paired preference-minus-item accuracy: -0.0863 [-0.1242, -0.0480], n=322.
- Fraction of capped history in old low-alignment positions: 0.1499 [0.1362, 0.1641], n=322 (post-hoc descriptor, not noise prevalence).
- Cluster excess among those positions: -0.0146 [-0.0356, 0.0086], n=254.
- Full-minus-recent mean-pooling margin: -0.0033 [-0.0065, -0.0003], n=322.


## Interpretation

Read `alignment_summary.csv` for observed, random and excess curves and user CIs.
A CI crossing zero does not establish equivalence; no threshold was selected.
`old_proxy_summary.csv` gives same-candidate model-free encoding diagnostics.
A negative full-minus-recent margin is evidence only for mean-pooling interference, not Mamba.
The coarse cluster encoding must not be called successful unless it beats the item-level control on the paired diagnostic; unfavorable results are retained.
No trained checkpoint was available or created, so learned-state influence and ver4 preference selectivity remain untested.

## Figures

- figures/figure1_history_alignment.pdf
- figures/figure2_preference_proxy.pdf
- figures/figure3_old_encoding_proxy.pdf
