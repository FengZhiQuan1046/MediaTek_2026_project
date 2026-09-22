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

## Interpretation

Read `alignment_summary.csv` for observed, random and excess curves and user CIs.
A CI crossing zero does not establish equivalence; no threshold was selected.
`old_proxy_summary.csv` gives same-candidate model-free encoding diagnostics.
A negative full-minus-recent margin is evidence only for mean-pooling interference, not Mamba.
No trained checkpoint was available or created, so learned-state influence and ver4 preference selectivity remain untested.

## Figures

- figures/figure1_history_alignment.pdf
- figures/figure2_preference_proxy.pdf
- figures/figure3_old_encoding_proxy.pdf
