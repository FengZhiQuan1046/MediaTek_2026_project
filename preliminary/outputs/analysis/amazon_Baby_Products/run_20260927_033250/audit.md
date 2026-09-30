# ver4 source audit

Repository revision: `310d59568aafab588f5592f7851b21e12e72ce14`.

Dirty worktree at run start (preserved, not modified by this analysis):
- `M preliminary/aggregated/README.md`
- ` D preliminary/aggregated/figure0_preliminary_overview.pdf`
- ` D preliminary/aggregated/figure0_preliminary_overview.png`
- ` D preliminary/aggregated/figure0_preliminary_overview.svg`
- ` M preliminary/aggregated/figure1_alignment.pdf`
- ` M preliminary/aggregated/figure1_alignment.png`
- ` M preliminary/aggregated/figure1_alignment.svg`
- ` M preliminary/aggregated/figure2_preference.pdf`
- ` M preliminary/aggregated/figure2_preference.png`
- ` M preliminary/aggregated/figure2_preference.svg`
- ` M preliminary/aggregated/figure3_influence.pdf`
- ` M preliminary/aggregated/figure3_influence.png`
- ` M preliminary/aggregated/figure3_influence.svg`
- ` M preliminary/aggregated/figure4_coverage.pdf`
- ` M preliminary/aggregated/figure4_coverage.png`
- ` M preliminary/aggregated/figure4_coverage.svg`
- ` M preliminary/aggregated/figureS1_influence_all_positions.pdf`
- ` M preliminary/aggregated/figureS1_influence_all_positions.png`
- ` M preliminary/aggregated/figureS1_influence_all_positions.svg`
- ` M preliminary/aggregated/plot_aggregated.py`
- ` M preliminary/aggregated/sources.json`
- ` M preliminary/experiment.py`
- `?? preliminary/aggregated/figure0_acd_row.pdf`
- `?? preliminary/aggregated/figure0_acd_row.png`
- `?? preliminary/aggregated/figure0_acd_row.svg`
- `?? preliminary/aggregated/figure0_b_preference.pdf`
- `?? preliminary/aggregated/figure0_b_preference.png`
- `?? preliminary/aggregated/figure0_b_preference.svg`
- `?? preliminary/outputs/analysis/amazon-all-beauty/run_20260927_033236/`
- `?? preliminary/outputs/analysis/amazon_Baby_Products/run_20260927_033250/`

## Data and model contract

- Input is `ver4/src/train_mamba_rl.py`'s exact schema-v2 cached `InteractionData`; no legacy digest substitution.
- `ver4/src/data.py` uses one-pass user/item minimum interactions of 5, sorts each user by timestamp, then holds out the last two items for validation and test.
- Validation history is train-only; test history is train plus the observed validation item. This analysis caps both at 100 interactions by default.
- The cache preserves integer item IDs and item text, but not the item-ID reverse mapping, timestamps or per-text metadata/review provenance.
- Item IDs are built from all filtered interactions before splitting; matching pools, popularity counts, reference cluster fit and text features selected for fit use training items only.
- `ver4/run_amazons_full_rl.sh` specifies frozen Mamba item encoding, dimension 128 online model, LoRA rank 16, long horizon 100, short window 10, and Long/Short/Preference/Graph enabled by default.
- The pretrained Mamba text encoder is an offline item-feature cache; online selective-state specialists and preference GRU are distinct modules. When Long is disabled, ver4 limits Preference to the short window too.
- Missing exact caches are prepared before this analysis by the fixed ver4 LoRA Mamba + LightGCN training path; model weights/checkpoints are not saved.
- Training graph and catalog priors are built from training histories, but they are not used directly by this text-hash diagnostic.
- Figure 3 uses leave-one-item-out scores computed from the restored best trained model before that model is released. No checkpoint or model weights are saved.
- Figures 1, 2, and 4 remain model-free text diagnostics and must not be interpreted as trained-state interventions.
- The auxiliary text-hash feature is a deterministic 1,024-dimensional word/bigram representation. Cluster labels are fitted on sampled training items only; they are *proxy* preferences.
- Random controls match training-popularity quintiles within the same Amazon dataset and exclude observed history. Exact-bin and widened-bin frequencies are in diagnostics.json.
- Every cosine is normalized with the mean and standard deviation of random distinct usable item pairs sampled uniformly from the subset's complete catalog. Normalizer values are stored in diagnostics.json.
- Day-based analysis is unavailable because timestamps were discarded by the ver4 cache. Interaction lag must not be described as elapsed time.
- User bootstrap CIs are conditional on this split, feature proxy and (when capped) sampled users; they do not measure training-seed uncertainty.
