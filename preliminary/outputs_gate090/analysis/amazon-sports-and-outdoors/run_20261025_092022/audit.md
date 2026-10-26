# ver4 source audit

Repository revision: `af89d8dd8ab1ea92ebd5f7a08c52d1146af1e560`.

Dirty worktree at run start (preserved, not modified by this analysis):
- `M preliminary/aggregated/plot_aggregated.py`
- ` M preliminary/experiment.py`
- ` M preliminary/prepare_missing.py`
- ` M preliminary/run.sh`
- ` M preliminary_grouped/additional_datasets.py`
- ` M preliminary_grouped/finalize_additional.py`
- ` M preliminary_grouped/plot_grouped_preliminary.py`
- ` M preliminary_grouped/plot_grouped_row.py`
- ` M preliminary_grouped/run_grouped_preliminary.sh`
- ` M ver4/src/model_mamba_rl.py`
- ` M ver4/src/train_mamba_rl.py`
- ` M ver5/outputs_mamba_rl/steam/steam_tune_lr2e4_ep10_noes_20261024_153111/train_20261024_153119.log`
- `?? preliminary/gate_sweep_gpu0.log`
- `?? preliminary/outputs_gate030/`
- `?? preliminary/outputs_gate070/`
- `?? preliminary/outputs_gate090/`
- `?? preliminary/run_gate_sweep.sh`
- `?? preliminary_grouped/additional_gate_sweep_gpu1.log`
- `?? preliminary_grouped/gate_stream_gpu1.log`
- `?? preliminary_grouped/outputs_gate030/`
- `?? preliminary_grouped/outputs_gate070/`
- `?? preliminary_grouped/outputs_gate090/`
- `?? preliminary_grouped/run_additional_gate_sweep_gpu1.sh`
- `?? preliminary_grouped/run_gate_stream.py`
- `?? preliminary_grouped/run_gate_stream_gpu1.sh`

## Data and model contract

- Input is `ver4/src/train_mamba_rl.py`'s exact schema-v2 cached `InteractionData`; no legacy digest substitution.
- `ver4/src/data.py` uses one-pass user/item minimum interactions of 5, sorts each user by timestamp, then holds out the last two items for validation and test.
- Validation history is train-only; test history is train plus the observed validation item. Histories are truncated before any reversal.
- Sequence order for model and text diagnostics: chronological. In inverse mode, lag 1 means the last element of the reversed input (the oldest item in the truncated original history).
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
