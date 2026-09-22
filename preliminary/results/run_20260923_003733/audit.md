# ver4 source audit

Repository revision: `a34ae510bf3fc8770c2f3f62c7bd057570528183`.

Dirty worktree at run start (preserved, not modified by this analysis):
- `D Reproduce/EAGER/outputs/Sports_and_Outdoors/eager_20260909_072330_r1/train_20260909_072332.log`
- ` D Reproduce/EAGER/outputs/Sports_and_Outdoors/eager_20260909_181141_r1/train_20260909_181143.log`
- ` D Reproduce/EAGER/outputs/Sports_and_Outdoors/eager_20260909_181320_r1/train_20260909_181323.log`
- ` D Reproduce/EAGER/outputs/Toys_and_Games/eager_20260909_075108_r1/train_20260909_075110.log`
- ` D Reproduce/EAGER/outputs/Toys_and_Games/eager_20260909_181320_r1/train_20260909_181323.log`
- ` M Reproduce/EAGER/run.sh`
- ` M Reproduce/EAGER/train.py`
- `?? Reproduce/EAGER/outputs/Sports_and_Outdoors/eager_20260922_000631_r1/`
- `?? Reproduce/EAGER/outputs/Sports_and_Outdoors/eager_20260922_035932_r1/`
- `?? guideline.txt`
- `?? preliminary/`

## Data and model contract

- Input is `ver4/src/train_mamba_rl.py`'s exact schema-v2 cached `InteractionData`; no legacy digest substitution.
- `ver4/src/data.py` uses one-pass user/item minimum interactions of 5, sorts each user by timestamp, then holds out the last two items for validation and test.
- Validation history is train-only; test history is train plus the observed validation item. This analysis caps both at 100 interactions by default.
- The cache preserves integer item IDs and item text, but not the item-ID reverse mapping, timestamps or per-text metadata/review provenance.
- Item IDs are built from all filtered interactions before splitting; matching pools, popularity counts, reference cluster fit and text features selected for fit use training items only.
- `ver4/run_amazons_full_rl.sh` specifies frozen Mamba item encoding, dimension 128 online model, LoRA rank 16, long horizon 100, short window 10, and Long/Short/Preference/Graph enabled by default.
- The pretrained Mamba text encoder is an offline item-feature cache; online selective-state specialists and preference GRU are distinct modules. When Long is disabled, ver4 limits Preference to the short window too.
- Training graph and catalog priors are built from training histories, but they are not used in this checkpoint-free diagnostic.
- No trained ver4 state, optimizer, LoRA tensor, checkpoint or weights are loaded or saved here. Therefore Mamba-state harm and learned preference benefit cannot be tested in this run.
- The auxiliary text-hash feature is a deterministic 1,024-dimensional word/bigram representation. Cluster labels are fitted on sampled training items only; they are *proxy* preferences.
- Random controls match training-popularity quintiles within the same Amazon dataset and exclude observed history. Exact-bin and widened-bin frequencies are in diagnostics.json.
- Day-based analysis is unavailable because timestamps were discarded by the ver4 cache. Interaction lag must not be described as elapsed time.
- User bootstrap CIs are conditional on this split, feature proxy and (when capped) sampled users; they do not measure training-seed uncertainty.
