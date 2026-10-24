# SIGMA with the ver5 data boundary

The upstream SIGMA source and MIT license are preserved under `upstream/`.
`upstream_source.json` records the source revision and SHA-256. The model loads
those original class and method definitions, replacing only RecBole's base
attributes and the CUDA-specific Mamba backend with the existing mambapy 1.2.0
backend. No RecBole dataset creation, filtering, reindexing or re-splitting runs.
The upstream gated forward/backward Mamba, GRU, convolution, gating and CE loss
are retained, including the original fixed 45-position reversal rule.

`run.sh 1` runs Steam, Yelp19 and Yelp23 in order. It calls the shared project
entry point, which requires the exact same `InteractionData` cache used by ver5.
It fails if that cache is missing. The loaded user/item IDs, user/item filtering,
training histories, validation targets and test targets are identical to ver5.
Test inputs are training histories plus the validation target; validation inputs
contain training history only. Full-catalog evaluation uses all eligible users,
seen-item masking and the project's conservative `score >= target_score` ties.
The padding ID shift exists only within the neural network, with no data remapping.

Both training prefixes and evaluation inputs use complete observed histories
(`SIGMA_MAXLEN=0`), matching the new unrestricted ver5 preference input. The
objective and architecture remain SIGMA-specific. Training defaults to 20 epochs
with patience 10, Adam at 1e-3, effective batch 128 (micro-batch 4 with gradient accumulation), evaluation batch 4, hidden size 64 and one gated layer.
Set `EPOCHS`, `BATCH_SIZE`, or `EVAL_BATCH_SIZE` to override run settings.

Outputs are `outputs/<dataset>/<run>/`: full-precision `evaluations.jsonl`,
`latest_evaluation.json`, configuration, exact-cache metadata, training log,
final metrics and status. No model weights are saved. Raw data, model caches,
console progress and preprocessing artifacts stay in `/workspace/P78123011/cache`.
Yelp releases lack real timestamps; the existing ver5 source-row ordering is
retained. Raw and normalized files are unchanged.
