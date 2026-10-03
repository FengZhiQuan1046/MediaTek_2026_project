# ver5: Short and Preference Multi-LoRA Mamba Recommender

ver5 has two sequence agents. The short agent reads the latest `SHORT_WINDOW`
interactions. The preference agent reads the complete supplied history up to
`MAX_HISTORY`, maps each item to soft preference prototypes, and encodes the
prototype trajectory with its own Mamba-style selective-state recurrence and
LoRA bank. It does not use a GRU. The two agents share cached features from a
frozen pretrained Mamba text encoder and optional LightGCN item features, but
their trainable LoRA parameters are separate. A coordinator combines their
candidate scores using preference-transition probability and certainty.

With `JOINT_EPOCH=0`, training has one DL stage: every enabled agent, the
coordinator, shared item projection, and optional LightGCN train together for
`SPECIALISTS_EPOCH` epochs. No RL loss is computed, even if `RL_COEF` is set.
With `JOINT_EPOCH>0`, training retains supervised specialist pretraining
followed by joint supervised ranking and sampled Top-K policy gradient. That
reward is an offline NDCG proxy from future items in the training sequence,
not feedback for unshown recommendations. `RL_COEF=0` disables the policy
gradient in the two-stage mode.

The default limits in `run_mamba_rl.sh` are `SHORT_WINDOW=10` and
`MAX_HISTORY=100`. With both agents enabled, preference still reads up to 100
interactions when short reads only 10. `USE_SHORT` and `USE_PREFERENCE` can be
set independently; there is no long agent or `USE_LONG` option in ver5.
`USE_GCN` and `USE_COORDINATOR` are additional ablation switches in the suite
launchers. A disabled module is excluded from forward computation and training.
With both sequence agents enabled and the coordinator disabled, their scores
are mixed by the fixed `PREFERENCE_SCORE_WEIGHT` instead.

## Run

```bash
cd /workspace/P78123011/MediaTek_2026_project/ver5
PYTHON_BIN=/dataspace/P78123011/miniconda3/envs/py31014/bin/python bash run_amazons_full_rl.sh
```

`PYTHON_BIN` can point to an existing Python environment with the project
dependencies. No new virtual environment is needed. The launchers write new
results under ver5 and label ablations with `S`, `P`, and `G`. The copied ver4
outputs and paper tables predate this architecture change; retrain and evaluate
before using them as ver5 results. See `LONG_SHORT.md` for history-length
cohort evaluation.

## Verification

```bash
/dataspace/P78123011/miniconda3/envs/py31014/bin/python -m unittest discover -s tests
bash -n run_mamba_rl.sh run_amazons_full_rl.sh run_long_short.sh
```
