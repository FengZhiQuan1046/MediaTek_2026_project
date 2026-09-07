# LLaRA reproduction for MediaTek 2026

This integration imports the upstream LLaRA `MInterface`, MLP projector,
SASRec class, LoRA targets, language-model loss, prompt embedding replacement,
and HR@1 generation evaluation. Amazon interactions are loaded exclusively
through ver4's one-pass 5-core and chronological leave-two-out protocol.

The authors do not publish Amazon recommender checkpoints. On first use, the
upstream SASRec class is pretrained on the ver4 training partition and cached
under the shared `cache/llara` directory. No model checkpoint is written below
the experiment output directory.

Run the four requested subsets with two RTX 4090 GPUs:

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce/LLaRA
GPU_IDS=0,1 bash run.sh
```

`run.sh` resolves the existing Llama-2 snapshot below the same workspace cache
used by ver4. If it is absent, set `LLM_PATH` to an accessible local snapshot
or authenticate with Hugging Face before using the gated model ID.

Llama-2-7B LoRA training uses Lightning DDP, BF16, gradient checkpointing,
batch size 1 per GPU, and gradient accumulation 16. Outputs contain only
`config.json`, `metrics.json`, and `train_{time}.log`.

Candidate-based LLM generation is expensive on the full Sports/Toys sets.
Defaults cap training at 10,000 prefix examples and evaluation at 1,000 users.
Set `MAX_TRAIN_SAMPLES=0 EVAL_USER_LIMIT=0` for all eligible examples/users.
