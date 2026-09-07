# LLMRec reproduction for MediaTek 2026

This adapter imports the upstream `LLMRec/Models.py` model and retains its
full-graph propagation, multimodal/profile fusion, BPR, pruning, feature
regularization, and augmentation objectives. Dataset loading, one-pass 5-core
filtering, ID mapping, and chronological leave-two-out come directly from ver4.

Amazon Reviews 2023 does not include the official LLMRec GPT augmentation or
Netflix-style poster features. Consequently this adapter uses deterministic
hashed ver4 metadata for text/item/user-profile features, a zero vector for the
unavailable image modality, and training-edge proxies for augmented samples.
Results must be labelled `hashed_ver4_metadata_proxy`; they are not official
GPT-3.5-augmented LLMRec results.

Run the four requested subsets on two GPUs:

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce/LLMRec
GPU_IDS=0,1 bash run.sh
```

The upstream implementation is single-device and hard-codes CUDA tensors.
Rather than modifying the model into DDP, `run.sh` runs two subsets concurrently,
one per GPU. This is suitable for two 24 GB RTX 4090 cards. Each run writes only
`config.json`, `metrics.json`, and `train_{time}.log` under
`outputs/{subset}/llmrec_{time}_r{repeat}/`.
