# Why LLMRec is excluded from the GPU 1 comparison

The supplied [LLMRec.pdf](LLMRec.pdf) is *LLMRec: Large Language Models with Graph Augmentation for Recommendation* (WSDM 2024; arXiv:2311.00423). Section 4.1.1–4.1.2, page 6, describes GPT-3.5 API calls for interaction, item-attribute, and user-profile augmentation, and `text-embedding-ada-002` API calls for embeddings. The paper reports approximate API costs of USD 15.65, 20.40, and 3.12 for its two study datasets, Netflix and MovieLens. It says recommender training used a 24 GB RTX 3090. These are **not** measured A100 peak memory or Full Beauty test-user throughput, so they cannot replace missing measurements in the bar chart. [Official paper](https://arxiv.org/pdf/2311.00423)

The previous local Full Beauty pilot at `gpu1_runs/llmrec_full_beauty/` measured a different workload:

- `Reproduce/LLMRec/train.py` declares `text=hashed_ver4_metadata image=zeros llm_edges=train_proxy`, so it does not perform the paper's GPT augmentation, API embedding, or CLIP-ViT image encoding.
- Its `evaluate()` computes `model(*graphs)` **before** starting the timer. The reported 17,714.6 users/s measures the downstream score and rank loop after graph embeddings exist. It does not time the model forward pass, graph construction, or LLM augmentation.
- Its 656 MiB NVML peak is the process peak of that local proxy pilot. It does not include remote API compute, and it cannot be presented as the end-to-end memory requirement of the published LLMRec method.

Therefore the plot, `gpu1_summary.*`, `archive_summary.*`, and plot source exclude LLMRec. The original pilot files remain untouched for audit. A defensible LLMRec row would need the published augmentation artifacts or a faithful reconstruction, plus an explicitly scoped timing and memory protocol on physical GPU 1. The paper itself does not report the requested A100 values.
