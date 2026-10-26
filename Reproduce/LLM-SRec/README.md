# LLM-SRec on the shared Yelp protocol

The vendored official implementation is pinned in `upstream_source.json`.
`train.py` uses its frozen Llama-3.2-3B-Instruct, trainable user/item output
embeddings, projection heads, four-candidate retrieval loss, SASRec matching loss,
and uniformity objective. A 64-dimensional SASRec teacher is pretrained on the
training partition and selected by validation NDCG@10. LLM adapters use the final
training target and ten preceding items, as upstream. Missing real timestamps are
omitted from prompts. The teacher uses this project's SASRec implementation.

Local compatibility patches in `upstream/models/seqllm4rec.py` allow a configurable
model path, load the frozen LLM without bitsandbytes quantization, and supply
padding attention masks. Retrieval evaluation uses the base transformer hidden
states to avoid allocating unused vocabulary logits. Model weights are not saved;
best teacher and adapter states remain in memory for final evaluation.

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce
bash run_llm_yelp.sh LLM-SRec 0 my_run
bash run_llm_yelp.sh LLMRec 0 my_run
```

Run each command in a separate tmux. Each session processes Yelp19, Yelp23, MovieLens 1M, then Steam, on GPU 0 only.
`prepare_hf_yelp.py` downloads pinned HF releases under `cache/hf_yelp`, normalizes
them under `cache/normalized`, and records source hashes. Both models use the same
ver5 cache identity, preprocessing, and leave-two-out split. If old caches are
missing, they are rebuilt and their hashes recorded in `data_info.json`.

`outputs/{dataset}/{run_id}/` includes `config.json`, `data_info.json`, logs,
`status.txt`, `metrics.json`, and the SIGMA-compatible `evaluations.jsonl` and
`latest_evaluation.json`. Teacher evaluations are labelled `sasrec_teacher`;
LLM evaluations use `recommender` and `best_validation_model`. Evaluation covers
all eligible users and the full catalog, masking all previously consumed items.
Metrics are Recall, Hit and NDCG at 5 and 10. No user sampling or event cap is used.

Defaults: 2 teacher epochs and 6 adapter epochs (8 total per dataset), validation patience 10,
training batch 8, evaluation batch 16, seed 25252. Both jobs share GPU 0 when
launched with the commands above.

LLMRec clamps epochs to 8. LLM-SRec caps teacher plus adapter training at 8 total epochs. Every epoch in both stages evaluates both validation and test; model selection still uses validation only. Each paired evaluation updates metrics.json and the SIGMA-compatible JSONL log.

The four-dataset launcher is `../run_llm_yelp.sh`. `../prepare_hf_four.py` downloads full HF ML1M and Steam releases into workspace cache. Data loading calls the unmodified ver5 loaders and build_data; Steam uses ver5 load_steam_events. Data fingerprints and loader source hashes are recorded in data_info.json. The HF MovieLens raw files are verified against the existing original release when available.
