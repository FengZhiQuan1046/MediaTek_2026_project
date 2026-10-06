# Memory and runtime overhead study

The requested `RAGER` name was clarified as **EAGER**. The comparison contains seven methods: Ours (ver5), SIGMA, Mamba4Rec, EAGER, BERT4Rec, ReSID, and LLM-SRec. LLMRec was removed after auditing its local proxy against [LLMRec.pdf](LLMRec.pdf); see [llmrec_paper_audit.md](llmrec_paper_audit.md).

## Current evidence

Run `python collect_archive.py` to regenerate [archive_summary.csv](archive_summary.csv) and [archive_summary.md](archive_summary.md) from existing **Full Beauty** run directories. The table records the selected source run, logged trainable parameter count, elapsed time between the first and last log lines, and the test throughput reported by each run. The log span includes data loading, training, validation, and evaluation; it is **not** a per-epoch training time. Throughput was recorded under different configurations and GPU environments and is not a controlled speed comparison.

No archived run includes a reliable physical-GPU-1 peak-memory trace, so the archive GPU memory cells are blank. A parameter count is not a VRAM measurement: it excludes some frozen weights, cached vectors, optimizer state, activations, graph buffers, and allocator overhead. Ours' count excludes the frozen Mamba text encoder. ReSID and EAGER have multiple phases; their profiled process peak spans the whole short run. The Mamba4Rec row uses the small backbone (`use_large_mamba=false`); the archived 2.8B variant used model parallelism on two GPUs and cannot serve as a one-GPU-1 measurement.

The local SIGMA and LLM-SRec checkouts did not include project-compatible Full Beauty run artifacts. Short GPU 1 pilots were therefore prepared below. EAGER's archived output covers only the recorded subsets. These differences matter when interpreting the seven rows.

## Measured GPU 1 pilot

Run `python collect_profiles.py` to regenerate [gpu1_summary.csv](gpu1_summary.csv) and [gpu1_summary.md](gpu1_summary.md). The seven included methods were profiled sequentially on physical GPU **1**, an NVIDIA A100 80GB PCIe. Every pilot used Full Beauty and evaluated the full catalog. The pilots are deliberately short and retain each method's own batch size and architecture:

- Ours: cached Mamba-130M item vectors, `JOINT_EPOCH=0`, one DL epoch, at most 1,024 training transitions, short window 4, 32 preference prototypes. The 130M text-encoding cache was already present, so its cold encoding cost is excluded.
- Mamba4Rec: small one-layer backbone, one epoch, at most 10 batches, `mambapy` 1.2.0 pure-PyTorch backend installed in `/tmp/mambapy_overhead_target`.
- BERT4Rec: one epoch, at most 10 batches.
- ReSID: one FAMAE epoch and one recommender epoch, each capped at 10 batches.
- EAGER: T5 semantics, 10 DIN steps and 10 recommender steps; its full validation and test phases ran. The semantic and DIN caches may be warm on later runs.
- SIGMA: upstream SIGMA model and RecBole 1.2.0, one epoch, batch 2,048, leave-two-out split. The missing `mamba_ssm` CUDA extension was replaced with `mambapy` 1.2.0's pure-PyTorch Mamba backend. The RecBole atomic input was exported from the cached Full Beauty chronology, preserving user order and target splits.
- LLM-SRec: upstream LLM-SRec and the official cached LLaMA-3.2-3B-Instruct weights, 8-bit loading, batch 2 for one training step. The missing pretrained SASRec teacher was initialized and trained for three batches from cached Full Beauty histories; this is **not** the official pretrained teacher. The upstream user encoder and full item embedding cache were used, and the timed test scored all 4,090 items for 128 users in batches of eight. Catalog embedding preparation is excluded from test users/s but included in process wall time and peak memory.

The Mamba4Rec pilot used `mambapy==1.2.0` installed into `/tmp/mambapy_overhead_target` and passed through `PYTHONPATH`; that temporary installation must be recreated if `/tmp` is cleared.

The sampled process peak is **allocated GPU process memory**, not a minimum VRAM requirement or a full-training peak. Wall time includes Python startup, cache access, training, and validation/test. No speed or memory ranking should be inferred across these unequal pilot workloads. The raw per-run profile and exact command are saved under `gpu1_runs/<model>/`.

### Ours: evaluation throughput improvement

[benchmark_ours_throughput.py](benchmark_ours_throughput.py) runs the original ver5 pilot on physical GPU 1, then compares its original test evaluator against two evaluation-only variants on the **same trained weights**. It changes no file outside `overhead/` and never launches the GPU memory sampler. Both variants keep batch size 64, full-history preference input, all 4,090 catalog items, scores, masking, and all metrics. The faster variant reuses the catalog item projections already computed before timed evaluation and packs each history batch on CPU before one GPU transfer.

For subsequent ver5 runs, [run_ver5_fast_eval.py](run_ver5_fast_eval.py) accepts the same arguments as `python -m src.train_mamba_rl` and activates the selected evaluation path without changing `src/`. Bind physical GPU 1 with `CUDA_VISIBLE_DEVICES=GPU-e9c44344-a6b8-2d6a-0b1a-6cbf88331918` and pass `--output-run-dir overhead/<your_run>` to keep outputs here. The unmodified `run_long_short.sh` continues to use its original evaluator. A separate launcher smoke run in `gpu1_runs/ours_fast_launcher_smoke/` completed successfully; it did not sample memory.

| Same-weight test evaluator | Test users/s |
|---|---:|
| Original, batch 64 | 2,604.3 |
| Reuse catalog projections, batch 64 (two-run median) | 2,651.0 |
| Reuse projections + packed histories, batch 64 (two-run median) | **2,842.3** |

The selected path is **9.1% faster** than the paired original evaluation. Recall and hit rates are identical; NDCG differs by less than `1e-7` due to reduction order, other numeric diagnostics by less than `1e-4`, and the sampled top-10 item IDs are identical. A batch-size-128 trial was rejected because some sampled top-10 item orders changed. The raw audit and trials are in [throughput_optimized.json](gpu1_runs/ours_throughput_optimized/throughput_optimized.json). `gpu1_summary.*` and the figure use its throughput. Ours' **1,578 MiB memory value remains the earlier measurement**, and the old wall time is omitted from the updated summary row because throughput and memory are from separate runs.

The two newly measured rows are:

| Model | Test users/s | Peak process memory | Total process time | Additional timing |
|---|---:|---:|---:|---|
| SIGMA | 16,638.8 | 29,638 MiB (28.9 GiB) | 5.71 s | Train 1.42 s; test 0.088 s / 1,457 users |
| LLM-SRec | 29.8 | 18,366 MiB (17.9 GiB) | 35.47 s | LLM load 6.35 s; one train step 1.11 s; catalog cache + first user 17.13 s; test 4.29 s / 128 users |

### Two-panel paper figure

Run `python plot_overhead.py` to regenerate [figure_gpu1_overhead.pdf](figure_gpu1_overhead.pdf), [figure_gpu1_overhead.png](figure_gpu1_overhead.png), [figure_gpu1_overhead.svg](figure_gpu1_overhead.svg), and [figure_gpu1_overhead_data.csv](figure_gpu1_overhead_data.csv) from the seven included profiles. [figure_gpu1_overhead.tex](figure_gpu1_overhead.tex) is an ACL-compatible inclusion snippet. The design follows `preliminary/aggregated/plot_aggregated.py`: a serif face, the same blue/orange palette, thin gray spines and grid, and embedded vector PDF text. Ours is orange. Both horizontal axes use logarithmic scales to fit the full range.

Suggested caption: “Full Beauty pilot throughput and sampled GPU process memory on one A100 80GB (physical GPU 1). Throughput is full-catalog test users per second after reusable item representations are ready; memory is the maximum 50 ms NVML process sample over each complete pilot. Ours' throughput uses cached history projections and batch packing, while its memory is the prior unoptimized measurement. SIGMA uses a pure-PyTorch Mamba backend, LLM-SRec uses a three-batch SASRec teacher and a 128-user timed test, and the other methods retain their individual shortened training settings. These measurements describe implementation overhead under the stated pilots rather than matched full-training cost.”

## GPU 1 measurement

`profile_gpu1.py` checks that physical GPU index **1** exists, obtains its UUID, and launches a **direct Python** training command with `CUDA_VISIBLE_DEVICES` set to that UUID. It refuses shell launchers because existing `run.sh` files can override GPU visibility. It samples memory attributed to the child Python process and its descendants using `nvidia-smi`, and writes `profile.json` plus `gpu1_samples.csv`. It exits before launching the command if GPU 1 is unavailable.
Use `--cwd` for modules that must start in their project directory and `--env KEY=VALUE` for additional child variables. The profiler rejects attempts to change GPU-visibility variables through `--env`.

Example for one BERT4Rec run (the command beneath `--` can be replaced with another model's direct Python command):

```bash
cd /workspace/P78123011/MediaTek_2026_project/ver5/overhead
python profile_gpu1.py --name BERT4Rec --output gpu1_runs/bert4rec_full_beauty -- \
  /dataspace/P78123011/miniconda3/envs/py31014/bin/python \
  ../../Reproduce/BERT4Rec/train.py \
  --dataset amazon-all-beauty \
  --cache-dir /workspace/P78123011/cache \
  --output-dir gpu1_runs/bert4rec_full_beauty/model_output \
  --device cuda
```

Use one process at a time on GPU 1. For a fair comparison, use the same dataset and evaluation catalog, record each phase separately (item-text encoding, graph/teacher preparation, training, and full-catalog inference), and report the maximum observed GPU memory across phases. Also report batch size, GPU name, software versions, warm/cold cache status, train users or steps per second, and test users per second. Run the full commands under identical hardware rather than comparing the archived throughput numbers directly. NVML polling is a sampled process-memory measure; a short-lived allocation can fall between samples. An in-process `torch.cuda.max_memory_reserved()` measurement is preferable when instrumenting the training loop.

The ordinary sandbox has no `/dev/nvidia*` devices. GPU 1 was measured in the GPU-enabled execution context, with the selected physical GPU bound by UUID. A full-length benchmark still requires matched run settings, the official SIGMA CUDA backend, and LLM-SRec's pretrained SASRec teacher.
