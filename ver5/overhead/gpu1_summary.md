# Physical GPU 1 pilot profiles

Full Beauty, one shortened run per available implementation. Memory is a 50 ms NVML process sample; time includes Python startup, cache access, training, and full evaluation. These are pilot overheads, not matched full-training requirements.

| Model | Status | Peak MiB | Wall seconds | Test users/s |
|---|---|---:|---:|---:|
| Ours (ver5) | optimized_throughput_prior_memory | 1578 | — | 2,842.3 |
| SIGMA | measured_pilot | 29638 | 5.71 | 16,638.8 |
| Mamba4Rec | measured_pilot | 2932 | 4.37 | 7,889.3 |
| EAGER | measured_pilot | 2436 | 59.63 | 79.3 |
| BERT4Rec | measured_pilot | 792 | 5.33 | 9,668.1 |
| ReSID | measured_pilot | 1420 | 12.22 | 4,382.4 |
| LLM-SRec | measured_pilot | 18366 | 35.47 | 29.8 |

Exact commands, source profiles and failure statuses are in `gpu1_summary.csv`.
