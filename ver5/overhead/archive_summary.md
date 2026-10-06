# Existing Full Beauty run records

These values come from archived logs, not a controlled GPU 1 benchmark. Blank GPU peak means no measurement was recorded. Different batch sizes, implementations, and hardware make throughput values descriptive only.

| Model | Trainable parameters logged | Log span (s) | Test users/s | GPU 1 peak (MiB) |
|---|---:|---:|---:|---:|
| Ours (ver5) | 886888 | 319.0 | 175.3 | — |
| SIGMA | — | — | — | — |
| Mamba4Rec | 334144 | 4.0 | 13,886.5 | — |
| EAGER | — | 988.0 | 615.2 | — |
| BERT4Rec | 954362 | 11.0 | 20,796.4 | — |
| ReSID | 3031680 | 89.0 | 657.6 | — |
| LLM-SRec | — | — | — | — |

`archive_summary.csv` contains the exact values, source run directories, and caveats.
