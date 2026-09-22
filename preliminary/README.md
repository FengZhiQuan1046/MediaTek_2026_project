# ver4 preliminary experiment

Run `bash run.sh` here. Results and PDF figures appear in `results/`. This
analysis does not train or save any recommender weights or checkpoints, and it
does not modify ver4. It uses the exact cached ver4 chronological leave-two-out
split. The default 2,000 users per dataset is a **pilot**; set `MAX_USERS=0` for
all eligible users. Set `DATASETS=amazon-sports-and-outdoors` for a quick
single-category run. Datasets whose *exact* ver4 split cache is missing are
skipped and listed in the manifest; a legacy cache with a different digest is
never silently substituted. `bash run.sh --help` lists the Python options.

The independent semantic representation is a deterministic 1,024-dimensional
word/bigram hashing vector of ver4's `item_texts`. It is not the learned ver4
item embedding and does not require a model download. Those texts can contain
review fallback when metadata was missing; the cached split does not preserve
provenance, so the analysis is **not** a strict metadata-only test. Popularity
matching and K-means preference-proxy fitting use training items only. The
cached split does not retain timestamps, so distance means *number of
interactions*, not days.

For user u, target y and lag k (k=1 is most recent), the primary statistic is
`excess_cosine(u,k) = cosine(text(history[-k]),text(y)) - mean_B cosine(text(matched_random),text(y))`.
Random items are sampled from the same dataset and training-popularity quintile;
unavailable bins are widened, with fallbacks counted. Curves report both an
all-available cohort and fixed cohorts with at least K=8,16,32 observed items.
`paired_decay.csv` reports, among the same history>=16 users, lag-1 excess
alignment minus the within-user mean for lags 9-16. Intervals are user
bootstrap 95% CIs. A CI crossing zero is **not** evidence
that old items are equivalent to random items. No cutoff is selected from test.

The second statistic is
`excess_cluster_match(u,k) = 1[cluster(history[-k])=cluster(y)]
 - mean_B 1[cluster(random)=cluster(y)]`, where K-means is fitted on training
items only. It tests whether a coarse preference *proxy* persists when
item-level affinity weakens. For each user with old history, the same held-out
target and popularity-matched negatives are evaluated with two old-only
readouts: `score_item(c)=max_old cosine(text(i),text(c))` and
`score_preference(c)=fraction_old[cluster(i)=cluster(c)]`. Their pairwise
accuracy is `mean_negative(1[score(y)>score(negative)] +
0.5*1[score(y)=score(negative)])`; their paired difference is reported.
These are diagnostic encodings, **not** the learned ver4 preference agent.
Separately, let `margin(H)=cosine(normalized_mean(text(H)),text(y)) -
mean_negative cosine(normalized_mean(text(H)),text(negative))`.
`margin(full)-margin(last_10)` is a model-free negative-transfer *proxy*,
not a causal estimate of Mamba harm. The reported fraction of old
low-alignment positions is `sum_{k>10} 1[excess_cosine(u,k)<=0] / T` with
T the capped, observed history length; this target-defined fraction is only
post-hoc descriptive and is **not** a noise rate.

The report deliberately does not assert that Mamba cannot perform multiple
functions. A direct Mamba+LoRA state/deletion experiment would require trained
weights, which this task explicitly forbids saving. The high-throughput setting
is therefore motivation and architecture context, not an experimentally proven
deficiency. Raw CSVs, manifest and figures are retained; no `.pt`, `.pth`,
checkpoint or fitted vectorizer is written.
