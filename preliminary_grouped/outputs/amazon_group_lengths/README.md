# Amazon preliminary: temporal group lengths

One JSON per subset, using the original preliminary test cohorts and 100-event history cap.
Groups split only when the gap between consecutive events is strictly greater than 30 days; exactly 30 days stays together. Duplicate events remain. A group may span more than 30 days.

Two distinct lengths are reported: interactions per group and elapsed days from its first to last event (singletons have zero-day span). The overall mean is weighted equally per group. User-weighted means and each group index counted backward from the newest group are included separately.

Each JSON includes the original figure 1/2 all-available and fixed-K16 cohorts, the figure 3 influence cohort, and the figure 4 coverage cohort. A full, untruncated observed-history reference is separate. These temporal groups are not the semantic preference clusters in the original figure 2.

| Subset | Mean interactions/group | Mean span (days) |
|---|---:|---:|
| [All_Beauty](All_Beauty.json) | 1.9148 | 7.7554 |
| [Baby_Products](Baby_Products.json) | 2.1573 | 3.4740 |
| [Sports_and_Outdoors](Sports_and_Outdoors.json) | 1.8041 | 2.5875 |
| [Toys_and_Games](Toys_and_Games.json) | 2.1447 | 3.1334 |

Timestamp recovery used only the existing local raw Arrow cache. Single-pass user/item thresholds, sorted user-ID mapping and full sampled item sequences were checked against the exact original preliminary InteractionData cache (SHA-256 recorded in each JSON). Figure 3 lag/item/target rows and figure 4 history lengths were also checked. No raw data or original plot was changed; no model was trained.

Reproduce from the repository root:

```bash
/workspace/P78123011/miniconda3/envs/py31014/bin/python preliminary_grouped/amazon_group_lengths.py
```
