# Aggregated preliminary figures

四個 subset 的最新完整分析結果，以一致的配色、線型、標記與字體重畫。
所有曲線直接取自已存在的 CSV；不合併不同 subset 的使用者、不重新估計統計量，也不平滑曲線。

## 圖檔

每張圖都有向量 PDF、SVG，以及 400 dpi PNG。SVG 字體轉為路徑，避免投稿系統替換字型。

| 檔名（不含副檔名） | 內容 |
| --- | --- |
| `figure0_abcd_row` | 四面板橫排跨欄圖，寬約 8.8 吋，四個面板的繪圖區等寬；依序為語意、模型影響、覆蓋率、偏好同群，(c) 與 (d) 間有加深的分隔線 |
| `figure0_acd_row` | 論文跨欄圖，原 (a)、(c)、(d) 三面板橫排，寬約 7.15 吋 |
| `figure0_b_preference` | 原 (b) 偏好同群相對個人歷史機會基準的獨立圖 |
| `figure1_alignment` | 語意對齊；available-history 與 history ≥ 16 兩種 cohort |
| `figure2_preference` | 偏好同群相對個人歷史機會基準；上述兩種 cohort |
| `figure3_influence` | 移除歷史 item 的模型 margin 變化，僅顯示 n ≥ 100 的位置 |
| `figure4_coverage` | 最近 X 個 item 的累積正向語意訊號覆蓋率，完整 1–100 範圍 |
| `figureS1_influence_all_positions` | 補充圖：包含低樣本數的所有位置，保留原圖範圍 |

## 重畫

使用專案已有的 Python 環境，從任意工作目錄執行：

```bash
/workspace/P78123011/miniconda3/envs/py31014/bin/python \
  /workspace/P78123011/MediaTek_2026_project/preliminary/aggregated/plot_aggregated.py
```

相依套件只有 `matplotlib` 和 `numpy`。最終投稿請提供已授權的 Times New Roman
字型檔，並加入 `--font-path /path/to/times.ttf`。程式會檢查檔案的字型家族確為
Times New Roman，再將字型嵌入 PDF；若本機沒有該字型，程式會顯示警告並使用
Liberation Serif 產生**預覽版**。目前本機尚無 Times New Roman，因此現有圖檔是
Liberation Serif 預覽版，尚未滿足指定字型。`sources.json` 記錄實際使用的字型。

預設自動選取每個 subset 最新、狀態為
`complete` 且具有三份必要 summary CSV 的 run，並確認分析方法與主要設定相同。
不需要 GPU，不會啟動模型訓練。

若要固定此次來源，於本目錄執行：

```bash
/workspace/P78123011/miniconda3/envs/py31014/bin/python plot_aggregated.py \
  --sources sources.json
```

可指定 `--analysis-root PATH`、`--output-dir PATH`、`--formats pdf png svg`、
`--dpi 400`、`--min-users 100`。`sources.json` 記錄來源 run、設定、CSV SHA-256、
套件版本與輸出設定。`plotted_data.csv` 保留三份 summary 的所有 test rows，
**包含主圖未顯示的 cohort、metric 與低樣本數點**，供核對數值。

## 統計與閱讀方式

- 所有圖使用 test split；兩張跨欄圖的 (a)、(b) 與獨立圖 (b) 使用 available-history cohort。
- 兩張跨欄圖的 (d) 只呈現最近 1–20 筆，以讓快速上升區域可辨；`figure4_coverage` 保留 1–100 的完整範圍。
- Alignment 和 preference 的 x 座標是原始 lag bin 的中點，採線性軸；標籤顯示 bin 範圍。
- Preference 圖使用 `history_normalized_cluster_match`。對每位使用者，令 K 為測試當下最多 100 筆可用歷史 item 涵蓋的不同偏好群數；若下一個 item 的群也在歷史中，均勻抽一個「群」的基準為 `1/K`，否則為 0。每個位置畫 `1[歷史 item 與下一 item 同群] − 該使用者基準`，再依 lag bin 對每位使用者先取平均，最後對使用者取平均並 bootstrap。零線代表與該個人歷史機會基準相同；y 軸以百分點顯示。此基準假設各個觀察到的群等機率，並非依每群 item 數量抽樣的基準。
- 跨欄圖的 (a–c) 和個別圖 1–3 只顯示至少 100 位使用者的點。不同 lag 的可用使用者可能不同。
- 圖 1–3 的陰影是原始的 95% user-bootstrap CI。圖 4 的陰影是使用者第 25–75 百分位範圍，**不是 CI**。
- Baby Products 的 history ≥ 16 cohort 未達 100 人，因此圖 1、2 的右側面板沒有它的曲線，圖內已明示。
- 模型影響力定義是 `margin(full history) - margin(history without this item)`，
  正值表示保留該 item 有助於該次 next-item margin。使用原實驗的 64 個固定抽樣負例，
  不代表 full-catalog ranking 改善。不同 subset 模型的 margin 尺度也未額外校正。
- 補充圖包含低至 1 人的位置；原 CSV 若沒有可估計的 CI，僅保留均值而不填造區間。
- Coverage 是每位使用者的正向 excess cosine 訊號比例，再取使用者平均；分母是
  最多 100 個位置的 observed history。沒有正向訊號者被排除。它不是推薦準確率，
  短歷史亦會提早達到 100%。
- 此次來源為 pilot 實驗，最多抽樣 2,000 位使用者／subset；各分析的實際 n 可能更低。
  圖中的 lag 是互動位置而非時間；文字來源可能含 review fallback，沿用來源 manifest 的限制。

## Suggested paper captions

**Four-panel cross-column figure (`figure0_abcd_row`).** Preliminary diagnostics
across four Amazon subsets. (a) Next-item semantic alignment above
popularity-matched random controls. (b) Change in next-item margin when a
history item is retained versus deleted, using fixed sampled negatives.
(c) Cumulative coverage of positive semantic signal by the most recent items,
showing the first 20 positions. (d) Preference-cluster match above the
user-specific uniform-cluster chance level; for K distinct clusters in the
observed history, this chance level is 1/K when the target cluster is present
and zero otherwise. Panel (d) reports excess match in percentage points. The vertical rule
separates (c) and (d) visually.
Panel (d) is a model-free text-cluster diagnostic, so interpreting it as
post-method effectiveness requires a separate model-based result. Panels (a,b,d)
show means and 95% user-bootstrap confidence intervals for points with at
least 100 users; panel (c) shows means and the user interquartile range. All
results use the test split.

**Three-panel cross-column figure (`figure0_acd_row`).** Preliminary diagnostics
across four Amazon subsets. (a) Semantic alignment with the next item,
measured by within-subset normalized cosine similarity above popularity-matched
random controls. (c) Change in next-item margin when a history item is retained
versus deleted from the trained Mamba+LoRA+LightGCN model using fixed sampled
negatives. (d) Cumulative coverage of positive semantic signal by the most
recent items, averaged over users with nonzero positive signal; only the first
20 items are shown. Panels (a,c) show means and 95% user-bootstrap confidence
intervals for points with at least 100 users; panel (d) shows means and the
user interquartile range. Results use the test split of pilot analyses.

**Separate panel (`figure0_b_preference`).** (b) History-item preference-cluster
match above a user-specific uniform-cluster chance level. For each test user,
K is the number of distinct preference clusters among usable items in the
observed history (up to 100 interactions). The chance level is 1/K when the
next item's cluster occurs in that history and zero otherwise. Per-position
indicators minus this chance level are averaged within user and lag bin, then
across users. The horizontal line marks zero excess match. Shaded regions are
95% user-bootstrap confidence intervals; points require at least 100 users.
The chance level assumes uniform sampling over distinct clusters, not over
history items or popularity-matched catalog items.
