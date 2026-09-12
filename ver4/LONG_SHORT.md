# 長／短序列分組評估

在 ver4 執行：

```bash
bash run_long_short.sh
# 調整門檻及 GPU
EVAL_LENGTH_THRESHOLD=20 GPU_IDS=0 bash run_long_short.sh
```

新腳本沿用目前 `run_amazons_full_rl.sh` 的模型、三階段訓練超參數與啟用測資（Sports_and_Outdoors、Toys_and_Games）。其他 Amazon2023 類別可在新腳本底部取消對應 `run_subset` 的註解。這份設定是獨立副本，後續原腳本的修改不會自動同步。

全部使用者一起訓練；每次 validation、periodic test、final test 使用同一模型與全商品排名，另外按使用者歷史分組統計。早停與最佳模型仍依整體 validation 指標決定。

- 短序列：原有 user/item 至少 5 筆的單次篩選後、leave-two-out 切分前的完整序列長度 ≤ 門檻（預設 10）。
- 長序列：篩選後完整序列長度 > 門檻。
- 完整長度以訓練歷史長度加上 valid/test 位置數重建，分組不讀取 target 的 item 身分，同一使用者在兩個 split 維持同組。模型輸入仍遵守既有 `MAX_HISTORY`，test 仍加入 validation item。
- 此實驗比較同一模型對不同歷史長度使用者的表現，不是分組重新訓練，也不能單憑指標差異推論序列長度的因果效果。

預設結果在 `outputs_mamba_rl/amazons_long_short_L1_S1_P1_G1/<subset>/<run>/`，每次執行建立獨立目錄。

`metrics.json` 的 `valid.sequence_length`、`test.sequence_length`，以及 `<subset>_scores.json` 的 `sequence_length.valid/test`，都有 `short`、`long`：

- `total_users`、`evaluated_users`：組內總人數／實際評估人數。
- `history_length_min/max/mean`：該 split 組內全體使用者的分組基準序列長度統計（欄位名保留 history_length）。
- `recall@5/10`、`ndcg@5/10`、`hit@5/10`：組內平均。空組或未評估組以 JSON `null` 表示，避免誤認為零分。

訓練 log 的 `SEQUENCE_LENGTH` 行記錄每次分組結果。可比較兩組 `ndcg@10`、`recall@10`，同時查看人數，避免把樣本極少的組當成穩定結論。

既有入口預設不啟用此功能。若直接使用 `run_mamba_rl.sh`，可追加 `--eval-length-threshold 10 --eval-length-basis filtered_full_sequence`。新腳本也會將額外 CLI 參數傳給訓練程式。

每次執行另存 `dataset_config.json`（該 dataset 的完整實際參數）、`short_scores.json` 與 `long_scores.json`。兩份分組分數各自包含 validation/test 的六個分數及人數；log 的 `EXPERIMENT_CONFIG` 記錄完整設定，`FINAL_LENGTH_SCORES` 記錄各組最終六個分數。

## Agent ablation

`run_amazons_full_rl.sh` 與 `run_long_short.sh` 都提供 `USE_LONG`、`USE_SHORT`、`USE_PREFERENCE`、`USE_GCN`，預設為 1；改成 0 即停用。可直接改腳本內預設值，或用環境變數：

```bash
USE_SHORT=0 USE_GCN=0 bash run_long_short.sh
USE_PREFERENCE=0 bash run_amazons_full_rl.sh
```

輸出根目錄追加 `L{USE_LONG}_S{USE_SHORT}_P{USE_PREFERENCE}_G{USE_GCN}`，例如 `outputs_mamba_rl/amazons_long_short_L1_S0_P1_G0/<subset>/<run>/`；一般 Amazon suite 則是 `amazons_lora_L1_S1_P0_G1`（full-rank 為 `amazons_full_...`）。log 的 `ABLATION`、完整 config 與 `metrics.json` 的 model.ablation 也會記錄設定。

停用 agent 會跳過 encoder/logits、凍結該 agent 參數，並移除其 ranking、preference auxiliary/contrastive 或 diversity 配對 loss；LONG/SHORT 的混合權重只在啟用的 agent 間正規化。模組仍保留以維持 checkpoint 的參數結構相容。GCN 停用時不建立或執行圖模型。LONG、SHORT、PREFERENCE 全關閉時必須啟用 GCN；只啟用一個 agent 時不建立 coordinator。停用 preference 的相關診斷值以 0 表示，不應視為已學得的偏好統計。

## 定期 test 的兩份 JSON

`run_long_short.sh` 預設使用 `filtered_full_sequence`，分界為 10：短組 ≤ 10、長組 > 10。每次定期 test 完成便更新 `short_scores.json` 和 `long_scores.json`，各自的 `test` 包含最新六個分數；`periodic_test_history` 保存每次 stage、epoch、global_step、人數及六個分數。訓練結束後 `status` 由 `training` 變為 `final_best_model`，`test` 變成最佳 validation 模型的最終 test 分數，歷史紀錄仍保留。尚未完成第一次 test 前不會有分數檔。

前處理維持原本的 `MIN_INTERACTIONS=5` 單次 user/item 篩選，沒有改為 iterative 5-core；因此低頻 item 移除後，個別 user 的剩餘序列可能少於 5 筆。這與既有實驗一致。訓練資料與模型輸入不因長短分組改變。

## USE_LONG=0 的歷史範圍

`USE_LONG=0` 除了停用 long agent 及其 loss，也會讓 preference 與其他序列 agent 只接收 `min(MAX_HISTORY, SHORT_WINDOW)` 筆近期歷史。此限制同時適用於訓練、hard-negative 模型打分、validation/test，以及直接呼叫模型。log 的 `AGENT_HISTORY` 記錄有效範圍。測試長短組仍依篩選後完整序列分組。

這是序列 agent 的輸入限制；GCN 仍由 `USE_GCN` 獨立控制，啟用時仍使用訓練互動圖。既有已互動商品排除、catalog priors、訓練樣本與目標生成維持原本規則。

## 其他開關的訓練連動

- `USE_SHORT=0`：不執行 short encoder/logits，不訓練 short 參數，不計 short ranking loss 或涉及 short 的 diversity loss；若 LONG 啟用，其完整歷史訓練仍保留。
- `USE_PREFERENCE=0`：不執行 preference encoder/logits/context gate，不訓練相關參數，移除 preference ranking、auxiliary、contrastive 以及相關 diversity loss。
- `USE_GCN=0`：不建立 graph edges、LightGCN 或圖參數，不進行圖傳播和圖參數訓練；商品文字投影仍可訓練。
- 使用 coordinator 且 LONG/SHORT 未同時啟用時，使用固定分支權重，不執行或訓練 mix adapter。每個 stage 的 `TRAINING_BRANCHES` log 記錄實際啟用分支及可訓練參數量。

## 單 agent 與純 LightGCN

三個序列 agent 僅啟用一個時，不建立 coordinator，直接用該 agent logits 排序；coordinator 訓練階段跳過，specialists 與 joint 階段只訓練剩餘 agent（及啟用的共享投影／GCN），ranking loss 不重複計算。PREFERENCE 本身是既有的 GRU preference 模型，不是 Mamba。

三個全關且 `USE_GCN=1` 時，以 LightGCN 傳播後的 user embedding 與 item embedding 內積打分。只訓練圖參數，不使用文字 Mamba 編碼、序列 encoder、coordinator 或 popularity/transition 排序先驗；coordinator 階段跳過。四個全關會報錯。輸出分組與六個分數的儲存方式不變。`PREDICTION_PATH` 記錄實際模型路徑。
