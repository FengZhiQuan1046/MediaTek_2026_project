# 歷史長短分組評估

`run_long_short.sh` 會以同一個 short + preference 模型訓練全部使用者，並額外依篩選後、leave-two-out 前的完整序列長度分組報告 full-catalog 指標。預設門檻為 10：短組 ≤ 10、長組 > 10。這裡的「長組」是**使用者分組名稱**，不是 long agent；ver5 沒有 long agent。分組不改動訓練樣本，兩組共用模型，早停仍依整體 validation 指標決定。

```bash
PYTHON_BIN=/dataspace/P78123011/miniconda3/envs/py31014/bin/python bash run_long_short.sh
EVAL_LENGTH_THRESHOLD=20 GPU_IDS=0 PYTHON_BIN=/dataspace/P78123011/miniconda3/envs/py31014/bin/python bash run_long_short.sh
```

## Agent 與輸入範圍

- Short：只讀最近 `min(MAX_HISTORY, SHORT_WINDOW)` 筆。
- Preference：只要啟用，就讀完整的可用歷史，最多 `MAX_HISTORY` 筆，和 `SHORT_WINDOW` 無關。它使用獨立的 Mamba 式選擇性狀態更新與 LoRA，不使用 GRU。
- Coordinator：只有 short 與 preference 同時啟用、且 `USE_COORDINATOR=1` 時才建立。若關閉但兩個 agent 都啟用，使用固定的 `PREFERENCE_SCORE_WEIGHT` 混合兩個分數。
- LightGCN：可獨立啟用，使用訓練互動圖建立商品向量。

`run_mamba_rl.sh` 的預設 `MAX_HISTORY=100`、`SHORT_WINDOW=10`；此分組腳本可覆寫 `SHORT_WINDOW`。訓練、hard-negative 打分、validation/test 與直接呼叫模型都使用相同的 agent 輸入規則。停用 preference 後，short-only 模型不需處理較早的輸入。若兩個序列 agent 都停用，必須啟用 LightGCN，模型改用純圖分數。

## Ablation 與輸出

兩個 suite 腳本接受 `USE_SHORT`、`USE_PREFERENCE`、`USE_GCN`、`USE_COORDINATOR`，值為 0 或 1。輸出標記為 `S{USE_SHORT}_P{USE_PREFERENCE}_G{USE_GCN}_C{有效 COORDINATOR}`，分組腳本再加入短期窗口 `SW{SHORT_WINDOW}`。關閉的模組不參與前向計算，也不進入 optimizer；只有一個序列 agent 時 coordinator 自動停用。`AGENT_HISTORY` log 和每次結果的 `short_history_limit`、`preference_history_limit` 記錄有效輸入長度。

設定 `JOINT_EPOCH=0` 時，只跑 `SPECIALISTS_EPOCH` 次**單階段 DL**：所有啟用的 agent、coordinator、共用投影與 LightGCN 從第一步一起更新，不計算 RL loss。`JOINT_EPOCH>0` 時，維持 specialists DL 預訓練後接 joint DL＋RL；其中 RL 只用訓練序列後續商品計算離線代理 reward。`RL_COEF=0` 可讓兩階段模式的 joint 也只用 DL。可用 `JOINT_EPOCH=0 USE_SHORT=1 USE_PREFERENCE=1 USE_GCN=1 USE_COORDINATOR=1 bash run_long_short.sh` 啟用單階段全模組 DL。

若套件在 Baby Products 後中斷，可設 `START_FROM=Sports_and_Outdoors`，從 Sports and Outdoors 繼續，之後仍會跑 Toys and Games；已完成的 Full Beauty 與 Baby Products 不會重跑。每個資料集啟動前都會檢查子腳本的 shell 語法。

分組結果保存在 `metrics.json` 的 `valid.sequence_length` 與 `test.sequence_length`，包含兩組人數、歷史長度及 Recall、NDCG、Hit 指標。空組的指標為 JSON `null`。定期評估另寫入兩組分數檔；它們的「short／long」仍僅指使用者歷史長度，不指 agent。

從 ver4 複製的舊輸出與表格反映舊架構，不能作為 ver5 的新結果；需要重新訓練與評估。
