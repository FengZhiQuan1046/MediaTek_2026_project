# BERT4Rec reproduction

此 baseline 是依 [BERT4Rec 論文](https://arxiv.org/abs/1904.06690) 方法撰寫的
PyTorch 實作，使用雙向 Transformer、可學習位置 embedding、GELU prediction head
與共享 item embedding 的 full-softmax Cloze loss。
[作者原始實作](https://github.com/FeiSun/BERT4Rec) 使用 TensorFlow；此處採用本專案的
資料與評估協定，不宣稱重現論文原始資料集的數值。

## 執行與 GPU

使用其他 baseline 的 Python 環境即可；如需安裝依賴，在本目錄執行：

```bash
python -m pip install -r requirements.txt
```

可從任何工作目錄啟動，預設 GPU 0、100 epochs，依序執行八個 Amazon subsets：

```bash
bash /workspace/P78123011/MediaTek_2026_project/Reproduce/BERT4Rec/run.sh 0
```

指定 GPU、資料集與 epoch：

```bash
GPU_IDS=0,1 SUBSETS=Full_Beauty,Toys_and_Games EPOCHS=50 \
BATCH_SIZE=128 EVAL_BATCH_SIZE=32 \
bash /workspace/P78123011/MediaTek_2026_project/Reproduce/BERT4Rec/run.sh
```

也可用 `bash run.sh 0,1`；位置參數優先於 `GPU_IDS`。多卡使用 PyTorch
`DataParallel` 共同訓練同一個 subset，`BATCH_SIZE` 是所有卡合計的 batch size。
每張卡保留完整模型，不會將模型參數分片。各 subset 依序執行。
`PYTHON_BIN` 可指定 Python；預設優先使用 workspace 的
`miniconda3/envs/py31014/bin/python`，不存在時使用 PATH 的 `python3`。

## 輸入與資料協定

直接使用 `ver4/src/data.py`、`ver4/src/data_mamba_rl.py`：Amazon Reviews 2023
載入、一次性 user/item frequency >= 5 過濾（非 iterative k-core）、ID mapping、
時間排序、最後兩筆 validation/test、共享 interaction cache 均與現有 baseline 一致。
過濾後少於三筆的 user 僅供 training。訓練只使用 `train_by_user`。

`SUBSETS=all` 支援八個名稱：`Full_Beauty`、`Beauty_and_Personal_Care`、
`Baby_Products`、`Sports_and_Outdoors`、`Books`、`Toys_and_Games`、
`Video_Games`、`Clothing_Shoes_and_Jewelry`。用逗號選擇部分 subsets。

`CACHE_DIR` 預設為 workspace 的 `cache`，沿用
`mamba_multi_agent_data/<dataset>_<hash>.pkl`，不需先轉成 BERT 格式。
也可設定 `DATASET` 使用共用 loader 的單一 dataset（此時忽略 `SUBSETS`）：

```bash
DATASET=movielens-1m DATA_PATH=/path/to/ml-1m EPOCHS=50 bash run.sh 0
```

`DATA_PATH` 用於共用 loader 支援的 MovieLens/Yelp 本地輸入；Amazon 仍透過
Hugging Face 及 `CACHE_DIR` 載入，並不將 `DATA_PATH` 當作任意 Amazon CSV。
`DATASET=synthetic` 可在無下載情況下測試。

## 超參數

皆可透過環境變數覆寫，也可直接執行 `python train.py --help` 查看 CLI。

| 環境變數 | 預設 | 說明 |
| --- | --- | --- |
| `EPOCHS` | `100` | 最大 epoch 數 |
| `BATCH_SIZE` / `EVAL_BATCH_SIZE` | `128` / `64` | 全域訓練／評估 batch |
| `LEARNING_RATE` / `WEIGHT_DECAY` | `1e-3` / `0.0` | Adam 設定 |
| `MAXLEN` | `100` | 最多保留的歷史 item 數，另留一格推論 MASK |
| `HIDDEN_UNITS` | `128` | 隱藏維度 |
| `NUM_BLOCKS` / `NUM_HEADS` | `2` / `2` | Transformer 層数／attention heads |
| `DROPOUT_RATE` | `0.2` | Dropout |
| `MASK_PROB` | `0.2` | 每個訓練 item 被選為預測目標的機率 |
| `LOSS_CHUNK_SIZE` | `128` | 每次 full-softmax 計算的 masked positions 數 |
| `EARLY_STOPPING_PATIENCE` | `20` | validation 無改善的 epoch 數；0 關閉 |
| `MONITOR_METRIC` | `ndcg@10` | 可改 `recall@10` |
| `USE_AMP` | `True` | CUDA 混合精度；`False` 關閉 |
| `SEED` / `REPEATS` | `25252` / `1` | 隨機種子／重複執行次數（固定同一 seed） |
| `MAX_EVENTS` | 空值 | 原始資料量上限；正式實驗保持空值 |
| `MAX_BATCHES_PER_EPOCH` | `0` | 每 epoch batch 上限；0 表示完整遍歷 |
| `MIN_RATING` | `4.0` | 共用 loader 的 MovieLens/Yelp 評分門檻 |
| `DEVICE` | `cuda` | 可設 `cpu` 做 smoke test |

每個 epoch 隨機遍歷非空訓練 user 一次，只取最後 `MAXLEN` 個 items，動態重新遮罩。
被選中的位置以 80% 機率改為 MASK、10% 隨機 item、10% 保持原樣；每條序列至少
選一個目標。只有被選位置計算 CE，padding 不計算 loss。多卡 loss 按有效目標數
加權。這裡不使用作者程式預先產生的固定多份遮罩資料，因此 epoch 次數不可直接
與原 TensorFlow 設定等同。

訓練 full-softmax 按 `LOSS_CHUNK_SIZE` 分塊並在 backward 重算該部分，以降低
大型 catalog 的 logits 顯存需求。OOM 時可降低此值及 `BATCH_SIZE`；評估 OOM
則降低 `EVAL_BATCH_SIZE`。

## 評估與輸出

validation 使用 train history；test 使用 train + validation history，尾端附加
MASK 預測下一個 item。比照 SASRec，進行 full-catalog 排序，遮蔽截斷後歷史中的
已見 item，保留重複出現的 gold target，並以 `>= gold score` 作保守 tie ranking。
輸出 Recall/Hit/NDCG@5、@10 與吞吐量。依 validation 選最佳 epoch，再跑 test。

`OUTPUT_ROOT` 預設為本目錄的 `outputs`。每次結果位於：

```text
outputs/<subset>/bert4rec_<timestamp>_r<repeat>/
  train_<timestamp>.log
  config.json
  metrics.json
```

`metrics.json` 包含最佳 epoch、validation/test metrics、資料統計與逐 epoch
history。比照 SASRec，最佳權重只暫存在記憶體，不輸出模型 checkpoint。

## 驗證

```bash
bash -n run.sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
DEVICE=cpu DATASET=synthetic EPOCHS=2 MAX_BATCHES_PER_EPOCH=2 \
  MAXLEN=5 HIDDEN_UNITS=8 NUM_BLOCKS=1 NUM_HEADS=2 BATCH_SIZE=4 \
  CACHE_DIR=/tmp/bert4rec-cache OUTPUT_ROOT=/tmp/bert4rec-smoke bash run.sh
```

測試包含遮罩目標與 padding、資料洩漏、雙向 attention、loss 分塊的數值與梯度、
已見 item／ties 評估，以及有兩張可見 CUDA 卡時的 DataParallel 梯度一致性。
