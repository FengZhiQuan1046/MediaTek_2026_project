# LLaRA reproduction

此目錄使用 [LLaRA 上游原始碼](../../../LLaRA) 的 `MInterface`、LoRA、MLP projector
與 SASRec 類別，並沿用本專案 `ver4` 的 Amazon Reviews 2023 資料載入、一次性
user/item 5-core 過濾、時間排序及最後兩筆 validation/test 切分。訓練只使用 training
partition；validation/test 對每位使用者抽樣候選商品，保留生成式 HR@1，並用模型對候選名稱的
平均 token log probability 排名，計算 NDCG@5/10 與 Recall@5/10。這些是
sampled-candidate 指標，不能與 SASRec/BERT4Rec 的 full-catalog 指標直接比較。
預設 `CANS_NUM=10` 且每位使用者只有一個正例，因此 Recall@10 必然為 1.0；
若要讓 Recall@10 有區分度，需增加 `CANS_NUM`，同時留意較長 prompt 的 GPU 記憶體。

先在本目錄安裝依賴：

```bash
cd /workspace/P78123011/MediaTek_2026_project/Reproduce/LLaRA
/workspace/P78123011/miniconda3/envs/py31014/bin/python -m pip install -r requirements.txt
```

LLaRA 使用 Llama-2-7B-hf。該權重須可透過 Hugging Face 帳號取得，或將
`LLM_PATH` 指向已下載的本地模型目錄。腳本會優先使用共享
`CACHE_DIR/models--meta-llama--Llama-2-7b-hf` 或
`CACHE_DIR/hub/models--meta-llama--Llama-2-7b-hf` 的 snapshot，預設
`CACHE_DIR=/workspace/P78123011/cache`。執行前會檢查 Llama 設定是否可讀，
以免在推薦器預訓練完成後才發現模型不可用。上游原始碼預設位於
`/workspace/P78123011/LLaRA`；需要時可設 `LLARA_UPSTREAM_DIR`。

從任意目錄執行單一 subset：

```bash
SUBSETS=Toys_and_Games EPOCHS=5 GPU_IDS=0,1 \
  bash /workspace/P78123011/MediaTek_2026_project/Reproduce/LLaRA/run.sh
```

也支援位置參數 `bash run.sh 0,1`。兩卡時由 PyTorch Lightning DDP 執行，
`BATCH_SIZE` 是每張 GPU 的 microbatch size，`ACCUMULATE_GRAD_BATCHES`
預設為 16。各 subset 依序執行。`SUBSETS=all` 包含與 SASRec/BERT4Rec 相同
的八個 Amazon subsets；可用逗號選擇其中部分。`REPEATS` 控制重複次數。

常用可覆寫的環境變數：

| 變數 | 預設 | 意義 |
| --- | --- | --- |
| `GPU_IDS` | `1` | 使用的實體 GPU |
| `EPOCHS` (`MAX_EPOCHS`) | `5` | LLaRA 最大 epoch；`EPOCHS` 優先 |
| `BATCH_SIZE` | `4` | 每卡 microbatch |
| `ACCUMULATE_GRAD_BATCHES` | `16` | 梯度累積次數 |
| `LR` | `8e-4` | LLaRA learning rate |
| `EARLY_STOPPING_PATIENCE` | `10` | Early stopping patience |
| `MAXLEN` / `CANS_NUM` | `10` / `10` | 歷史長度／候選數 |
| `REC_EPOCHS` / `REC_BATCH_SIZE` | `10` / `128` | 上游 SASRec 預訓練設定 |
| `MAX_TRAIN_SAMPLES` / `EVAL_USER_LIMIT` | `10000` / `1000` | 資料量上限；0 表示全部 |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `CACHE_DIR` / `OUTPUT_ROOT` | workspace `cache` / 本目錄 `outputs` | 快取／實驗輸出位置 |
| `PYTHON_BIN` / `LLM_PATH` | workspace Python / Llama-2-7B-hf | 執行環境／模型位置 |

正式完整資料評估請設定 `MAX_TRAIN_SAMPLES=0 EVAL_USER_LIMIT=0`。每個 run
產生 `outputs/<subset>/llara_<timestamp>_r<repeat>/train_*.log`、
`config.json` 和 `metrics.json`。`metrics.json` 內含候選式 validation/test
HR@1、有效生成比例、候選式 NDCG@5/10、Recall@5/10 和資料統計。
`tqdm` 進度條只顯示在終端，不寫入 `train_*.log`。推薦器權重依資料切分與設定存於共享
`cache/llara`，實驗輸出目錄不儲存模型 checkpoint。

驗證啟動與資料 adapter：

```bash
bash -n run.sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python train.py --help
```
