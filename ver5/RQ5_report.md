# RQ5：Short-term agent 不確定時，Preference-transition information 是否有幫助？

整理日期：2026-10-17。範圍為原版 `run_rq5.sh` 的 uncertainty 分組實驗，**不包含 `run_rq5_sim.sh` 的 SPC/SLC 相似度 cohort 實驗**。本報告只讀取已完成結果，沒有重新訓練或修改原始數據。

## 1. 研究問題與比較基準

研究假說：當 short-term head 的預測分布更不確定時，完整 RIBS 相對 short-only 的排名增益是否更大，以及這種增益是否依賴使用者特定的 preference transition probability `ρ`。

| 名稱 | 模型／處理 | 用途 | 不能單獨推論的事情 |
| --- | --- | --- | --- |
| Full RIBS（SPC） | Short + Preference-transition + Coordinator；共享 frozen Mamba features 與 LightGCN | 主要待測模型 | 完整模型增益不等於 transition probability 的因果貢獻 |
| Short-only（S） | 獨立重新訓練；移除 preference specialist 與 coordinator，保留語意特徵、GCN、priors | 整體架構控制組 | 同時改變容量、融合路徑與聯合訓練，不能隔離單一機制 |
| ρ-shuffled | 同一個 Full 最佳模型，在整個 subset 的測試使用者間打亂 ρ，重複 5 次 | 檢查 coordinator 是否依賴 ρ 與使用者的正確對應 | 未移除 preference state、預測分布或其餘 preference 資訊 |

本實驗沒有 SLC、外部推薦模型或「完全移除所有 transition 資訊」的控制組。ρ-shuffled 不需重新訓練，Full 與 Short-only 各自以 validation NDCG@10 選模；三種設定使用相同的測試使用者分組。

## 2. 資料、切分與模型設定

四個 Amazon Reviews 2023 subset 使用既有 ver5 cache。設定最低 rating 為 4.0。前處理先計算輸入事件的 user/item 次數，以最低 5 次門檻做一次過濾，再刪除空序列；**不是 iterative 5-core**，也不會對過濾後的 user 長度反覆套用門檻。依時間排序，最後一筆作 test、倒數第二筆作 validation，其餘作 train；存活互動不足 3 筆者只參與訓練，不參與這次測試。

Validation 的 observed history 是 train；test 的 observed history 是 train 加上 validation item。兩者都只保留**最近最多 100 筆**，不是最早 100 筆。Short specialist 使用其中最近 SW4；Preference specialist 使用整段截斷後歷史。Graph 與 popularity/transition priors 由 training interactions 建立。所有測試使用者均參與分析，沒有啟用 random-feature smoke test 或 analysis-user-limit。

| 設定 | 數值 |
| --- | --- |
| Text encoder | state-spaces/mamba-130m-hf，凍結並快取 item vectors |
| Short window / prototype count / max history | 4 / 32 / 100 |
| Ranker dimension / graph layers | 128 / 2 |
| LoRA rank / alpha / dropout | 16 / 32 / .05 |
| 最大訓練 epochs / joint epochs | 4 / 0；單階段 supervised objective |
| Learning rate / AdamW weight decay | 1e-4 / 1e-5 |
| Train / evaluation / text encoding batch | 128 / 64 / 32 |
| Text token limit / prototype temperature | 32（含 prompt）/ .2 |
| Future horizon / decay | 3 / .5 |
| Negative pool multiplier / hard fraction / warm-up | 12 / .75 / 2 epochs |
| In-batch negatives / full-catalog training | 開啟 / 關閉；訓練使用 sampled slate |
| Preference prediction / transition / balance / separation / sharpness coefficients | .2 / .1 / .01 / .01 / .05 |
| Preference contrastive / state diversity coefficients | .05 / .02 |
| 選模 / early-stopping patience / LR patience | Validation NDCG@10 / 6 evaluations / 2 validation intervals |
| Seed / shuffle repetitions / bootstrap repetitions | 25252 / 5 / 1000 |
| Checkpoint 檔案 | 不儲存；仍在記憶體中還原 validation 最佳 state |

`joint_epochs=0` 時 RL/policy-gradient objective 不啟用，即使設定中保留 `rl_coef=.1`、`rl_topk=10`。4 epochs 是設定上限；選模依 validation，不能假定最後一個 epoch 就是測試所用模型。Short-only 不計算已移除 preference 分支的 auxiliary losses。

| Subset | Test users | Train prefix cap | Train slate size | Validation interval（設定 steps） | Popularity α | Transition β |
| --- | --- | --- | --- | --- | --- | --- |
| All Beauty | 1,457 | 500,000 | 64 | 250 | -0.25 | 4.0 |
| Baby Products | 184,584 | 1,500,000 | 192 | 6,000 | 0.3 | 0.5 |
| Sports and Outdoors | 625,563 | 1,000,000 | 256 | 8,000 | 0.35 | 0.5 |
| Toys and Games | 564,666 | 1,500,000 | 192 | 6,000 | 0.3 | 0.5 |

訓練程式會把正的 validation interval 上限限制在當前 epoch 的 steps。Prefix cap 限制訓練 transition 數，不限制測試使用者數。

## 3. Short-term uncertainty 與 Low / Medium / High 的計算

取 **Full RIBS 自己的 short head** 的 full-catalog 分數，不取 preference entropy，也不使用 Short-only 控制組重新分組。對每個使用者，先遮蔽已觀察歷史中的 item，再對其餘候選計算 softmax。此步不加入 priors，也不查詢 held-out target ID。

令有效候選集合為 $A_u$、short 分數為 $z_S(u,i)$，則：

$$p_{u,i}=\frac{\exp z_S(u,i)}{\sum_{j\in A_u}\exp z_S(u,j)},\qquad U_u=\frac{-\sum_{i\in A_u}p_{u,i}\log p_{u,i}}{\log|A_u|}.$$

分析 softmax 的額外溫度固定為 1（short head 本身仍含模型學到的 score scale）。只有一個有效候選時 U=0；結果限制在 [0,1]。U 越高代表分布越均勻，是相對不確定性指標，**不是經校準的推薦正確機率**。

各 subset 使用其 validation 使用者 U 的 1/3、2/3 分位數作門檻，固定後套用 test：Low ≤ 下門檻；Medium > 下門檻且 ≤ 上門檻；High > 上門檻。同值保留在同一組；validation/test 分布不同，故 test 三組不必等大。

| Subset | Validation users | Low/Medium 門檻 | Medium/High 門檻 | Low test users | Medium test users | High test users |
| --- | --- | --- | --- | --- | --- | --- |
| All Beauty | 1,457 | .9997360110 | .9997531772 | 406 | 425 | 626 |
| Baby Products | 184,584 | .9876017372 | .9906597336 | 74,027 | 57,273 | 53,284 |
| Sports and Outdoors | 625,563 | .9876575470 | .9902969003 | 242,507 | 216,766 | 166,290 |
| Toys and Games | 564,666 | .9805169503 | .9851197004 | 209,555 | 189,093 | 166,018 |

## 4. Full-catalog ranking 與四項指標

對每位 test user 的唯一 held-out target 進行 full-catalog ranking。Full 與 ρ-shuffled 使用融合分數；Short-only 使用自己的 short 分數。三者均加上同一資料集的 training-only popularity/transition priors，再遮蔽已觀察歷史的 item，但若 target 曾出現，保留該 target 可被預測。歷史與遮蔽範圍沿用既有 evaluator 的最近 100 筆。

排名使用悲觀同分規則：

$$r_u=\#\{i:s(u,i)\ge s(u,y_u)\}.$$

每位使用者只有一個 relevant target，所以：

$$\mathrm{Recall@K}_u=\mathbf{1}[r_u\le K],\qquad\mathrm{NDCG@K}_u=\frac{\mathbf{1}[r_u\le K]}{\log_2(r_u+1)},\quad K\in\{5,10\}.$$

各組數字是組內使用者的算術平均；不是訓練 batch 平均。沒有將多個 subset 合併為同一個 macro/micro 分數。所有排名值、差值與 CI 都是原始比例單位，**不是百分比或 percentage points**。

## 5. ρ 打亂、Coordinator 權重與統計推論

先收集整個 test subset 每位使用者的原始 ρ，再以 seed=25753 產生 5 組全體使用者排列，排列在 batching 之前完成；不是 batch 內打亂，也不限定在同一 uncertainty 組。打亂後保留各使用者的 short state、preference state、predicted preference certainty 與候選表示，只替換 coordinator context 中的 ρ，再重新計算混合權重、coordinator state 與融合分數。

ρ-shuffled mean 先對**每位使用者的五次指標**取平均，再作組內平均；不是先平均 rank 再計算 NDCG，也不是平均五次分數後重新排名。

JSON 中的 `preference_weight` 是實際 softmax 混合權重 $w_P$，不是設定檔中的固定初始值 `.2`。同時記錄 $w_S$，兩者加總約為 1。Coordinator 的 residual state 路徑仍直接接收 preference state，因此 $w_P$ 很小**不代表 preference 所有路徑都沒有影響**，也不能把 w_P 當成完整因果貢獻。

對每個組別、每個指標先計算每位使用者的配對差值 $d_u=m_u^{Full}-m_u^{control}$，再以使用者為單位有放回抽樣 1000 次，使用 bootstrap mean 的 2.5% 與 97.5% 分位數形成 CI。程式以相同差值的出現次數及 multinomial 抽樣計算，等價於該平均數的 ordinary user bootstrap。ρ-shuffled 控制以使用者五次平均指標作比較。

High-minus-Low 是 High 組的 Full−control 增益減去 Low 組增益；CI 由兩組各自的 user-bootstrap mean 相減得到。**CI 包含 0（即使 0 恰好是端點）不記為排除零。** 這些區間沒有做多重比較校正，也不包含不同 training seed 或重新訓練變異。

## 6. 各 subset 分組結果

表格中 N@5/N@10 為 NDCG，R@5/R@10 為 Recall。分數保留六位小數；CI 表為「Full−控制組差值 [95% CI]」。差值由原始逐筆值計算，與已四捨五入分數相減可能略有不同。

### All Beauty

| Group | Users | Setting | N@5 | N@10 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Low | 406 | Full SPC | .160080 | .161669 | .165025 | .169951 |
| Low | 406 | Short-only | .156556 | .158045 | .160099 | .165025 |
| Low | 406 | ρ-shuffled mean | .160080 | .161669 | .165025 | .169951 |
| Medium | 425 | Full SPC | .122945 | .124410 | .136471 | .141176 |
| Medium | 425 | Short-only | .120181 | .120181 | .131765 | .131765 |
| Medium | 425 | ρ-shuffled mean | .122945 | .124410 | .136471 | .141176 |
| High | 626 | Full SPC | .065242 | .067910 | .070288 | .078275 |
| High | 626 | Short-only | .067318 | .068830 | .073482 | .078275 |
| High | 626 | ρ-shuffled mean | .065242 | .067910 | .070288 | .078275 |

| Group | Control | ΔN@5 [CI] | ΔN@10 [CI] | ΔR@5 [CI] | ΔR@10 [CI] |
| --- | --- | --- | --- | --- | --- |
| Low | Short-only | +.003524 [.000000, +.008279] | +.003624 [-.000363, +.008280] | +.004926 [.000000, +.012315] | +.004926 [-.004926, +.014778] |
| Low | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Medium | Short-only | +.002764 [.000000, +.006480] | +.004229 [+.000782, +.008619] | +.004706 [.000000, +.011765] | +.009412 [+.002353, +.021176] |
| Medium | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| High | Short-only | -.002076 [-.005349, +.000688] | -.000919 [-.004184, +.002098] | -.003195 [-.009585, +.003195] | .000000 [-.006390, +.007987] |
| High | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |

### Baby Products

| Group | Users | Setting | N@5 | N@10 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Low | 74,027 | Full SPC | .031990 | .036517 | .041674 | .055750 |
| Low | 74,027 | Short-only | .031567 | .036205 | .042620 | .056993 |
| Low | 74,027 | ρ-shuffled mean | .031990 | .036517 | .041674 | .055750 |
| Medium | 57,273 | Full SPC | .039277 | .042748 | .048051 | .058841 |
| Medium | 57,273 | Short-only | .036453 | .040483 | .046619 | .059190 |
| Medium | 57,273 | ρ-shuffled mean | .039277 | .042748 | .048051 | .058841 |
| High | 53,284 | Full SPC | .043167 | .046021 | .050053 | .058911 |
| High | 53,284 | Short-only | .042691 | .045851 | .050653 | .060450 |
| High | 53,284 | ρ-shuffled mean | .043167 | .046021 | .050053 | .058911 |

| Group | Control | ΔN@5 [CI] | ΔN@10 [CI] | ΔR@5 [CI] | ΔR@10 [CI] |
| --- | --- | --- | --- | --- | --- |
| Low | Short-only | +.000422 [-.000169, +.000980] | +.000312 [-.000258, +.000855] | -.000946 [-.001797, -.000094] | -.001243 [-.002243, -.000216] |
| Low | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Medium | Short-only | +.002823 [+.002078, +.003541] | +.002265 [+.001586, +.002991] | +.001432 [+.000366, +.002462] | -.000349 [-.001519, +.000873] |
| Medium | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| High | Short-only | +.000476 [-.000350, +.001319] | +.000169 [-.000594, +.000878] | -.000601 [-.001652, +.000507] | -.001539 [-.002778, -.000394] |
| High | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |

### Sports and Outdoors

| Group | Users | Setting | N@5 | N@10 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Low | 242,507 | Full SPC | .028751 | .032296 | .036910 | .047929 |
| Low | 242,507 | Short-only | .025640 | .029264 | .034279 | .045545 |
| Low | 242,507 | ρ-shuffled mean | .028751 | .032296 | .036910 | .047929 |
| Medium | 216,766 | Full SPC | .031319 | .033833 | .038000 | .045791 |
| Medium | 216,766 | Short-only | .027505 | .030284 | .034392 | .043005 |
| Medium | 216,766 | ρ-shuffled mean | .031319 | .033833 | .038000 | .045791 |
| High | 166,290 | Full SPC | .039749 | .041619 | .045210 | .051025 |
| High | 166,290 | Short-only | .033652 | .035601 | .038631 | .044711 |
| High | 166,290 | ρ-shuffled mean | .039749 | .041619 | .045210 | .051025 |

| Group | Control | ΔN@5 [CI] | ΔN@10 [CI] | ΔR@5 [CI] | ΔR@10 [CI] |
| --- | --- | --- | --- | --- | --- |
| Low | Short-only | +.003111 [+.002742, +.003446] | +.003032 [+.002706, +.003342] | +.002631 [+.002165, +.003097] | +.002383 [+.001851, +.002953] |
| Low | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Medium | Short-only | +.003814 [+.003410, +.004221] | +.003549 [+.003144, +.003947] | +.003608 [+.003058, +.004124] | +.002786 [+.002228, +.003372] |
| Medium | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| High | Short-only | +.006098 [+.005596, +.006624] | +.006019 [+.005530, +.006501] | +.006579 [+.005923, +.007199] | +.006314 [+.005628, +.006952] |
| High | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |

### Toys and Games

| Group | Users | Setting | N@5 | N@10 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Low | 209,555 | Full SPC | .042468 | .047624 | .055470 | .071447 |
| Low | 209,555 | Short-only | .041541 | .046716 | .054983 | .071041 |
| Low | 209,555 | ρ-shuffled mean | .042470 | .047626 | .055470 | .071447 |
| Medium | 189,093 | Full SPC | .039064 | .042401 | .048072 | .058410 |
| Medium | 189,093 | Short-only | .039133 | .042534 | .048748 | .059257 |
| Medium | 189,093 | ρ-shuffled mean | .039064 | .042401 | .048072 | .058410 |
| High | 166,018 | Full SPC | .041000 | .043042 | .047121 | .053434 |
| High | 166,018 | Short-only | .040395 | .042677 | .046772 | .053844 |
| High | 166,018 | ρ-shuffled mean | .041000 | .043042 | .047121 | .053434 |

| Group | Control | ΔN@5 [CI] | ΔN@10 [CI] | ΔR@5 [CI] | ΔR@10 [CI] |
| --- | --- | --- | --- | --- | --- |
| Low | Short-only | +.000927 [+.000550, +.001336] | +.000908 [+.000518, +.001289] | +.000487 [-.000105, +.001060] | +.000406 [-.000234, +.001064] |
| Low | ρ-shuffled mean | -.000002 [-.000007, .000000] | -.000002 [-.000006, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Medium | Short-only | -.000069 [-.000465, +.000330] | -.000133 [-.000499, +.000232] | -.000677 [-.001227, -.000106] | -.000846 [-.001433, -.000259] |
| Medium | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| High | Short-only | +.000605 [+.000191, +.001020] | +.000365 [-.000045, +.000751] | +.000349 [-.000223, +.000916] | -.000410 [-.000982, +.000163] |
| High | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |

## 7. High uncertainty 是否帶來更大的增益？

| Subset | Control | High−Low ΔN@5 [CI] | High−Low ΔN@10 [CI] | High−Low ΔR@5 [CI] | High−Low ΔR@10 [CI] |
| --- | --- | --- | --- | --- | --- |
| All Beauty | Short-only | -.005600 [-.011330, -.001073] | -.004543 [-.010103, +.000382] | -.008121 [-.018705, .000000] | -.004926 [-.017108, +.008121] |
| All Beauty | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Baby Products | Short-only | +.000054 [-.000926, +.001118] | -.000142 [-.001119, +.000788] | +.000345 [-.001029, +.001753] | -.000296 [-.001894, +.001123] |
| Baby Products | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Sports and Outdoors | Short-only | +.002987 [+.002377, +.003643] | +.002987 [+.002416, +.003555] | +.003948 [+.003112, +.004695] | +.003931 [+.003038, +.004828] |
| Sports and Outdoors | ρ-shuffled mean | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |
| Toys and Games | Short-only | -.000322 [-.000918, +.000252] | -.000543 [-.001110, -.000006] | -.000137 [-.000930, +.000659] | -.000815 [-.001718, +.000082] |
| Toys and Games | ρ-shuffled mean | +.000002 [.000000, +.000007] | +.000002 [.000000, +.000006] | .000000 [.000000, .000000] | .000000 [.000000, .000000] |

只有這個比較能直接檢查「High 的增益比 Low 更大」；High 組本身優於控制組，不自動表示比 Low 組更有幫助。

## 8. 實際 preference 混合權重

權重使用科學記號，避免極小值四捨五入成 0。以下是 Full 模型的 w_P。

| Subset | Group | Mean w_P | Median w_P | Q25 | Q75 |
| --- | --- | --- | --- | --- | --- |
| All Beauty | Low | 1.71760990e-01 | 1.71650514e-01 | 1.71017915e-01 | 1.72465645e-01 |
| All Beauty | Medium | 1.71353092e-01 | 1.71364322e-01 | 1.70533895e-01 | 1.72110960e-01 |
| All Beauty | High | 1.71243545e-01 | 1.71113119e-01 | 1.70551166e-01 | 1.71891470e-01 |
| Baby Products | Low | 2.42317612e-07 | 4.07575413e-08 | 1.94794403e-08 | 9.62564215e-08 |
| Baby Products | Medium | 2.16971363e-07 | 8.07400298e-08 | 3.87394934e-08 | 1.75664923e-07 |
| Baby Products | High | 4.45589269e-07 | 1.54419368e-07 | 5.72532599e-08 | 3.49351296e-07 |
| Sports and Outdoors | Low | 1.57953006e-05 | 4.10772039e-08 | 3.34558248e-09 | 3.76266115e-07 |
| Sports and Outdoors | Medium | 4.22764029e-06 | 1.78753385e-08 | 2.18128077e-09 | 1.61830101e-07 |
| Sports and Outdoors | High | 3.70779756e-06 | 4.61843435e-08 | 9.09423647e-09 | 2.49670137e-07 |
| Toys and Games | Low | 5.41260892e-05 | 6.86673445e-07 | 7.52579794e-08 | 9.62842478e-06 |
| Toys and Games | Medium | 2.78765929e-05 | 4.26365375e-07 | 8.56122426e-08 | 2.60161642e-06 |
| Toys and Games | High | 2.72314457e-05 | 1.10159544e-06 | 2.48696779e-07 | 5.15374768e-06 |

## 9. 逐筆數據核對：ρ 打亂是否真的改變輸出？

| Subset | Test uncertainty range | ρ range | 至少一次 w_P 改變的 users | 至少一次 full rank 改變的 users | 最大 |Δw_P| |
| --- | --- | --- | --- | --- | --- |
| All Beauty | .9996356368–.9997997284 | .38598633–.40136719 | 0 | 0 | 0.00000000e+00 |
| Baby Products | .9738512039–.9963516593 | .03561401–.45385742 | 134,736 | 133 | 5.05975913e-05 |
| Sports and Outdoors | .9505338073–.9939128757 | .02827454–.16174316 | 383,888 | 1,705 | 1.16994977e-03 |
| Toys and Games | .9267278910–.9913761616 | .02774048–.14929199 | 481,179 | 7,631 | 2.44107842e-03 |

此表直接比對保存陣列；排名改變包含 Top-10 之外的位置。沒有排名改變／沒有 Top-K 指標改變，不等於所有候選分數完全不變。混合精度與 gate 敏感度均可能影響干預的可觀察程度；未另作高精度重算，不能由本表判定單一原因。

## 10. 結果解讀與限制

- **All Beauty：** Medium 的 NDCG@10、Recall@10 增益 CI 高於 0；High 無穩定正增益。High−Low 的 NDCG@5 反而為負且 CI 排除 0，不支持「High 更受益」。
- **Baby Products：** Medium 的 NDCG@5、NDCG@10、Recall@5 有正向 CI；High 的 Recall@10 下降，四項 High−Low CI 都包含 0。
- **Sports and Outdoors：** 需依第 7 節各指標的 High−Low CI 判斷；其中 NDCG@10 的 High 組 Full−Short-only 增益較 Low 大，是對「完整架構在高相對不確定性組更有幫助」的正向證據。這仍不能歸因於 ρ，因為該 subset 的 ρ-shuffled 指標差為 0。
- **Toys and Games：** Low 的 NDCG@10 有正向 CI；High 的該項 CI 包含 0，不能宣稱高不確定性有更大收益。ρ-shuffled 的微小差異應依表格呈現，不能籠統寫成四個 subset 全部完全不變。

結論應區分：**Full vs Short-only 的架構效果、High vs Low 的效果異質性、以及 ρ 特定對應的干預效果**。這三個問題不能互相替代。Short-only 的移除同時改變 coordinator、容量與聯合訓練路徑；ρ-shuffled 只破壞一個 context 值的對應關係，仍保留 preference 的其他資訊。

各 subset 的 uncertainty 分布與門檻不同，尤其 All Beauty entropy 非常接近 1；不能把各 subset 的 High 當成相同的絕對不確定程度，也不能把它當成模型錯誤率。所有結果只有 seed 25252；CI 是使用者抽樣而非跨訓練 seed 的不確定性，且多組／多指標比較未校正。

## 11. 原始資料與程式索引

- [啟動腳本](run_rq5.sh)
- [RQ5 訓練擴充、分組、評估、bootstrap 與 JSON 寫入程式](src/rq5.py)
- [共享模型與 coordinator 實作](src/model_mamba_rl.py)
- [既有訓練／full-catalog 評估程式](src/train_mamba_rl.py)
- [資料前處理與切分](src/data.py)

| Subset | 分組摘要及 CI | 逐筆結果 | Full scores | Short-only scores |
| --- | --- | --- | --- | --- |
| All Beauty | [rq5_summary.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Full_Beauty/rq5_summary.json) | [rq5.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Full_Beauty/rq5.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Full_Beauty/full/scores.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Full_Beauty/short_only/scores.json) |
| Baby Products | [rq5_summary.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Baby_Products/rq5_summary.json) | [rq5.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Baby_Products/rq5.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Baby_Products/full/scores.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Baby_Products/short_only/scores.json) |
| Sports and Outdoors | [rq5_summary.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Sports_and_Outdoors/rq5_summary.json) | [rq5.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Sports_and_Outdoors/rq5.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Sports_and_Outdoors/full/scores.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Sports_and_Outdoors/short_only/scores.json) |
| Toys and Games | [rq5_summary.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Toys_and_Games/rq5_summary.json) | [rq5.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Toys_and_Games/rq5.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Toys_and_Games/full/scores.json) | [scores.json](outputs_mamba_rl/rq5/rq5_20261016_172900_246709/SW4_P32/Toys_and_Games/short_only/scores.json) |

`rq5.json` 的 groups.low/medium/high 保存每筆 user/target 的內部整數 ID、short_uncertainty、實際 w_P/w_S、原始 ρ、四項 Full 指標，以及 Short-only 指標與每次 shuffled 的指標／權重／ρ。`rq5_arrays.npz` 是陣列核對來源。使用者 ID 為過濾後資料內部索引，不是原始 Amazon ID。

## 12. 數據一致性稽核

RQ5 分組分析會重新計算每位使用者的 rank；下表比較「分組指標依人數加權」減去 `full/short_only/scores.json` 的原始 evaluator 總分。報告第 6–8 節統一使用 RQ5 summary/arrays 的結果，不混用兩次評估。

| Subset | Setting | ΔN@5 | ΔN@10 | ΔR@5 | ΔR@10 |
| --- | --- | --- | --- | --- | --- |
| All Beauty | full | -1.14663140e-11 | -2.73835940e-10 | +0.00000000e+00 | +0.00000000e+00 |
| All Beauty | short_only | +8.86622928e-10 | +8.29968691e-10 | +0.00000000e+00 | +0.00000000e+00 |
| Baby Products | full | +7.24670740e-11 | -1.50672232e-10 | +0.00000000e+00 | +0.00000000e+00 |
| Baby Products | short_only | +5.01492806e-11 | -1.58030790e-10 | +0.00000000e+00 | +0.00000000e+00 |
| Sports and Outdoors | full | -6.99884430e-08 | +3.86453047e-07 | +0.00000000e+00 | +1.59856002e-06 |
| Sports and Outdoors | short_only | +4.43670517e-11 | -4.81334643e-07 | +0.00000000e+00 | -1.59856002e-06 |
| Toys and Games | full | +5.45748793e-11 | +3.15010717e-08 | +0.00000000e+00 | +0.00000000e+00 |
| Toys and Games | short_only | -6.85070117e-07 | -5.44362757e-08 | -1.77095841e-06 | +0.00000000e+00 |

Sports and Outdoors 的 Full Recall@10 多 1 位命中、Short-only Recall@10 少 1 位命中；Toys and Games 的 Short-only Recall@5 少 1 位命中。其他細小 NDCG 差異也一併保留。這可能涉及重評分／浮點同分邊界，但僅憑現有 artifacts 不能確定原因；本報告未修改任何原始分數或重跑模型。表格四捨五入可能將極小值顯示為零，CI 是否排除零應以原始 JSON 判斷。
