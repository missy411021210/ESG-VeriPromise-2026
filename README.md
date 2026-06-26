# ESG-VeriPromise-2026

本專案用於 VeriPromiseESG4K 競賽任務，目標是從企業永續報告段落中，自動判斷 ESG 承諾、佐證依據、佐證品質與承諾可驗證時間。

最後採用的主流程是：

```text
資料前處理
→ 5-fold cross validation
→ 四個任務分開訓練
→ 產生 OOF validation 與 test 預測
→ 整理模型機率
→ 搜尋多模型 probability-average ensemble
→ 套用 hierarchy 修正
→ 輸出 submission CSV
```

## 任務定義

| 任務 | 輸出類別 | 說明 |
|---|---|---|
| `promise_status` | `Yes`, `No` | 判斷段落是否為 ESG 承諾 |
| `evidence_status` | `Yes`, `No`, `N/A` | 判斷承諾是否有佐證或具體執行依據 |
| `evidence_quality` | `N/A`, `Clear`, `Not Clear`, `Misleading` | 判斷佐證品質 |
| `verification_timeline` | `N/A`, `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years` | 判斷承諾可驗證時間 |

## Hierarchy 規則

產生 submission 前會套用以下規則：

```text
若 promise_status = No：
    verification_timeline = N/A
    evidence_status = N/A
    evidence_quality = N/A

若 evidence_status = No 或 N/A：
    evidence_quality = N/A
```

這些規則由 `scripts/make_hybrid_submission.py` 與 `scripts/search_probability_combinations.py` 產生 submission 時處理。

## 資料檔案

| 檔案 | 用途 |
|---|---|
| `data/vpesg_4k_train_1000.csv` | 原始訓練資料 |
| `data/vpesg4k_val_1000.csv` | 原始驗證資料 |
| `data/train_plus_val_2000.csv` | train + val 合併後做 5-fold 的資料 |
| `data/vpesg4k_test_2000.csv` | 最終 test 資料 |
| `data/cv5/fold*_train.csv` | 5-fold 訓練切分 |
| `data/cv5/fold*_val.csv` | 5-fold OOF validation 切分 |

## 最後有用到的程式

以下是最後流程中真正有用到的主要程式。

| 程式 | 用途 |
|---|---|
| `train_multitask.py` | 訓練模型。支援四任務、多模型、指定單一 task、class weights、bf16、不同 pooling。 |
| `predict_multitask.py` | 使用 checkpoint 產生 validation 或 test 預測。支援 `--save_probs` 輸出每個類別的機率。 |
| `scripts/make_cv_folds.py` | 建立 5-fold 切分資料。 |
| `scripts/run_cv_task_ensemble.py` | 自動跑「模型 × 任務 × 5-fold」訓練與預測，是後期主要 pipeline。 |
| `scripts/build_cv_probability_predictions.py` | 整理 fold 預測機率，產生每個模型/任務的 OOF 與 test probability CSV。 |
| `scripts/search_probability_combinations.py` | 搜尋多模型機率平均組合，並產生最終 submission。 |
| `scripts/search_voting_combinations.py` | 搜尋 hard voting 組合；最後分數較低，主要作為比較。 |
| `scripts/make_hybrid_submission.py` | 將不同任務的模型輸出合併成 submission，並套用 hierarchy。 |
| `scripts/summarize_task_scores.py` | 彙整各模型在 validation 上的任務分數。 |
| `scripts/run_single_task_grid.py` | 早期用於測試不同模型在單一任務上的表現。 |

以下程式曾用於資料擴增或 pseudo label 實驗，但不是最後主流程：

| 程式 | 用途 |
|---|---|
| `scripts/collect_esg_paragraphs.py` | 蒐集外部 ESG 段落 |
| `scripts/make_llm_aug_data.py` | 建立 LLM 擴增資料 |
| `scripts/make_pseudolabel_data.py` | 建立 pseudo-label prompt / CSV |
| `scripts/run_pseudolabel_llm.py` | 使用 LLM 產生 pseudo label |
| `scripts/run_trainval_task_ensemble.py` | train+val 直接訓練任務模型的早期版本 |
| `scripts/search_hybrid_combinations.py` | 搜尋不同任務模型混搭的早期版本 |

## 最後使用過的 5-fold 模型

最後主要搜尋來源位於：

```text
predictions/cv5_probs/
```

目前完成 5-fold 且有機率預測檔的模型包含：

| 簡稱 | Hugging Face 模型 |
|---|---|
| `bge_m3` | `BAAI/bge-m3` |
| `chinese_roberta_wwm_ext_large` | `hfl/chinese-roberta-wwm-ext-large` |
| `multilingual_e5_large` | `intfloat/multilingual-e5-large` |
| `qwen3emb4b` | `Qwen/Qwen3-Embedding-4B` |
| `labse` | `sentence-transformers/LaBSE` |
| `jina_embeddings_v3` | `jinaai/jina-embeddings-v3` |
| `xlm_roberta_large` | `FacebookAI/xlm-roberta-large` |
| `chinese_macbert_large` | `hfl/chinese-macbert-large` |
| `chinese_electra_180g_large_discriminator` | `hfl/chinese-electra-180g-large-discriminator` |
| `chinese_lert_large` | `hfl/chinese-lert-large` |
| `erlangshen_deberta_v2_710m_chinese` | `IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese` |

## 訓練參數

後期 5-fold 訓練主要使用以下設定：

| 參數 | 設定 |
|---|---|
| 訓練資料 | `data/train_plus_val_2000.csv` |
| fold 數 | 5 |
| 訓練方式 | 四個任務分開訓練 |
| checkpoint 選擇 | 依該任務 macro-F1 |
| epochs | 通常為 4 或 5，部分模型在候選清單中有不同預設 |
| loss | Cross Entropy Loss |
| 類別不平衡 | `--class_weights` |
| mixed precision | `--bf16` |
| 機率輸出 | `--save_probs` |
| ensemble | probability average |

範例指令：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias bge \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --grad_accum 2 \
  --bf16 \
  --class_weights \
  --save_probs
```

Jina v3 使用獨立環境：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias jina_embeddings_v3 \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --grad_accum 4 \
  --bf16 \
  --class_weights \
  --save_probs
```

## 機率檔整理

5-fold 預測完成後，整理每個模型的 OOF 與 test probability：

```bash
python scripts/build_cv_probability_predictions.py \
  --pred_dir predictions/cv5 \
  --output_dir predictions/cv5_probs
```

整理後會產生類似：

```text
predictions/cv5_probs/cv5_bge_m3_promise_val.csv
predictions/cv5_probs/cv5_bge_m3_promise_test.csv
predictions/cv5_probs/cv5_labse_evidence_quality_val.csv
predictions/cv5_probs/cv5_labse_evidence_quality_test.csv
```

## 最終模型組合搜尋

最後採用 probability average，而不是 hard voting。

核心概念：

```text
對同一任務的多個模型：
    平均每個類別的預測機率
    再用 argmax 選出最終類別
```

最後主要使用的搜尋指令：

```bash
python scripts/search_probability_combinations.py \
  --val_csv data/train_plus_val_2000.csv \
  --pred_dir predictions/cv5_probs \
  --pattern "*_val.csv" \
  --min_vote_size 1 \
  --max_vote_size 6 \
  --per_task_top_n 200 \
  --top_k 30 \
  --output predictions/cv5_probability_search_1to6_top200.csv \
  --make_submission predictions/cv5_best_probability_submission_1to6_top200.csv
```

參數說明：

| 參數 | 說明 |
|---|---|
| `--min_vote_size 1` | 每個任務至少使用 1 個模型 |
| `--max_vote_size 6` | 每個任務最多使用 6 個模型做機率平均 |
| `--per_task_top_n 200` | 每個任務先保留 OOF 分數前 200 組，再做跨任務搜尋 |
| `--top_k 30` | 輸出前 30 名組合到搜尋結果 CSV |
| `--output` | 儲存搜尋排行榜 |
| `--make_submission` | 產生最終 submission CSV |

## 最佳 OOF 結果

`predictions/cv5_probability_search_1to6_top200.csv` 第一名：

| 指標 | 分數 |
|---|---:|
| weighted score | 0.645890 |
| promise_status | 0.838182 |
| evidence_status | 0.726280 |
| evidence_quality | 0.461506 |
| verification_timeline | 0.658953 |

## 最後選用的模型組合

最後 submission 使用：

```text
predictions/cv5_best_probability_submission_1to6_top200.csv
```

各任務使用的模型如下。

### promise_status

- `bge_m3`
- `chinese_macbert_large`
- `chinese_roberta_wwm_ext_large`
- `labse`
- `multilingual_e5_large`
- `qwen3emb4b`

### evidence_status

- `chinese_electra_180g_large_discriminator`
- `chinese_roberta_wwm_ext_large`
- `jina_embeddings_v3`
- `multilingual_e5_large`

### evidence_quality

- `bge_m3`
- `chinese_electra_180g_large_discriminator`
- `chinese_roberta_wwm_ext_large`
- `jina_embeddings_v3`
- `labse`
- `multilingual_e5_large`

### verification_timeline

- `bge_m3`
- `chinese_roberta_wwm_ext_large`
- `jina_embeddings_v3`
- `labse`
- `xlm_roberta_large`

## 最終 submission 檢查

最終 submission 需包含以下欄位：

```text
id
promise_status
verification_timeline
evidence_status
evidence_quality
```

已檢查項目：

```text
列數與 test 相同
id 與 test 完全對齊
無重複 id
無空值
無非法 label
已套用 hierarchy 規則
CSV 為 UTF-8
```

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `predictions/cv5_probability_search_1to5_top200.csv` | 最終 probability ensemble 搜尋結果 |
| `predictions/cv5_best_probability_submission_1to5_top200.csv` | 最終 submission |
| `predictions/cv5_probability_search_1to5_top100.csv` | top100 剪枝版本搜尋結果 |
| `predictions/cv5_best_probability_submission_1to5_top100.csv` | top100 剪枝版本 submission |
| `reports/ESG_Veripromise_完整流程與模型組合報告.pptx` | 完整實驗流程簡報 |

## 注意事項

- OOF validation 分數高不代表 public/private leaderboard 一定同步。
- `per_task_top_n` 可以加速搜尋，但可能漏掉單任務排名較後、但跨任務總分較好的組合。
- `--save_probs` 不影響訓練，只是額外輸出類別機率，供 probability ensemble 使用。
- 早期 pseudo label / data augmentation 實驗沒有穩定提升，因此最後主流程改用 5-fold + model ensemble。

## 程式使用方式

這一節說明每支重要程式怎麼使用。一般情況下，請先進入專案資料夾：

```bash
cd <PROJECT_ROOT>
```

若使用一般模型，例如 `bge_m3`、`roberta`、`e5`、`labse`：

```bash
conda activate esg-veripromise-main
```

若使用 `jinaai/jina-embeddings-v3`：

```bash
conda activate esg-veripromise-jina
```

### 1. 建立 5-fold 資料：`scripts/make_cv_folds.py`

用途：把 `train_plus_val_2000.csv` 切成 5 個 fold，供 5-fold 訓練使用。

輸入：

```text
data/train_plus_val_2000.csv
```

輸出：

```text
data/cv5/fold0_train.csv
data/cv5/fold0_val.csv
...
data/cv5/fold4_train.csv
data/cv5/fold4_val.csv
```

範例：

```bash
python scripts/make_cv_folds.py \
  --csv data/train_plus_val_2000.csv \
  --output_dir data/cv5 \
  --folds 5 \
  --seed 42
```

如果 `data/cv5/` 已經存在，通常不需要重跑。

### 2. 訓練單一模型：`train_multitask.py`

用途：訓練一個模型，可以訓練全部任務，也可以只訓練單一任務。後期主要用「單一任務訓練」。

單一任務訓練範例：

```bash
python train_multitask.py \
  --csv data/cv5/fold0_train.csv \
  --val_csv data/cv5/fold0_val.csv \
  --output_dir checkpoints/cv5/bge_m3/promise/fold0 \
  --model BAAI/bge-m3 \
  --pooling mean \
  --epochs 5 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --grad_accum 2 \
  --bf16 \
  --class_weights \
  --train_tasks promise_status \
  --select_metric promise_status
```

重要參數：

| 參數 | 說明 |
|---|---|
| `--csv` | 訓練資料 |
| `--val_csv` | 驗證資料 |
| `--output_dir` | checkpoint 輸出資料夾 |
| `--model` | Hugging Face 模型名稱 |
| `--pooling` | pooling 方式，常用 `cls`、`mean`、`last` |
| `--epochs` | 訓練 epoch 數 |
| `--batch_size` | 訓練 batch size |
| `--eval_batch_size` | 驗證 batch size |
| `--grad_accum` | gradient accumulation |
| `--bf16` | 使用 bfloat16 |
| `--class_weights` | 使用類別權重處理不平衡 |
| `--train_tasks` | 指定訓練任務 |
| `--select_metric` | 用哪個任務分數選 best checkpoint |

輸出：

```text
checkpoints/.../best.pt
checkpoints/.../history.json
checkpoints/.../tokenizer/
```

### 3. 單一 checkpoint 預測：`predict_multitask.py`

用途：用訓練好的 `best.pt` 對 validation 或 test 做預測。

預測 test 範例：

```bash
python predict_multitask.py \
  --csv data/vpesg4k_test_2000.csv \
  --checkpoint checkpoints/cv5/bge_m3/promise/fold0/best.pt \
  --output predictions/example_test.csv \
  --batch_size 1 \
  --bf16 \
  --save_probs
```

重要參數：

| 參數 | 說明 |
|---|---|
| `--csv` | 要預測的 CSV |
| `--checkpoint` | 訓練好的 checkpoint |
| `--output` | 預測輸出檔 |
| `--batch_size` | 預測 batch size |
| `--bf16` | 使用 bfloat16 |
| `--save_probs` | 輸出每個類別的機率欄位 |
| `--hierarchy` | 對輸出套用 hierarchy 規則 |

若加上 `--save_probs`，輸出會包含：

```text
promise_status__No
promise_status__Yes
evidence_status__N/A
evidence_status__No
evidence_status__Yes
...
```

這些機率欄位是 probability ensemble 必需的。

### 4. 自動跑 5-fold：`scripts/run_cv_task_ensemble.py`

用途：自動執行「模型 × 任務 × 5 folds」的訓練與預測，是後期最主要的 pipeline。

使用單一模型跑四個任務：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias bge \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --grad_accum 2 \
  --bf16 \
  --class_weights \
  --save_probs
```

使用 Jina v3：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias jina_embeddings_v3 \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --grad_accum 4 \
  --bf16 \
  --class_weights \
  --save_probs
```

只跑某一個任務：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias bge \
  --only_tasks evidence_quality \
  --epochs 5 \
  --bf16 \
  --class_weights \
  --save_probs
```

先檢查會執行哪些指令，但不真的訓練：

```bash
python scripts/run_cv_task_ensemble.py \
  --only_alias bge \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --bf16 \
  --class_weights \
  --save_probs \
  --dry_run
```

重要參數：

| 參數 | 說明 |
|---|---|
| `--only_alias` | 指定要跑哪個模型 alias，例如 `bge`、`labse`、`jina_embeddings_v3` |
| `--only_tasks` | 指定任務，可用 `promise,evidence_status,evidence_quality,timeline` |
| `--epochs` | 覆蓋候選模型預設 epoch |
| `--force_train` | 即使 checkpoint 已存在也重新訓練 |
| `--force_predict` | 即使 prediction 已存在也重新預測 |
| `--skip_failed` | 某個 fold 失敗時繼續跑其他項目 |
| `--save_probs` | 輸出機率欄位 |
| `--dry_run` | 只印出指令，不執行 |

輸出位置：

```text
checkpoints/cv5/<model>/<task>/fold*/best.pt
predictions/cv5/oof_folds/
predictions/cv5/folds/
predictions/cv5/cv5_<model>_<task>_val.csv
predictions/cv5/cv5_<model>_<task>_test.csv
```

### 5. 整理 5-fold 機率：`scripts/build_cv_probability_predictions.py`

用途：把 `scripts/run_cv_task_ensemble.py` 產生的 fold 機率檔整理成每個模型/任務一組 OOF 與 test probability CSV。

範例：

```bash
python scripts/build_cv_probability_predictions.py \
  --pred_dir predictions/cv5 \
  --output_dir predictions/cv5_probs
```

輸出：

```text
predictions/cv5_probs/cv5_bge_m3_promise_val.csv
predictions/cv5_probs/cv5_bge_m3_promise_test.csv
predictions/cv5_probs/cv5_labse_evidence_quality_val.csv
predictions/cv5_probs/cv5_labse_evidence_quality_test.csv
```

如果 `predictions/cv5_probs/` 已經有最新檔案，通常不需要重跑。

### 6. 搜尋機率平均組合：`scripts/search_probability_combinations.py`

用途：用 OOF validation 機率搜尋最佳多模型 probability-average ensemble，並產生 submission。

最後主要使用的指令：

```bash
python scripts/search_probability_combinations.py \
  --val_csv data/train_plus_val_2000.csv \
  --pred_dir predictions/cv5_probs \
  --pattern "*_val.csv" \
  --min_vote_size 1 \
  --max_vote_size 5 \
  --per_task_top_n 200 \
  --top_k 30 \
  --output predictions/cv5_probability_search_1to5_top200.csv \
  --make_submission predictions/cv5_best_probability_submission_1to5_top200.csv
```

程式邏輯：

```text
1. 讀取每個模型在 OOF validation 的類別機率
2. 對每個任務嘗試不同模型組合
3. 對同一任務內模型的類別機率取平均
4. argmax 得到該任務預測
5. 套用 hierarchy-aware scoring
6. 找 weighted score 最高的四任務組合
7. 用同一組模型組合產生 test submission
```

重要參數：

| 參數 | 說明 |
|---|---|
| `--val_csv` | OOF validation 的真實標籤資料 |
| `--pred_dir` | probability CSV 所在資料夾 |
| `--pattern` | 讀取哪些 validation prediction |
| `--min_vote_size` | 每個任務最少用幾個模型 |
| `--max_vote_size` | 每個任務最多用幾個模型，`0` 表示不限制 |
| `--per_task_top_n` | 每個任務先保留前 N 個組合，`0` 表示不剪枝 |
| `--top_k` | 搜尋結果保留前幾名 |
| `--output` | 搜尋排行榜輸出 |
| `--make_submission` | 最佳組合的 submission 輸出 |

注意：如果 `--max_vote_size` 設太大且 `--per_task_top_n 0`，組合數會爆炸，可能跑非常久。

### 7. 搜尋多數決組合：`scripts/search_voting_combinations.py`

用途：使用 hard voting 搜尋模型組合。此方法只看每個模型最後預測的 label，不使用機率。

範例：

```bash
python scripts/search_voting_combinations.py \
  --val_csv data/train_plus_val_2000.csv \
  --pred_dir predictions/cv5 \
  --pattern "*_val.csv" \
  --min_vote_size 1 \
  --top_k 30 \
  --output predictions/cv5_voting_search_all.csv \
  --make_submission predictions/cv5_best_voting_submission_all.csv
```

後期實驗顯示 probability average 較佳，因此最後 submission 不採用 hard voting。

### 8. 手動合併任務輸出：`scripts/make_hybrid_submission.py`

用途：指定每個任務要使用哪些 prediction 檔，合成一個 submission。

範例：

```bash
python scripts/make_hybrid_submission.py \
  --base predictions/cv5_probs/cv5_bge_m3_promise_test.csv \
  --output predictions/custom_hybrid_submission.csv \
  --vote promise_status=predictions/cv5_probs/cv5_bge_m3_promise_test.csv,predictions/cv5_probs/cv5_labse_promise_test.csv \
  --vote evidence_status=predictions/cv5_probs/cv5_multilingual_e5_large_evidence_status_test.csv \
  --vote evidence_quality=predictions/cv5_probs/cv5_bge_m3_evidence_quality_test.csv \
  --vote verification_timeline=predictions/cv5_probs/cv5_xlm_roberta_large_timeline_test.csv
```

這支程式適合做手動實驗；最終自動搜尋則主要用 `scripts/search_probability_combinations.py`。

### 9. 彙整模型分數：`scripts/summarize_task_scores.py`

用途：把已經產生的 prediction / metrics 彙整成表格，方便比較模型。

範例：

```bash
python scripts/summarize_task_scores.py \
  --contains single_ \
  --output predictions/single_task_model_val_scores.csv
```

這支主要用於早期挑模型。

### 10. 早期單任務模型測試：`scripts/run_single_task_grid.py`

用途：對多個模型做單任務訓練，觀察哪個模型在哪個任務比較強。

範例：

```bash
python scripts/run_single_task_grid.py \
  --models bge_m3,multilingual_e5_large,chinese_roberta_wwm_ext_large \
  --tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --bf16 \
  --class_weights
```

後期已改用 `scripts/run_cv_task_ensemble.py` 做 5-fold。

### 11. 早期擴增資料程式

這些程式不是最後主流程，但保留作為實驗紀錄。

#### `scripts/collect_esg_paragraphs.py`

用途：從外部 ESG / 永續報告資料中蒐集段落。

範例：

```bash
python scripts/collect_esg_paragraphs.py \
  --output data/external_esg_paragraphs.csv
```

#### `scripts/make_llm_aug_data.py`

用途：建立 LLM 擴增資料，例如補強 `Misleading` 或 `Not Clear`。

範例：

```bash
python scripts/make_llm_aug_data.py \
  --input data/vpesg_4k_train_1000.csv \
  --output data/llm_aug_misleading.csv
```

#### `scripts/make_pseudolabel_data.py`

用途：把外部段落轉成 LLM 標註 prompt，或把 LLM response 轉成 pseudo-label CSV。

產生 prompts：

```bash
python scripts/make_pseudolabel_data.py export-prompts \
  --paragraphs data/external_esg_paragraphs.csv \
  --output data/pseudolabel_prompts.jsonl \
  --limit 500
```

整理 LLM responses：

```bash
python scripts/make_pseudolabel_data.py build-csv \
  --prompts data/pseudolabel_prompts.jsonl \
  --responses data/pseudolabel_responses.jsonl \
  --output data/pseudolabeled_external.csv \
  --source_model "Qwen/Qwen3-14B"
```

#### `scripts/run_pseudolabel_llm.py`

用途：使用本機或 Hugging Face LLM 對 prompts 產生 pseudo labels。

範例：

```bash
python scripts/run_pseudolabel_llm.py \
  --model Qwen/Qwen3-14B \
  --prompts data/pseudolabel_prompts.jsonl \
  --output data/pseudolabel_responses.jsonl
```

## 從零重現最終 submission

若要從已整理好的資料與 prediction 重新產生最後 submission，通常只需要：

```bash
python scripts/search_probability_combinations.py \
  --val_csv data/train_plus_val_2000.csv \
  --pred_dir predictions/cv5_probs \
  --pattern "*_val.csv" \
  --min_vote_size 1 \
  --max_vote_size 5 \
  --per_task_top_n 200 \
  --top_k 30 \
  --output predictions/cv5_probability_search_1to5_top200.csv \
  --make_submission predictions/cv5_best_probability_submission_1to5_top200.csv
```

若要完整從訓練開始重跑，順序是：

```text
1. scripts/make_cv_folds.py
2. scripts/run_cv_task_ensemble.py
3. scripts/build_cv_probability_predictions.py
4. scripts/search_probability_combinations.py
```
