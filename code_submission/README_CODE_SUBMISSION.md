# 程式碼交付說明

本資料夾整理競賽模型可重現所需的原始實作程式碼、參數設定、執行環境與重現流程。

## 目錄內容

```text
code_submission/
├── README_CODE_SUBMISSION.md
├── environment.txt
├── environment_setup.md
├── environment_esg-veripromise-main.yml
├── environment_esg-veripromise-jina.yml
├── final_model_config.json
├── requirements.txt
├── requirements_freeze_esg-veripromise-main.txt
├── run_reproduce.sh
├── run_reproduce_from_existing_probs.sh
└── src/
    ├── train_multitask.py
    ├── predict_multitask.py
    └── scripts/
        ├── make_cv_folds.py
        ├── run_cv_task_ensemble.py
        ├── build_cv_probability_predictions.py
        ├── search_probability_combinations.py
        ├── search_voting_combinations.py
        ├── make_hybrid_submission.py
        ├── summarize_task_scores.py
        ├── run_single_task_grid.py
        ├── collect_esg_paragraphs.py
        ├── make_llm_aug_data.py
        ├── make_pseudolabel_data.py
        ├── run_pseudolabel_llm.py
        ├── run_trainval_task_ensemble.py
        └── search_hybrid_combinations.py
```

## 包含內容對照

| 要求項目 | 對應檔案 |
|---|---|
| 前處理程式碼 | `src/scripts/make_cv_folds.py`, `src/train_multitask.py` |
| 訓練程式碼 | `src/train_multitask.py`, `src/scripts/run_cv_task_ensemble.py` |
| 辨識/推論程式碼 | `src/predict_multitask.py`, `src/scripts/search_probability_combinations.py` |
| 各項參數設定 | `final_model_config.json`, `README_CODE_SUBMISSION.md` |
| 訓練權重/ensemble 權重 | `final_model_config.json` 中的 `ensemble.model_weight` 與 `selected_models` |
| 執行環境 | `environment.txt`, `environment_setup.md`, `environment_esg-veripromise-main.yml`, `environment_esg-veripromise-jina.yml`, `requirements.txt`, `requirements_freeze_esg-veripromise-main.txt` |

## 主要程式邏輯

### 1. 前處理

前處理包含：

1. 讀取 CSV。
2. 將空字串、`nan`、`None` 等標籤正規化為 `N/A`。
3. 建立 5-fold validation 切分。
4. 在輸出階段套用 hierarchy 規則。

相關程式：

```text
src/scripts/make_cv_folds.py
src/train_multitask.py
src/scripts/search_probability_combinations.py
```

### 2. 訓練

訓練採用 Hugging Face `AutoModel` 作為 backbone，接上四個分類 heads。

模型輸出會先經過 pooling：

```text
mean pooling：BGE / E5 / LaBSE / Jina
CLS pooling：HFL Chinese RoBERTa / MacBERT / ELECTRA / LERT / XLM-RoBERTa
last token pooling：Qwen3 Embedding
```

訓練 loss：

```text
CrossEntropyLoss
```

類別不平衡處理：

```text
--class_weights
```

混合精度：

```text
--bf16
```

訓練策略：

```text
四個任務分開訓練
每個任務 5-fold
checkpoint 依該任務 validation macro-F1 選擇 best.pt
```

### 3. 推論

`src/predict_multitask.py` 載入 `best.pt` 後，對 validation 或 test 產生：

1. 預測 label。
2. 各類別機率欄位。

機率欄位範例：

```text
promise_status__No
promise_status__Yes
evidence_status__N/A
evidence_status__No
evidence_status__Yes
```

### 4. Ensemble

最終提交使用 probability-average ensemble。

同一任務內，對入選模型的類別機率做等權重平均：

```text
avg_prob = mean(model_probabilities)
final_label = argmax(avg_prob)
```

四個任務分別搜尋最佳模型組合，最後再依 hierarchy 修正。

最終搜尋設定：

```text
min_vote_size = 1
max_vote_size = 6
per_task_top_n = 200
top_k = 30
```

最終 OOF weighted score：

```text
0.645890
```

## 從資料切分開始重現

請先回到專案根目錄：

```bash
cd <PROJECT_ROOT>
```

若要完整從訓練開始重跑，可直接執行：

```bash
bash code_submission/run_reproduce.sh
```

此腳本會依序執行：

```text
1. 建立 5-fold
2. 訓練 final_model_config.json 中最後 ensemble 需要的模型
3. 產生 OOF 與 test prediction
4. 整理 probability CSV
5. 搜尋 probability-average ensemble
6. 產生 predictions/cv5_64589.csv
```

若已經保留 `predictions/cv5_probs/`，只想快速重現最後 submission，可執行：

```bash
bash code_submission/run_reproduce_from_existing_probs.sh
```

此快速腳本不會重新訓練模型，只會重新搜尋 ensemble 並輸出 submission。

### Step 1：建立 5-fold

```bash
python scripts/make_cv_folds.py \
  --csv data/train_plus_val_2000.csv \
  --output_dir data/cv5 \
  --folds 5 \
  --seed 42
```

### Step 2：訓練並預測 5-fold 模型

範例：訓練 BGE-M3 四個任務。

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

範例：訓練 Jina v3。

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

### Step 3：整理 fold 機率

```bash
python scripts/build_cv_probability_predictions.py \
  --pred_dir predictions/cv5 \
  --output_dir predictions/cv5_probs
```

### Step 4：搜尋最終 ensemble 並產生 submission

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

最終提交檔：

```text
predictions/cv5_64589.csv
```

此檔與：

```text
predictions/cv5_best_probability_submission_1to6_top200.csv
```

內容完全相同。
