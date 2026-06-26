# 執行環境配置

本專案主要使用兩個 Python/Conda 環境：

1. `esg-veripromise-main`：主要模型訓練、預測、ensemble 搜尋使用。
2. `esg-veripromise-jina`：專門給 `jinaai/jina-embeddings-v3` 使用。

## 硬體環境

實驗環境：

```text
OS: Linux
GPU: NVIDIA GeForce RTX 4090
CUDA runtime: 12.8
Python: 3.10
```

## 建立主要環境 esg-veripromise-main

若要用 conda YAML 重建：

```bash
conda env create -f code_submission/environment_esg-veripromise-main.yml
conda activate esg-veripromise-main
```

或使用 requirements 安裝：

```bash
conda create -y -n esg-veripromise-main python=3.10 pip
conda activate esg-veripromise-main
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r code_submission/requirements.txt
```

實驗時主要套件版本紀錄於：

```text
code_submission/environment_esg-veripromise-main.yml
code_submission/requirements_freeze_esg-veripromise-main.txt
```

## 建立 Jina v3 環境

`jinaai/jina-embeddings-v3` 的 remote code 與較新版 transformers 相容性較敏感，因此使用獨立環境：

```bash
conda env create -f code_submission/environment_esg-veripromise-jina.yml
conda activate esg-veripromise-jina
```

若手動安裝，可使用：

```bash
conda create -y -n esg-veripromise-jina python=3.10 pip
conda activate esg-veripromise-jina
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
pip install transformers==4.46.3 "tokenizers>=0.20,<0.21" accelerate scikit-learn pandas numpy sentencepiece protobuf safetensors "huggingface_hub>=0.23,<1.0" tqdm einops peft
```

## 檢查環境

主要環境：

```bash
conda activate esg-veripromise-main
python - <<'PY'
import torch, transformers, pandas, numpy, sklearn
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
PY
```

Jina 環境：

```bash
conda activate esg-veripromise-jina
python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

## 重現最終 submission

若已經存在 `predictions/cv5_probs/`，可快速重現：

```bash
bash code_submission/run_reproduce_from_existing_probs.sh
```

若要完整從 5-fold 訓練開始重跑：

```bash
bash code_submission/run_reproduce.sh
```

完整重訓會訓練多個模型、四個任務與五個 fold，耗時較長且需要足夠 GPU 記憶體。
