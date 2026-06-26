#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

MAIN_PYTHON=${MAIN_PYTHON:-python}
JINA_PYTHON=${JINA_PYTHON:-python}

# 1. Create CV folds.
$MAIN_PYTHON scripts/make_cv_folds.py \
  --csv data/train_plus_val_2000.csv \
  --output_dir data/cv5 \
  --folds 5 \
  --seed 42

# 2. Train 5-fold models.
# These are the non-Jina aliases used by the final ensemble.
$MAIN_PYTHON scripts/run_cv_task_ensemble.py \
  --only_alias bge,macbert,roberta,labse,e5,qwen3emb4b,electra,xlm_roberta_large \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --grad_accum 2 \
  --bf16 \
  --class_weights \
  --save_probs

# Jina v3 uses a separate compatible environment.
# If it is not the currently activated environment, pass:
# JINA_PYTHON=/path/to/esg-veripromise-jina/bin/python bash code_submission/run_reproduce.sh
$JINA_PYTHON scripts/run_cv_task_ensemble.py \
  --only_alias jina_embeddings_v3 \
  --only_tasks promise,evidence_status,evidence_quality,timeline \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --grad_accum 4 \
  --bf16 \
  --class_weights \
  --save_probs

# 3. Build model-level probability files.
$MAIN_PYTHON scripts/build_cv_probability_predictions.py \
  --pred_dir predictions/cv5 \
  --output_dir predictions/cv5_probs

# 4. Search probability ensemble and write final submission.
$MAIN_PYTHON scripts/search_probability_combinations.py \
  --val_csv data/train_plus_val_2000.csv \
  --pred_dir predictions/cv5_probs \
  --pattern "*_val.csv" \
  --min_vote_size 1 \
  --max_vote_size 6 \
  --per_task_top_n 200 \
  --top_k 30 \
  --output predictions/cv5_probability_search_1to6_top200.csv \
  --make_submission predictions/cv5_best_probability_submission_1to6_top200.csv

cp predictions/cv5_best_probability_submission_1to6_top200.csv predictions/cv5_64589.csv
