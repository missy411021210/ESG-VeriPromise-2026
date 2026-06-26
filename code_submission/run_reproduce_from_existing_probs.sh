#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

MAIN_PYTHON=${MAIN_PYTHON:-python}

# Fast reproduction path:
# This assumes predictions/cv5_probs already exists.
# It recomputes the ensemble search and regenerates the final submission.
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
