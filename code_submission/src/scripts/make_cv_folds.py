#!/usr/bin/env python
"""Create stratified CV folds for VeriPromiseESG training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


LABEL_COLUMNS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]


def build_stratify_labels(df: pd.DataFrame, n_splits: int) -> pd.Series:
    candidates = [
        LABEL_COLUMNS,
        ["promise_status", "evidence_status", "verification_timeline"],
        ["promise_status", "evidence_status"],
        ["promise_status"],
    ]
    for cols in candidates:
        labels = df[cols].astype(str).agg("||".join, axis=1)
        counts = labels.value_counts()
        rare = set(counts[counts < n_splits].index)
        if rare:
            collapsed = labels.where(~labels.isin(rare), "__rare__")
        else:
            collapsed = labels
        if collapsed.value_counts().min() >= n_splits:
            print(f"stratify columns: {cols}")
            print(f"stratify groups: {collapsed.nunique()}")
            return collapsed

    print("warning: falling back to row-index stratification; label distribution may be less stable")
    return pd.Series([str(i % n_splits) for i in range(len(df))], index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train_plus_val_2000.csv")
    parser.add_argument("--output_dir", default="data/cv5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)
    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing columns: {sorted(missing)}")

    labels = build_stratify_labels(df, args.folds)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, labels)):
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        train_path = output_dir / f"fold{fold}_train.csv"
        val_path = output_dir / f"fold{fold}_val.csv"
        train.to_csv(train_path, index=False, encoding="utf-8", lineterminator="\n")
        val.to_csv(val_path, index=False, encoding="utf-8", lineterminator="\n")

        print(f"fold {fold}: train={len(train)} val={len(val)}")
        for split_name, split_df in [("train", train), ("val", val)]:
            for col in LABEL_COLUMNS:
                counts = split_df[col].astype(str).value_counts().to_dict()
                summary_rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "task": col,
                        "counts": counts,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fold_summary.csv", index=False, encoding="utf-8")
    print(f"saved folds to {output_dir}")
    print(f"saved {output_dir / 'fold_summary.csv'}")


if __name__ == "__main__":
    main()
