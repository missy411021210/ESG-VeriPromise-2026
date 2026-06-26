#!/usr/bin/env python
"""Build model-level CV probability predictions from fold prediction CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]

TASK_KEYS = {
    "promise": "promise_status",
    "evidence_status": "evidence_status",
    "evidence_quality": "evidence_quality",
    "timeline": "verification_timeline",
}

LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Clear", "Not Clear", "Misleading"],
    "verification_timeline": [
        "N/A",
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    ],
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)


def task_from_group(group: str) -> str | None:
    for key, task in sorted(TASK_KEYS.items(), key=lambda item: len(item[0]), reverse=True):
        if group.endswith(f"_{key}"):
            return task
    return None


def prob_columns(task: str) -> list[str]:
    return [f"{task}__{label}" for label in LABELS[task]]


def read_prob_matrix(df: pd.DataFrame, columns: list[str], path: Path) -> np.ndarray | None:
    missing = set(columns) - set(df.columns)
    if missing:
        print(f"[skip] {path}: missing probability columns")
        return None
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = [col for col in columns if numeric[col].isna().any()]
        print(f"[skip] {path}: invalid/blank probability values in {bad_cols}")
        return None
    return numeric.to_numpy(dtype=float)


def labels_from_probs(probs: np.ndarray, task: str) -> list[str]:
    labels = LABELS[task]
    return [labels[int(i)] for i in probs.argmax(axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="predictions/cv5", type=Path)
    parser.add_argument("--output_dir", default="predictions/cv5_probs", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^(cv5_.+)_fold(\d+)_(val|test)\.csv$")
    groups: dict[str, dict[str, list[Path]]] = {}
    for folder in [args.pred_dir / "oof_folds", args.pred_dir / "folds"]:
        for path in folder.glob("cv5_*_fold*_*.csv"):
            match = pattern.match(path.name)
            if not match:
                continue
            group, _fold, split = match.groups()
            groups.setdefault(group, {"val": [], "test": []})[split].append(path)

    built = 0
    for group, split_paths in sorted(groups.items()):
        task = task_from_group(group)
        if task is None:
            print(f"[skip] cannot infer task from {group}")
            continue
        pcols = prob_columns(task)

        val_paths = sorted(split_paths["val"])
        test_paths = sorted(split_paths["test"])
        if len(val_paths) != args.folds or len(test_paths) != args.folds:
            print(f"[skip] {group}: val_folds={len(val_paths)} test_folds={len(test_paths)}")
            continue

        val_frames = []
        ok = True
        for path in val_paths:
            df = read_csv(path)
            probs = read_prob_matrix(df, pcols, path)
            if probs is None:
                ok = False
                break
            df = df.copy()
            df[task] = labels_from_probs(probs, task)
            val_frames.append(df)
        if not ok:
            continue

        val_df = pd.concat(val_frames, ignore_index=True)
        if val_df["id"].duplicated().any():
            raise ValueError(f"{group} has duplicated OOF ids")
        val_df = val_df.sort_values("id").reset_index(drop=True)
        val_out = args.output_dir / f"{group}_val.csv"
        val_df.to_csv(val_out, index=False, encoding="utf-8", lineterminator="\n")

        test_dfs = [read_csv(path) for path in test_paths]
        ids = test_dfs[0]["id"].tolist()
        if any(df["id"].tolist() != ids for df in test_dfs[1:]):
            raise ValueError(f"{group} test fold ids are not aligned")
        test_probs = []
        ok = True
        for path, df in zip(test_paths, test_dfs):
            probs = read_prob_matrix(df, pcols, path)
            if probs is None:
                ok = False
                break
            test_probs.append(probs)
        if not ok:
            continue
        avg_probs = np.mean(test_probs, axis=0)
        test_df = test_dfs[0].copy()
        test_df[task] = labels_from_probs(avg_probs, task)
        for col_idx, col in enumerate(pcols):
            test_df[col] = avg_probs[:, col_idx]
        test_out = args.output_dir / f"{group}_test.csv"
        test_df.to_csv(test_out, index=False, encoding="utf-8", lineterminator="\n")
        built += 1
        print(f"saved {val_out} and {test_out}")

    print(f"built {built} probability model predictions")


if __name__ == "__main__":
    main()
