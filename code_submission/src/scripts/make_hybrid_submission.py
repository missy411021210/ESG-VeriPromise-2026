#!/usr/bin/env python3
"""
Build a submission CSV by mixing task columns from multiple prediction files.

Examples:
    python scripts/make_hybrid_submission.py \
        --base predictions/qwen3emb_val_predictions.csv \
        --override evidence_status=predictions/bge_m3_test.csv \
        --override verification_timeline=predictions/qwen3emb_aug_mis50_gov80_test.csv \
        --output predictions/hybrid_submission.csv

    python scripts/make_hybrid_submission.py \
        --base predictions/qwen3emb_val_predictions.csv \
        --vote promise_status=predictions/bge_m3_test.csv,predictions/e5_test.csv \
        --output predictions/hybrid_vote_submission.csv

    python scripts/make_hybrid_submission.py \
        --base predictions/qwen3emb_val_predictions.csv \
        --weighted_vote promise_status=predictions/bge_m3_test.csv:2,predictions/e5_test.csv:1 \
        --output predictions/hybrid_weighted_vote_submission.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd


TASKS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

SUBMISSION_COLUMNS = [
    "id",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

LABELS = {
    "promise_status": {"Yes", "No"},
    "verification_timeline": {
        "N/A",
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    },
    "evidence_status": {"Yes", "No", "N/A"},
    "evidence_quality": {"Clear", "Not Clear", "Misleading", "N/A"},
}


def parse_task_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected TASK=CSV")
    task, source = raw.split("=", 1)
    task = task.strip()
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unknown task: {task}")
    return task, Path(source)


def parse_task_sources(raw: str) -> tuple[str, list[Path]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected TASK=CSV1,CSV2,...")
    task, sources = raw.split("=", 1)
    task = task.strip()
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unknown task: {task}")
    paths = [Path(x.strip()) for x in sources.split(",") if x.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("at least one CSV is required")
    return task, paths


def parse_weighted_task_sources(raw: str) -> tuple[str, list[tuple[Path, float]]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected TASK=CSV1:WEIGHT,CSV2:WEIGHT,...")
    task, sources = raw.split("=", 1)
    task = task.strip()
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unknown task: {task}")

    weighted_paths = []
    for item in [x.strip() for x in sources.split(",") if x.strip()]:
        if ":" not in item:
            weighted_paths.append((Path(item), 1.0))
            continue
        source, weight = item.rsplit(":", 1)
        try:
            parsed_weight = float(weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid weight in {item!r}") from exc
        if parsed_weight <= 0:
            raise argparse.ArgumentTypeError("weights must be positive")
        weighted_paths.append((Path(source), parsed_weight))

    if not weighted_paths:
        raise argparse.ArgumentTypeError("at least one CSV is required")
    return task, weighted_paths


def read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str},
        keep_default_na=False,
    )
    missing = {"id", *TASKS} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def align_column(source: pd.DataFrame, ids: pd.Series, task: str, path: Path) -> pd.Series:
    if source["id"].duplicated().any():
        dupes = source.loc[source["id"].duplicated(), "id"].head().tolist()
        raise ValueError(f"{path} contains duplicated ids, examples: {dupes}")

    aligned = source.set_index("id").reindex(ids)[task]
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()[:5]
        raise ValueError(f"{path} does not contain all ids, examples: {missing}")

    bad = sorted(set(aligned.astype(str)) - LABELS[task])
    if bad:
        raise ValueError(f"{path} has invalid labels for {task}: {bad}")
    return aligned.astype(str).reset_index(drop=True)


def majority_vote(columns: list[pd.Series], fallback: pd.Series) -> pd.Series:
    voted = []
    for i in range(len(fallback)):
        counts = Counter(col.iloc[i] for col in columns)
        top_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == top_count}
        fallback_label = fallback.iloc[i]
        voted.append(fallback_label if fallback_label in winners else sorted(winners)[0])
    return pd.Series(voted)


def weighted_vote(
    columns: list[tuple[pd.Series, float]],
    fallback: pd.Series,
) -> pd.Series:
    voted = []
    for i in range(len(fallback)):
        scores: dict[str, float] = {}
        for col, weight in columns:
            label = col.iloc[i]
            scores[label] = scores.get(label, 0.0) + weight
        top_score = max(scores.values())
        winners = {label for label, score in scores.items() if score == top_score}
        fallback_label = fallback.iloc[i]
        voted.append(fallback_label if fallback_label in winners else sorted(winners)[0])
    return pd.Series(voted)


def apply_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    no_promise = out["promise_status"] == "No"
    out.loc[no_promise, ["verification_timeline", "evidence_status", "evidence_quality"]] = "N/A"

    no_evidence = out["evidence_status"].isin(["No", "N/A"])
    out.loc[no_evidence, "evidence_quality"] = "N/A"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        type=parse_task_source,
        help="replace one task column using TASK=CSV",
    )
    parser.add_argument(
        "--vote",
        action="append",
        default=[],
        type=parse_task_sources,
        help="majority vote one task using TASK=CSV1,CSV2,...; base breaks ties",
    )
    parser.add_argument(
        "--weighted_vote",
        action="append",
        default=[],
        type=parse_weighted_task_sources,
        help="weighted vote one task using TASK=CSV1:WEIGHT,CSV2:WEIGHT,...; missing weights default to 1",
    )
    parser.add_argument("--no_hierarchy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_prediction(args.base)
    result = base[SUBMISSION_COLUMNS].copy()

    ids = result["id"]
    for task, source_path in args.override:
        source = read_prediction(source_path)
        result[task] = align_column(source, ids, task, source_path)

    for task, source_paths in args.vote:
        columns = []
        for source_path in source_paths:
            source = read_prediction(source_path)
            columns.append(align_column(source, ids, task, source_path))
        result[task] = majority_vote(columns, result[task].astype(str))

    for task, weighted_source_paths in args.weighted_vote:
        columns = []
        for source_path, weight in weighted_source_paths:
            source = read_prediction(source_path)
            columns.append((align_column(source, ids, task, source_path), weight))
        result[task] = weighted_vote(columns, result[task].astype(str))

    for task in TASKS:
        bad = sorted(set(result[task].astype(str)) - LABELS[task])
        if bad:
            raise ValueError(f"invalid labels for {task}: {bad}")

    if not args.no_hierarchy:
        result = apply_hierarchy(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"saved {args.output} ({len(result)} rows)")
    print(result[SUBMISSION_COLUMNS].head().to_string(index=False))


if __name__ == "__main__":
    main()
