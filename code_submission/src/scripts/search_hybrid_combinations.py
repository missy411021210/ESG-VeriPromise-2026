#!/usr/bin/env python
"""Search task-wise hybrid combinations from existing validation predictions."""

from __future__ import annotations

import argparse
import subprocess
import sys
import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score


TASKS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]

SUBMISSION_COLUMNS = [
    "id",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

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

TASK_WEIGHTS = {
    "promise_status": 0.20,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
    "verification_timeline": 0.15,
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "n/a"}:
        return "N/A"
    return text


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)


def apply_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    no_promise = out["promise_status"].eq("No")
    out.loc[no_promise, ["verification_timeline", "evidence_status", "evidence_quality"]] = "N/A"
    no_evidence = out["evidence_status"].isin(["No", "N/A"])
    out.loc[no_evidence, "evidence_quality"] = "N/A"
    return out


def score(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    pred_df = apply_hierarchy(pred_df)
    parts = {}
    for task in TASKS:
        parts[task] = f1_score(
            true_df[task].map(normalize_label),
            pred_df[task].map(normalize_label),
            labels=LABELS[task],
            average="macro",
            zero_division=0,
        )
    weighted = sum(parts[task] * TASK_WEIGHTS[task] for task in TASKS)
    return weighted, parts


def paired_test_path(val_path: Path) -> Path | None:
    text = str(val_path)
    candidates = [
        Path(text.replace("_val.csv", "_test.csv")),
        Path(text.replace("_val.csv", "_submission.csv")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", default="data/vpesg4k_val_1000.csv", type=Path)
    parser.add_argument("--pred_dir", default="predictions", type=Path)
    parser.add_argument("--pattern", default="*_val.csv")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--per_task_top_n",
        type=int,
        default=0,
        help="restrict each task to its top N single-column sources before searching; 0 means use all",
    )
    parser.add_argument("--output", default="", type=Path)
    parser.add_argument(
        "--make_submission",
        default="",
        type=Path,
        help="build a submission CSV from the best validation combination using paired *_test.csv files",
    )
    parser.add_argument(
        "--require_test_pair",
        action="store_true",
        help="only use validation predictions that have a paired test/submission file",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="comma-separated substrings; skip prediction files whose names contain any of them",
    )
    args = parser.parse_args()

    true_df = read_csv(args.val_csv).sort_values("id").reset_index(drop=True)
    pred_paths = sorted(args.pred_dir.glob(args.pattern))
    excludes = [value.strip() for value in args.exclude.split(",") if value.strip()]
    candidates: dict[str, pd.DataFrame] = {}
    for path in pred_paths:
        if excludes and any(value in path.name for value in excludes):
            continue
        if args.require_test_pair and paired_test_path(path) is None:
            continue
        df = read_csv(path)
        missing = {"id", *TASKS} - set(df.columns)
        if missing:
            continue
        df = df.sort_values("id").reset_index(drop=True)
        if df["id"].tolist() != true_df["id"].tolist():
            continue
        candidates[path.name] = df

    if not candidates:
        raise SystemExit("no usable prediction files found")

    names = sorted(candidates)

    task_names = {task: names for task in TASKS}
    if args.per_task_top_n > 0:
        for task in TASKS:
            scored = []
            for name in names:
                pred = pd.DataFrame(
                    {
                        "id": true_df["id"],
                        "promise_status": candidates[name]["promise_status"],
                        "evidence_status": candidates[name]["evidence_status"],
                        "evidence_quality": candidates[name]["evidence_quality"],
                        "verification_timeline": candidates[name]["verification_timeline"],
                    }
                )
                _, parts = score(true_df, pred)
                scored.append((parts[task], name))
            scored.sort(reverse=True)
            task_names[task] = [name for _, name in scored[: args.per_task_top_n]]
            print(
                f"{task} candidates: "
                + ", ".join(f"{name}={value:.4f}" for value, name in scored[: args.per_task_top_n])
            )
        print()
    else:
        print(f"using all {len(names)} prediction files for each task")

    total_combinations = 1
    for task in TASKS:
        total_combinations *= len(task_names[task])
    print(
        "search combinations: "
        + " x ".join(f"{task}={len(task_names[task])}" for task in TASKS)
        + f" => {total_combinations}"
    )

    rows = []
    for promise, evidence_status, evidence_quality, timeline in itertools.product(
        task_names["promise_status"],
        task_names["evidence_status"],
        task_names["evidence_quality"],
        task_names["verification_timeline"],
    ):
        pred = pd.DataFrame(
            {
                "id": true_df["id"],
                "promise_status": candidates[promise]["promise_status"],
                "evidence_status": candidates[evidence_status]["evidence_status"],
                "evidence_quality": candidates[evidence_quality]["evidence_quality"],
                "verification_timeline": candidates[timeline]["verification_timeline"],
            }
        )
        weighted, parts = score(true_df, pred)
        rows.append(
            {
                "weighted": weighted,
                **parts,
                "promise_source": promise,
                "evidence_status_source": evidence_status,
                "evidence_quality_source": evidence_quality,
                "timeline_source": timeline,
            }
        )

    result = pd.DataFrame(rows).sort_values("weighted", ascending=False).reset_index(drop=True)
    cols = [
        "weighted",
        *TASKS,
        "promise_source",
        "evidence_status_source",
        "evidence_quality_source",
        "timeline_source",
    ]
    print(result[cols].head(args.top_k).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result[cols].to_csv(args.output, index=False, encoding="utf-8")
        print(f"\nsaved {args.output}")

    if args.make_submission:
        best = result.iloc[0]
        source_by_task = {
            "promise_status": best["promise_source"],
            "evidence_status": best["evidence_status_source"],
            "evidence_quality": best["evidence_quality_source"],
            "verification_timeline": best["timeline_source"],
        }
        cmd = [
            sys.executable,
            "scripts/make_hybrid_submission.py",
            "--base",
            str(paired_test_path(args.pred_dir / source_by_task["promise_status"])),
            "--output",
            str(args.make_submission),
        ]
        for task, source_name in source_by_task.items():
            test_path = paired_test_path(args.pred_dir / source_name)
            if test_path is None:
                raise SystemExit(f"no paired test file for {source_name}")
            cmd.extend(["--override", f"{task}={test_path}"])
        print("\n" + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"saved best-combination submission {args.make_submission}")


if __name__ == "__main__":
    main()
