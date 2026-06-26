#!/usr/bin/env python
"""Search task-wise majority-vote combinations from OOF validation predictions."""

from __future__ import annotations

import argparse
import heapq
import itertools
from collections import Counter
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
    "promise_status": "promise",
    "evidence_status": "evidence_status",
    "evidence_quality": "evidence_quality",
    "verification_timeline": "timeline",
}

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


def encode(values: pd.Series, task: str) -> np.ndarray:
    label_to_id = {label: i for i, label in enumerate(LABELS[task])}
    encoded = []
    for value in values:
        label = normalize_label(value)
        if label not in label_to_id:
            raise ValueError(f"invalid {task} label: {label!r}")
        encoded.append(label_to_id[label])
    return np.asarray(encoded, dtype=np.int16)


def decode(values: np.ndarray, task: str) -> list[str]:
    labels = LABELS[task]
    return [labels[int(value)] for value in values]


def macro_f1(true: np.ndarray, pred: np.ndarray, num_labels: int) -> float:
    scores = []
    for label_id in range(num_labels):
        true_is_label = true == label_id
        pred_is_label = pred == label_id
        tp = np.logical_and(true_is_label, pred_is_label).sum()
        fp = np.logical_and(~true_is_label, pred_is_label).sum()
        fn = np.logical_and(true_is_label, ~pred_is_label).sum()
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def batch_macro_f1(true: np.ndarray, preds: np.ndarray, num_labels: int) -> np.ndarray:
    scores = []
    for label_id in range(num_labels):
        true_is_label = true[None, :] == label_id
        pred_is_label = preds == label_id
        tp = np.logical_and(true_is_label, pred_is_label).sum(axis=1)
        fp = np.logical_and(~true_is_label, pred_is_label).sum(axis=1)
        fn = np.logical_and(true_is_label, ~pred_is_label).sum(axis=1)
        denom = (2 * tp) + fp + fn
        scores.append(np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0))
    return np.mean(np.vstack(scores), axis=0)


def majority_vote_matrix(preds: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    # Tie break: follow the first source in the subset if it is tied, otherwise
    # use the smallest label id for deterministic output.
    selected = preds[list(subset)]
    fallback = selected[0]
    out = np.empty(selected.shape[1], dtype=np.int16)
    for col_idx in range(selected.shape[1]):
        counts = Counter(int(value) for value in selected[:, col_idx])
        top_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == top_count}
        fallback_label = int(fallback[col_idx])
        out[col_idx] = fallback_label if fallback_label in winners else min(winners)
    return out


def build_vote_candidates(
    task: str,
    paths: list[Path],
    predictions: dict[str, pd.DataFrame],
    min_size: int,
    max_size: int,
) -> tuple[list[str], list[tuple[str, ...]], np.ndarray]:
    raw = np.vstack([encode(predictions[path.name][task], task) for path in paths])
    names = [path.name for path in paths]
    voted_names = []
    voted_subsets = []
    voted_columns = []
    upper = min(max_size, len(paths)) if max_size > 0 else len(paths)
    lower = max(1, min_size)
    for size in range(lower, upper + 1):
        for subset in itertools.combinations(range(len(paths)), size):
            voted_subsets.append(tuple(names[i] for i in subset))
            voted_names.append("+".join(names[i].replace("_val.csv", "") for i in subset))
            voted_columns.append(majority_vote_matrix(raw, subset))
    return voted_names, voted_subsets, np.vstack(voted_columns)


def apply_hierarchy_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    no_promise = out["promise_status"].eq("No")
    out.loc[no_promise, ["verification_timeline", "evidence_status", "evidence_quality"]] = "N/A"
    no_evidence = out["evidence_status"].isin(["No", "N/A"])
    out.loc[no_evidence, "evidence_quality"] = "N/A"
    return out


def write_submission(
    output: Path,
    test_ids: pd.Series,
    selected: dict[str, tuple[str, ...]],
    pred_dir: Path,
) -> None:
    result = pd.DataFrame({"id": test_ids})
    for task, subset in selected.items():
        cols = []
        for val_name in subset:
            test_path = paired_test_path(pred_dir / val_name)
            if test_path is None:
                raise SystemExit(f"no paired test file for {val_name}")
            df = read_csv(test_path)
            if df["id"].tolist() != test_ids.tolist():
                df = df.set_index("id").reindex(test_ids).reset_index()
            cols.append(encode(df[task], task))
        raw = np.vstack(cols)
        voted = majority_vote_matrix(raw, tuple(range(len(cols))))
        result[task] = decode(voted, task)

    result = apply_hierarchy_frame(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result[SUBMISSION_COLUMNS].to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    print(f"saved best voting submission {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", default="data/train_plus_val_2000.csv", type=Path)
    parser.add_argument("--pred_dir", default="predictions/cv5", type=Path)
    parser.add_argument("--pattern", default="*_val.csv")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--make_submission", default=None, type=Path)
    parser.add_argument("--min_vote_size", type=int, default=1)
    parser.add_argument("--max_vote_size", type=int, default=0, help="0 means no maximum")
    parser.add_argument(
        "--all_sources_for_each_task",
        action="store_true",
        help="use every prediction file for every task; default uses task-specific files only",
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
    true = {task: encode(true_df[task], task) for task in TASKS}

    excludes = [value.strip() for value in args.exclude.split(",") if value.strip()]
    predictions = {}
    paths = []
    for path in sorted(args.pred_dir.glob(args.pattern)):
        if excludes and any(value in path.name for value in excludes):
            continue
        if args.require_test_pair and paired_test_path(path) is None:
            continue
        df = read_csv(path)
        if {"id", *TASKS} - set(df.columns):
            continue
        df = df.sort_values("id").reset_index(drop=True)
        if df["id"].tolist() != true_df["id"].tolist():
            continue
        predictions[path.name] = df
        paths.append(path)

    if not paths:
        raise SystemExit("no usable validation prediction files found")

    task_paths = {}
    for task in TASKS:
        if args.all_sources_for_each_task:
            selected_paths = paths
        else:
            suffix = f"_{TASK_KEYS[task]}_val.csv"
            selected_paths = [path for path in paths if path.name.endswith(suffix)]
        if not selected_paths:
            raise SystemExit(f"no candidate files for {task}")
        task_paths[task] = selected_paths

    voted_names = {}
    voted_subsets = {}
    voted_preds = {}
    single_scores = {}
    for task in TASKS:
        names, subsets, preds = build_vote_candidates(
            task,
            task_paths[task],
            predictions,
            args.min_vote_size,
            args.max_vote_size,
        )
        voted_names[task] = names
        voted_subsets[task] = subsets
        voted_preds[task] = preds
        single_scores[task] = batch_macro_f1(true[task], preds, len(LABELS[task]))
        print(
            f"{task}: files={len(task_paths[task])} vote_candidates={len(names)} "
            f"best_single_task={single_scores[task].max():.4f}"
        )

    p_preds = voted_preds["promise_status"]
    e_preds = voted_preds["evidence_status"]
    q_preds = voted_preds["evidence_quality"]
    t_preds = voted_preds["verification_timeline"]
    no_promise_masks = p_preds == LABELS["promise_status"].index("No")
    na_evidence = LABELS["evidence_status"].index("N/A")
    no_evidence = LABELS["evidence_status"].index("No")
    na_quality = LABELS["evidence_quality"].index("N/A")
    na_timeline = LABELS["verification_timeline"].index("N/A")

    promise_scores = single_scores["promise_status"]
    evidence_scores = np.empty((len(p_preds), len(e_preds)), dtype=float)
    timeline_scores = np.empty((len(p_preds), len(t_preds)), dtype=float)
    quality_scores = np.empty((len(p_preds), len(e_preds), len(q_preds)), dtype=float)

    print("precomputing hierarchy-aware task scores...")
    for p_idx, no_promise in enumerate(no_promise_masks):
        adjusted_e = e_preds.copy()
        adjusted_e[:, no_promise] = na_evidence
        evidence_scores[p_idx] = batch_macro_f1(
            true["evidence_status"],
            adjusted_e,
            len(LABELS["evidence_status"]),
        )

        adjusted_t = t_preds.copy()
        adjusted_t[:, no_promise] = na_timeline
        timeline_scores[p_idx] = batch_macro_f1(
            true["verification_timeline"],
            adjusted_t,
            len(LABELS["verification_timeline"]),
        )

        for e_idx, adjusted_e_col in enumerate(adjusted_e):
            no_good_evidence = np.logical_or(adjusted_e_col == na_evidence, adjusted_e_col == no_evidence)
            force_quality_na = np.logical_or(no_promise, no_good_evidence)
            adjusted_q = q_preds.copy()
            adjusted_q[:, force_quality_na] = na_quality
            quality_scores[p_idx, e_idx] = batch_macro_f1(
                true["evidence_quality"],
                adjusted_q,
                len(LABELS["evidence_quality"]),
            )

    total = (
        len(voted_names["promise_status"])
        * len(voted_names["evidence_status"])
        * len(voted_names["evidence_quality"])
        * len(voted_names["verification_timeline"])
    )
    print(f"searching vote combinations: {total}")

    heap: list[tuple[float, tuple[int, int, int, int]]] = []
    for p_idx in range(len(voted_names["promise_status"])):
        p_score = TASK_WEIGHTS["promise_status"] * promise_scores[p_idx]
        for e_idx in range(len(voted_names["evidence_status"])):
            pe_score = p_score + TASK_WEIGHTS["evidence_status"] * evidence_scores[p_idx, e_idx]
            q_part = TASK_WEIGHTS["evidence_quality"] * quality_scores[p_idx, e_idx]
            t_part = TASK_WEIGHTS["verification_timeline"] * timeline_scores[p_idx]
            # Keep this vectorized over quality/timeline for the current promise/evidence pair.
            combo_scores = q_part[:, None] + t_part[None, :] + pe_score
            if len(heap) < args.top_k:
                for q_idx, t_idx in np.ndindex(combo_scores.shape):
                    heapq.heappush(heap, (float(combo_scores[q_idx, t_idx]), (p_idx, e_idx, q_idx, t_idx)))
                    if len(heap) > args.top_k:
                        heapq.heappop(heap)
            else:
                threshold = heap[0][0]
                hits = np.argwhere(combo_scores > threshold)
                for q_idx, t_idx in hits:
                    heapq.heappushpop(
                        heap,
                        (float(combo_scores[q_idx, t_idx]), (p_idx, e_idx, int(q_idx), int(t_idx))),
                    )

    rows = []
    for weighted, (p_idx, e_idx, q_idx, t_idx) in sorted(heap, reverse=True):
        rows.append(
            {
                "weighted": weighted,
                "promise_status": promise_scores[p_idx],
                "evidence_status": evidence_scores[p_idx, e_idx],
                "evidence_quality": quality_scores[p_idx, e_idx, q_idx],
                "verification_timeline": timeline_scores[p_idx, t_idx],
                "promise_sources": "|".join(voted_subsets["promise_status"][p_idx]),
                "evidence_status_sources": "|".join(voted_subsets["evidence_status"][e_idx]),
                "evidence_quality_sources": "|".join(voted_subsets["evidence_quality"][q_idx]),
                "timeline_sources": "|".join(voted_subsets["verification_timeline"][t_idx]),
            }
        )

    result = pd.DataFrame(rows)
    print(result.head(args.top_k).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\nsaved {args.output}")

    if args.make_submission:
        best = result.iloc[0]
        selected = {
            "promise_status": tuple(best["promise_sources"].split("|")),
            "evidence_status": tuple(best["evidence_status_sources"].split("|")),
            "evidence_quality": tuple(best["evidence_quality_sources"].split("|")),
            "verification_timeline": tuple(best["timeline_sources"].split("|")),
        }
        first_test = paired_test_path(args.pred_dir / selected["promise_status"][0])
        if first_test is None:
            raise SystemExit(f"no paired test file for {selected['promise_status'][0]}")
        test_ids = read_csv(first_test)["id"]
        write_submission(args.make_submission, test_ids, selected, args.pred_dir)


if __name__ == "__main__":
    main()
