#!/usr/bin/env python
"""Search task-wise probability-average combinations from OOF predictions."""

from __future__ import annotations

import argparse
import heapq
import itertools
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


def prob_columns(task: str) -> list[str]:
    return [f"{task}__{label}" for label in LABELS[task]]


def encode(values: pd.Series, task: str) -> np.ndarray:
    label_to_id = {label: i for i, label in enumerate(LABELS[task])}
    return np.asarray([label_to_id[normalize_label(v)] for v in values], dtype=np.int16)


def decode(values: np.ndarray, task: str) -> list[str]:
    labels = LABELS[task]
    return [labels[int(value)] for value in values]


def macro_f1_batch(true: np.ndarray, preds: np.ndarray, num_labels: int) -> np.ndarray:
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


def paired_test_path(val_path: Path) -> Path | None:
    text = str(val_path)
    for candidate in [Path(text.replace("_val.csv", "_test.csv")), Path(text.replace("_val.csv", "_submission.csv"))]:
        if candidate.exists():
            return candidate
    return None


def average_probability_candidates(
    task: str,
    paths: list[Path],
    predictions: dict[str, pd.DataFrame],
    min_size: int,
    max_size: int,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    pcols = prob_columns(task)
    prob_mats = []
    for path in paths:
        df = predictions[path.name]
        missing = set(pcols) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing probability columns: {sorted(missing)}")
        prob_mats.append(df[pcols].to_numpy(dtype=float))

    subsets = []
    pred_cols = []
    upper = min(max_size, len(paths)) if max_size > 0 else len(paths)
    for size in range(max(1, min_size), upper + 1):
        for subset in itertools.combinations(range(len(paths)), size):
            avg = np.mean([prob_mats[i] for i in subset], axis=0)
            subsets.append(tuple(paths[i].name for i in subset))
            pred_cols.append(avg.argmax(axis=1).astype(np.int16))
    return subsets, np.vstack(pred_cols)


def apply_hierarchy_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    no_promise = out["promise_status"].eq("No")
    out.loc[no_promise, ["verification_timeline", "evidence_status", "evidence_quality"]] = "N/A"
    no_evidence = out["evidence_status"].isin(["No", "N/A"])
    out.loc[no_evidence, "evidence_quality"] = "N/A"
    return out


def write_submission(output: Path, pred_dir: Path, selected: dict[str, tuple[str, ...]]) -> None:
    first_test = paired_test_path(pred_dir / selected["promise_status"][0])
    if first_test is None:
        raise SystemExit(f"no paired test file for {selected['promise_status'][0]}")
    base = read_csv(first_test)
    result = pd.DataFrame({"id": base["id"]})
    for task, subset in selected.items():
        pcols = prob_columns(task)
        mats = []
        for source in subset:
            test_path = paired_test_path(pred_dir / source)
            if test_path is None:
                raise SystemExit(f"no paired test file for {source}")
            df = read_csv(test_path)
            if df["id"].tolist() != result["id"].tolist():
                df = df.set_index("id").reindex(result["id"]).reset_index()
            missing = set(pcols) - set(df.columns)
            if missing:
                raise ValueError(f"{test_path} missing probability columns: {sorted(missing)}")
            mats.append(df[pcols].to_numpy(dtype=float))
        result[task] = decode(np.mean(mats, axis=0).argmax(axis=1), task)
    result = apply_hierarchy_frame(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result[SUBMISSION_COLUMNS].to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    print(f"saved probability-average submission {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", default="data/train_plus_val_2000.csv", type=Path)
    parser.add_argument("--pred_dir", default="predictions/cv5_probs", type=Path)
    parser.add_argument("--pattern", default="*_val.csv")
    parser.add_argument("--min_vote_size", type=int, default=2)
    parser.add_argument("--max_vote_size", type=int, default=3)
    parser.add_argument(
        "--per_task_top_n",
        type=int,
        default=0,
        help="keep only the top N probability subsets per task before the cross-task search; 0 means keep all",
    )
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--make_submission", default=None, type=Path)
    args = parser.parse_args()

    true_df = read_csv(args.val_csv).sort_values("id").reset_index(drop=True)
    true = {task: encode(true_df[task], task) for task in TASKS}

    predictions = {}
    all_paths = []
    for path in sorted(args.pred_dir.glob(args.pattern)):
        df = read_csv(path).sort_values("id").reset_index(drop=True)
        if df["id"].tolist() != true_df["id"].tolist():
            continue
        predictions[path.name] = df
        all_paths.append(path)
    if not all_paths:
        raise SystemExit("no usable probability validation files found")

    subsets = {}
    preds = {}
    scores = {}
    for task in TASKS:
        suffix = f"_{TASK_KEYS[task]}_val.csv"
        task_paths = [path for path in all_paths if path.name.endswith(suffix)]
        if not task_paths:
            raise SystemExit(f"no probability files for {task}")
        subsets[task], preds[task] = average_probability_candidates(
            task,
            task_paths,
            predictions,
            args.min_vote_size,
            args.max_vote_size,
        )
        scores[task] = macro_f1_batch(true[task], preds[task], len(LABELS[task]))
        print(
            f"{task}: files={len(task_paths)} prob_candidates={len(subsets[task])} "
            f"best_single_task={scores[task].max():.4f}"
        )
        if args.per_task_top_n > 0 and len(subsets[task]) > args.per_task_top_n:
            keep = np.argsort(scores[task])[-args.per_task_top_n:][::-1]
            subsets[task] = [subsets[task][int(i)] for i in keep]
            preds[task] = preds[task][keep]
            scores[task] = scores[task][keep]
            print(
                f"  kept top {args.per_task_top_n} for {task}; "
                f"worst_kept={scores[task].min():.4f}"
            )

    p_preds = preds["promise_status"]
    e_preds = preds["evidence_status"]
    q_preds = preds["evidence_quality"]
    t_preds = preds["verification_timeline"]
    no_promise_masks = p_preds == LABELS["promise_status"].index("No")
    na_evidence = LABELS["evidence_status"].index("N/A")
    no_evidence = LABELS["evidence_status"].index("No")
    na_quality = LABELS["evidence_quality"].index("N/A")
    na_timeline = LABELS["verification_timeline"].index("N/A")

    promise_scores = scores["promise_status"]
    evidence_scores = np.empty((len(p_preds), len(e_preds)), dtype=float)
    timeline_scores = np.empty((len(p_preds), len(t_preds)), dtype=float)
    quality_scores = np.empty((len(p_preds), len(e_preds), len(q_preds)), dtype=float)

    print("precomputing hierarchy-aware task scores...")
    for p_idx, no_promise in enumerate(no_promise_masks):
        adjusted_e = e_preds.copy()
        adjusted_e[:, no_promise] = na_evidence
        evidence_scores[p_idx] = macro_f1_batch(true["evidence_status"], adjusted_e, len(LABELS["evidence_status"]))

        adjusted_t = t_preds.copy()
        adjusted_t[:, no_promise] = na_timeline
        timeline_scores[p_idx] = macro_f1_batch(true["verification_timeline"], adjusted_t, len(LABELS["verification_timeline"]))

        for e_idx, adjusted_e_col in enumerate(adjusted_e):
            force_quality_na = np.logical_or(no_promise, np.logical_or(adjusted_e_col == na_evidence, adjusted_e_col == no_evidence))
            adjusted_q = q_preds.copy()
            adjusted_q[:, force_quality_na] = na_quality
            quality_scores[p_idx, e_idx] = macro_f1_batch(true["evidence_quality"], adjusted_q, len(LABELS["evidence_quality"]))

    total = len(subsets["promise_status"]) * len(subsets["evidence_status"]) * len(subsets["evidence_quality"]) * len(subsets["verification_timeline"])
    print(f"searching probability combinations: {total}")

    heap: list[tuple[float, tuple[int, int, int, int]]] = []
    for p_idx in range(len(subsets["promise_status"])):
        p_score = TASK_WEIGHTS["promise_status"] * promise_scores[p_idx]
        for e_idx in range(len(subsets["evidence_status"])):
            pe_score = p_score + TASK_WEIGHTS["evidence_status"] * evidence_scores[p_idx, e_idx]
            combo_scores = (
                pe_score
                + TASK_WEIGHTS["evidence_quality"] * quality_scores[p_idx, e_idx][:, None]
                + TASK_WEIGHTS["verification_timeline"] * timeline_scores[p_idx][None, :]
            )
            if len(heap) < args.top_k:
                for q_idx, t_idx in np.ndindex(combo_scores.shape):
                    heapq.heappush(heap, (float(combo_scores[q_idx, t_idx]), (p_idx, e_idx, q_idx, t_idx)))
                    if len(heap) > args.top_k:
                        heapq.heappop(heap)
            else:
                hits = np.argwhere(combo_scores > heap[0][0])
                for q_idx, t_idx in hits:
                    heapq.heappushpop(heap, (float(combo_scores[q_idx, t_idx]), (p_idx, e_idx, int(q_idx), int(t_idx))))

    rows = []
    for weighted, (p_idx, e_idx, q_idx, t_idx) in sorted(heap, reverse=True):
        rows.append(
            {
                "weighted": weighted,
                "promise_status": promise_scores[p_idx],
                "evidence_status": evidence_scores[p_idx, e_idx],
                "evidence_quality": quality_scores[p_idx, e_idx, q_idx],
                "verification_timeline": timeline_scores[p_idx, t_idx],
                "promise_sources": "|".join(subsets["promise_status"][p_idx]),
                "evidence_status_sources": "|".join(subsets["evidence_status"][e_idx]),
                "evidence_quality_sources": "|".join(subsets["evidence_quality"][q_idx]),
                "timeline_sources": "|".join(subsets["verification_timeline"][t_idx]),
            }
        )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
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
        write_submission(args.make_submission, args.pred_dir, selected)


if __name__ == "__main__":
    main()
