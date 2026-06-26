#!/usr/bin/env python
"""Summarize validation macro-F1 by model and task from checkpoint histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASKS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]


def best_for_task(history: list[dict], task: str) -> tuple[float, int]:
    best = max(history, key=lambda row: row.get("macro", {}).get(task, -1.0))
    return float(best["macro"][task]), int(best["epoch"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--output", default="")
    p.add_argument(
        "--contains",
        default="",
        help="optional comma-separated substrings; keep checkpoint names containing any of them",
    )
    args = p.parse_args()

    filters = [v.strip() for v in args.contains.split(",") if v.strip()]
    rows = []
    for history_path in sorted(Path(args.checkpoint_root).glob("*/history.json")):
        name = history_path.parent.name
        if filters and not any(f in name for f in filters):
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            continue

        row = {"model": name}
        for task in TASKS:
            score, epoch = best_for_task(history, task)
            row[task] = score
            row[f"{task}_epoch"] = epoch
        weighted_best = max(history, key=lambda x: x.get("weighted", -1.0))
        row["weighted_best"] = float(weighted_best.get("weighted", 0.0))
        row["weighted_best_epoch"] = int(weighted_best.get("epoch", 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no histories found")

    print(df[["model", *TASKS, "weighted_best"]].to_string(index=False))
    print("\nBest by task:")
    for task in TASKS:
        idx = df[task].idxmax()
        best = df.loc[idx]
        print(f"{task:24s} {best['model']}  {best[task]:.4f}  epoch={int(best[f'{task}_epoch'])}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
