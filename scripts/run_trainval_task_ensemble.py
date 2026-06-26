#!/usr/bin/env python
"""Train selected task/model candidates on train+val and build an ensemble submission.

The candidate set is tailored to the current VeriPromiseESG experiments:

promise_status:
  - hfl/chinese-roberta-wwm-ext-large
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large
evidence_status:
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large
  - hfl/chinese-roberta-wwm-ext-large
evidence_quality:
  - BAAI/bge-m3
verification_timeline:
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large
  - Qwen/Qwen3-Embedding-0.6B

Existing checkpoints/predictions are skipped by default.
The final submission uses majority voting by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TASKS = {
    "promise": "promise_status",
    "evidence_status": "evidence_status",
    "evidence_quality": "evidence_quality",
    "timeline": "verification_timeline",
}


@dataclass(frozen=True)
class Candidate:
    task_key: str
    alias: str
    model: str
    out: str
    pooling: str
    epochs: int
    legacy_checkpoint: str | None = None

    @property
    def task(self) -> str:
        return TASKS[self.task_key]


CANDIDATES = [
    Candidate("promise", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 3),
    Candidate("promise", "bge", "BAAI/bge-m3", "bge_m3", "mean", 2, "checkpoints/final_bge_m3_promise_trainval"),
    Candidate("promise", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 2),
    Candidate("evidence_status", "bge", "BAAI/bge-m3", "bge_m3", "mean", 3),
    Candidate("evidence_status", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 4, "checkpoints/final_e5_evidence_status_trainval"),
    Candidate("evidence_status", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 4),
    Candidate("evidence_quality", "bge", "BAAI/bge-m3", "bge_m3", "mean", 4, "checkpoints/final_bge_m3_evidence_quality_trainval"),
    Candidate("timeline", "bge", "BAAI/bge-m3", "bge_m3", "mean", 2, "checkpoints/final_bge_m3_timeline_trainval"),
    Candidate("timeline", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 4),
    Candidate("timeline", "qwen3emb", "Qwen/Qwen3-Embedding-0.6B", "qwen3emb", "last", 4),
]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n" + "=" * 88)
    print(" ".join(cmd))
    print("=" * 88)
    if not dry_run:
        subprocess.run(cmd, check=True)


def checkpoint_dir(candidate: Candidate, output_root: Path) -> Path:
    if candidate.legacy_checkpoint and (Path(candidate.legacy_checkpoint) / "best.pt").exists():
        return Path(candidate.legacy_checkpoint)
    return output_root / candidate.out / candidate.task_key


def prediction_path(candidate: Candidate, pred_dir: Path) -> Path:
    return pred_dir / f"trainval_{candidate.out}_{candidate.task_key}_test.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train_plus_val_2000.csv")
    parser.add_argument("--val_csv", default="data/vpesg4k_val_1000.csv")
    parser.add_argument("--test_csv", default="data/vpesg4k_test_2000.csv")
    parser.add_argument("--output_root", default="checkpoints/final_trainval")
    parser.add_argument("--pred_dir", default="predictions/final_trainval")
    parser.add_argument("--submission", default="predictions/final_trainval_task_ensemble_submission.csv")
    parser.add_argument("--base_prediction", default="predictions/qwen3emb_fixed_val_test.csv")
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="override all per-candidate epoch settings; 0 uses each candidate default",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_predict", action="store_true")
    parser.add_argument("--weighted_vote", action="store_true", help="use weighted vote instead of majority vote")
    parser.add_argument(
        "--use_val_eval",
        action="store_true",
        help="use --val_csv for checkpoint selection; off by default to avoid train+val leakage",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    used_checkpoints: list[tuple[Candidate, Path]] = []

    for candidate in CANDIDATES:
        ckpt_dir = checkpoint_dir(candidate, output_root)
        if not args.force_train and (ckpt_dir / "best.pt").exists():
            print(f"[skip train] {ckpt_dir}")
        else:
            train_cmd = [
                sys.executable,
                "train_multitask.py",
                "--csv",
                args.csv,
                "--output_dir",
                str(ckpt_dir),
                "--model",
                candidate.model,
                "--pooling",
                candidate.pooling,
                "--epochs",
                str(args.epochs or candidate.epochs),
                "--batch_size",
                str(args.batch_size),
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--grad_accum",
                str(args.grad_accum),
                "--seed",
                str(args.seed),
                "--train_tasks",
                candidate.task,
                "--select_metric",
                candidate.task,
            ]
            if args.use_val_eval:
                train_cmd.extend(["--val_csv", args.val_csv])
            else:
                train_cmd.append("--no_eval_save_last")
            if args.bf16:
                train_cmd.append("--bf16")
            if args.class_weights:
                train_cmd.append("--class_weights")
            run(train_cmd, args.dry_run)
        used_checkpoints.append((candidate, ckpt_dir))

    prediction_paths: list[tuple[Candidate, Path]] = []
    for candidate, ckpt_dir in used_checkpoints:
        out_csv = prediction_path(candidate, pred_dir)
        if not args.force_predict and out_csv.exists():
            print(f"[skip predict] {out_csv}")
        else:
            pred_cmd = [
                sys.executable,
                "predict_multitask.py",
                "--csv",
                args.test_csv,
                "--checkpoint",
                str(ckpt_dir / "best.pt"),
                "--output",
                str(out_csv),
                "--batch_size",
                str(args.predict_batch_size),
            ]
            if args.bf16:
                pred_cmd.append("--bf16")
            run(pred_cmd, args.dry_run)
        prediction_paths.append((candidate, out_csv))

    vote_sources = {task_key: [] for task_key in TASKS}
    for candidate, out_csv in prediction_paths:
        vote_sources[candidate.task_key].append(str(out_csv))

    hybrid_cmd = [
        sys.executable,
        "scripts/make_hybrid_submission.py",
        "--base",
        args.base_prediction,
        "--output",
        args.submission,
    ]
    for task_key, task in TASKS.items():
        sources = vote_sources[task_key]
        if not sources:
            continue
        if args.weighted_vote:
            weighted_sources = [f"{source}:{len(sources) - i}" for i, source in enumerate(sources)]
            hybrid_cmd.extend(["--weighted_vote", f"{task}={','.join(weighted_sources)}"])
        else:
            hybrid_cmd.extend(["--vote", f"{task}={','.join(sources)}"])
    run(hybrid_cmd, args.dry_run)

    print("\nCandidates used:")
    for candidate, ckpt_dir in used_checkpoints:
        print(f"  {candidate.task:24s} {candidate.alias:12s} {ckpt_dir}")
    print(f"\nsubmission: {args.submission}")


if __name__ == "__main__":
    main()
