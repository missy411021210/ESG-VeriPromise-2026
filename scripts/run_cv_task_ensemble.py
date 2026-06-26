#!/usr/bin/env python
"""Run 5-fold task/model ensemble for VeriPromiseESG.

自動化訓練腳本
自動跑：多模型 × 四個任務 × 5-fold
1. 對每個模型、每個任務、每個 fold 訓練
2. 產生 fold validation 預測
3. 產生 fold test 預測
4. 建立 OOF validation
5. 建立每個模型的 test 預測

The default candidate set trains selected model families for all four tasks.

Existing 5-fold models:
  - hfl/chinese-roberta-wwm-ext-large
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large
New 5-fold models:
  - Qwen/Qwen3-Embedding-4B
  - FacebookAI/xlm-roberta-large
  - sentence-transformers/LaBSE
  - hfl/chinese-macbert-large
  - hfl/chinese-electra-180g-large-discriminator
  - hfl/chinese-lert-large
  - IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese

For each candidate, this script:
1. trains one checkpoint per fold,
2. predicts fold validation data to build out-of-fold validation predictions,
3. predicts test with each fold checkpoint,
4. votes the fold test predictions into one model-level prediction,
5. votes model-level predictions into the final submission.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
    freeze_backbone: bool = False
    backbone_dtype: str = "auto"
    batch_size: int | None = None
    eval_batch_size: int | None = None
    grad_accum: int | None = None
    max_length: int | None = None
    grad_ckpt: bool = False

    @property
    def task(self) -> str:
        return TASKS[self.task_key]


CANDIDATES = [
    Candidate("promise", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 3),
    Candidate("promise", "macbert", "hfl/chinese-macbert-large", "chinese_macbert_large", "cls", 4),
    Candidate("promise", "electra", "hfl/chinese-electra-180g-large-discriminator", "chinese_electra_180g_large_discriminator", "cls", 4),
    Candidate("promise", "lert", "hfl/chinese-lert-large", "chinese_lert_large", "cls", 4),
    Candidate("promise", "bge", "BAAI/bge-m3", "bge_m3", "mean", 2),
    Candidate("promise", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 2),
    Candidate("promise", "qwen3emb4b", "Qwen/Qwen3-Embedding-4B", "qwen3emb4b", "last", 4, True, "bfloat16", 2, 4, 8),
    Candidate("promise", "xlm_roberta_large", "FacebookAI/xlm-roberta-large", "xlm_roberta_large", "cls", 4, False, "bfloat16", 4, 8, 4, None, True),
    Candidate("promise", "labse", "sentence-transformers/LaBSE", "labse", "mean", 4),
    Candidate("promise", "jina_embeddings_v3", "jinaai/jina-embeddings-v3", "jina_embeddings_v3", "mean", 4, False, "bfloat16", 4, 8, 4),
    Candidate("promise", "erlangshen_deberta", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "erlangshen_deberta_v2_710m_chinese", "cls", 4, False, "bfloat16", 2, 4, 8, 256, True),
    Candidate("evidence_status", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 4),
    Candidate("evidence_status", "macbert", "hfl/chinese-macbert-large", "chinese_macbert_large", "cls", 4),
    Candidate("evidence_status", "electra", "hfl/chinese-electra-180g-large-discriminator", "chinese_electra_180g_large_discriminator", "cls", 4),
    Candidate("evidence_status", "lert", "hfl/chinese-lert-large", "chinese_lert_large", "cls", 4),
    Candidate("evidence_status", "bge", "BAAI/bge-m3", "bge_m3", "mean", 3),
    Candidate("evidence_status", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 4),
    Candidate("evidence_status", "qwen3emb4b", "Qwen/Qwen3-Embedding-4B", "qwen3emb4b", "last", 4, True, "bfloat16", 2, 4, 8),
    Candidate("evidence_status", "xlm_roberta_large", "FacebookAI/xlm-roberta-large", "xlm_roberta_large", "cls", 4, False, "bfloat16", 4, 8, 4, None, True),
    Candidate("evidence_status", "labse", "sentence-transformers/LaBSE", "labse", "mean", 4),
    Candidate("evidence_status", "jina_embeddings_v3", "jinaai/jina-embeddings-v3", "jina_embeddings_v3", "mean", 4, False, "bfloat16", 4, 8, 4),
    Candidate("evidence_status", "erlangshen_deberta", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "erlangshen_deberta_v2_710m_chinese", "cls", 4, False, "bfloat16", 2, 4, 8, 256, True),
    Candidate("evidence_quality", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 4),
    Candidate("evidence_quality", "macbert", "hfl/chinese-macbert-large", "chinese_macbert_large", "cls", 4),
    Candidate("evidence_quality", "electra", "hfl/chinese-electra-180g-large-discriminator", "chinese_electra_180g_large_discriminator", "cls", 4),
    Candidate("evidence_quality", "lert", "hfl/chinese-lert-large", "chinese_lert_large", "cls", 4),
    Candidate("evidence_quality", "bge", "BAAI/bge-m3", "bge_m3", "mean", 4),
    Candidate("evidence_quality", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 4),
    Candidate("evidence_quality", "qwen3emb4b", "Qwen/Qwen3-Embedding-4B", "qwen3emb4b", "last", 4, True, "bfloat16", 2, 4, 8),
    Candidate("evidence_quality", "xlm_roberta_large", "FacebookAI/xlm-roberta-large", "xlm_roberta_large", "cls", 4, False, "bfloat16", 4, 8, 4, None, True),
    Candidate("evidence_quality", "labse", "sentence-transformers/LaBSE", "labse", "mean", 4),
    Candidate("evidence_quality", "jina_embeddings_v3", "jinaai/jina-embeddings-v3", "jina_embeddings_v3", "mean", 4, False, "bfloat16", 4, 8, 4),
    Candidate("evidence_quality", "erlangshen_deberta", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "erlangshen_deberta_v2_710m_chinese", "cls", 4, False, "bfloat16", 2, 4, 8, 256, True),
    Candidate("timeline", "roberta", "hfl/chinese-roberta-wwm-ext-large", "chinese_roberta_wwm_ext_large", "cls", 4),
    Candidate("timeline", "macbert", "hfl/chinese-macbert-large", "chinese_macbert_large", "cls", 4),
    Candidate("timeline", "electra", "hfl/chinese-electra-180g-large-discriminator", "chinese_electra_180g_large_discriminator", "cls", 4),
    Candidate("timeline", "lert", "hfl/chinese-lert-large", "chinese_lert_large", "cls", 4),
    Candidate("timeline", "bge", "BAAI/bge-m3", "bge_m3", "mean", 2),
    Candidate("timeline", "e5", "intfloat/multilingual-e5-large", "multilingual_e5_large", "mean", 4),
    Candidate("timeline", "qwen3emb4b", "Qwen/Qwen3-Embedding-4B", "qwen3emb4b", "last", 4, True, "bfloat16", 2, 4, 8),
    Candidate("timeline", "xlm_roberta_large", "FacebookAI/xlm-roberta-large", "xlm_roberta_large", "cls", 4, False, "bfloat16", 4, 8, 4, None, True),
    Candidate("timeline", "labse", "sentence-transformers/LaBSE", "labse", "mean", 4),
    Candidate("timeline", "jina_embeddings_v3", "jinaai/jina-embeddings-v3", "jina_embeddings_v3", "mean", 4, False, "bfloat16", 4, 8, 4),
    Candidate("timeline", "erlangshen_deberta", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "erlangshen_deberta_v2_710m_chinese", "cls", 4, False, "bfloat16", 2, 4, 8, 256, True),
]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n" + "=" * 88)
    print(" ".join(cmd))
    print("=" * 88)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_maybe(cmd: list[str], dry_run: bool, skip_failed: bool) -> bool:
    try:
        run(cmd, dry_run)
        return True
    except subprocess.CalledProcessError as exc:
        if not skip_failed:
            raise
        print(f"[failed, skipped] exit={exc.returncode} command: {' '.join(cmd)}")
        return False


def fold_train_path(fold_dir: Path, fold: int) -> Path:
    return fold_dir / f"fold{fold}_train.csv"


def fold_val_path(fold_dir: Path, fold: int) -> Path:
    return fold_dir / f"fold{fold}_val.csv"


def checkpoint_dir(candidate: Candidate, output_root: Path, fold: int) -> Path:
    return output_root / candidate.out / candidate.task_key / f"fold{fold}"


def fold_prediction_path(candidate: Candidate, pred_dir: Path, fold: int) -> Path:
    return pred_dir / "folds" / f"cv5_{candidate.out}_{candidate.task_key}_fold{fold}_test.csv"


def fold_oof_prediction_path(candidate: Candidate, pred_dir: Path, fold: int) -> Path:
    return pred_dir / "oof_folds" / f"cv5_{candidate.out}_{candidate.task_key}_fold{fold}_val.csv"


def model_prediction_path(candidate: Candidate, pred_dir: Path) -> Path:
    return pred_dir / f"cv5_{candidate.out}_{candidate.task_key}_test.csv"


def model_oof_prediction_path(candidate: Candidate, pred_dir: Path) -> Path:
    return pred_dir / f"cv5_{candidate.out}_{candidate.task_key}_val.csv"


def build_oof_prediction(candidate: Candidate, pred_dir: Path, fold_paths: list[Path], output: Path) -> None:
    frames = [
        pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)
        for path in fold_paths
    ]
    df = pd.concat(frames, ignore_index=True)
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].head().tolist()
        raise ValueError(f"duplicated OOF ids for {candidate.out}/{candidate.task_key}: {dupes}")
    df = df.sort_values("id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    print(f"saved OOF {output} ({len(df)} rows)")


def ensure_folds(args: argparse.Namespace) -> None:
    fold_dir = Path(args.fold_dir)
    missing = [
        path
        for fold in range(args.folds)
        for path in [fold_train_path(fold_dir, fold), fold_val_path(fold_dir, fold)]
        if not path.exists()
    ]
    if not missing:
        return

    cmd = [
        sys.executable,
        "scripts/make_cv_folds.py",
        "--csv",
        args.csv,
        "--output_dir",
        args.fold_dir,
        "--folds",
        str(args.folds),
        "--seed",
        str(args.seed),
    ]
    run(cmd, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train_plus_val_2000.csv")
    parser.add_argument("--test_csv", default="data/vpesg4k_test_2000.csv")
    parser.add_argument("--fold_dir", default="data/cv5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_root", default="checkpoints/cv5")
    parser.add_argument("--pred_dir", default="predictions/cv5")
    parser.add_argument("--submission", default="predictions/cv5_task_ensemble_submission.csv")
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
    parser.add_argument("--save_probs", action="store_true", help="append probability columns to prediction CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_predict", action="store_true")
    parser.add_argument("--force_vote", action="store_true")
    parser.add_argument(
        "--only_out",
        default="",
        help="comma-separated candidate output names to run, e.g. bge_m3",
    )
    parser.add_argument(
        "--only_alias",
        default="",
        help="comma-separated candidate aliases to run, e.g. bge",
    )
    parser.add_argument(
        "--only_tasks",
        default="",
        help="comma-separated task keys to run: promise,evidence_status,evidence_quality,timeline",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        help="continue when a model/fold fails; failed candidates are excluded from final votes",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_folds(args)

    output_root = Path(args.output_root)
    pred_dir = Path(args.pred_dir)
    (pred_dir / "folds").mkdir(parents=True, exist_ok=True)
    (pred_dir / "oof_folds").mkdir(parents=True, exist_ok=True)

    model_predictions: list[tuple[Candidate, Path]] = []
    failed: list[str] = []

    only_out = {value.strip() for value in args.only_out.split(",") if value.strip()}
    only_alias = {value.strip() for value in args.only_alias.split(",") if value.strip()}
    only_tasks = {value.strip() for value in args.only_tasks.split(",") if value.strip()}
    selected_candidates = []
    for candidate in CANDIDATES:
        if only_out and candidate.out not in only_out:
            continue
        if only_alias and candidate.alias not in only_alias:
            continue
        if only_tasks and candidate.task_key not in only_tasks:
            continue
        selected_candidates.append(candidate)
    if not selected_candidates:
        raise SystemExit("no candidates selected; check --only_out/--only_alias/--only_tasks")

    for candidate in selected_candidates:
        fold_predictions = []
        fold_oof_predictions = []
        candidate_failed = False
        for fold in range(args.folds):
            ckpt_dir = checkpoint_dir(candidate, output_root, fold)
            if not args.force_train and (ckpt_dir / "best.pt").exists():
                print(f"[skip train] {ckpt_dir}")
            else:
                train_cmd = [
                    sys.executable,
                    "train_multitask.py",
                    "--csv",
                    str(fold_train_path(Path(args.fold_dir), fold)),
                    "--val_csv",
                    str(fold_val_path(Path(args.fold_dir), fold)),
                    "--output_dir",
                    str(ckpt_dir),
                    "--model",
                    candidate.model,
                    "--pooling",
                    candidate.pooling,
                    "--epochs",
                    str(args.epochs or candidate.epochs),
                    "--batch_size",
                    str(candidate.batch_size or args.batch_size),
                    "--eval_batch_size",
                    str(candidate.eval_batch_size or args.eval_batch_size),
                    "--grad_accum",
                    str(candidate.grad_accum or args.grad_accum),
                    "--seed",
                    str(args.seed),
                    "--train_tasks",
                    candidate.task,
                    "--select_metric",
                    candidate.task,
                ]
                if args.bf16:
                    train_cmd.append("--bf16")
                if args.class_weights:
                    train_cmd.append("--class_weights")
                if candidate.freeze_backbone:
                    train_cmd.append("--freeze_backbone")
                if candidate.backbone_dtype != "auto":
                    train_cmd.extend(["--backbone_dtype", candidate.backbone_dtype])
                if candidate.max_length is not None:
                    train_cmd.extend(["--max_length", str(candidate.max_length)])
                if candidate.grad_ckpt:
                    train_cmd.append("--grad_ckpt")
                if not run_maybe(train_cmd, args.dry_run, args.skip_failed):
                    failed.append(f"{candidate.out}/{candidate.task_key}/fold{fold}/train")
                    candidate_failed = True
                    break

            val_csv = fold_oof_prediction_path(candidate, pred_dir, fold)
            if not args.force_predict and val_csv.exists():
                print(f"[skip oof predict] {val_csv}")
            else:
                oof_cmd = [
                    sys.executable,
                    "predict_multitask.py",
                    "--csv",
                    str(fold_val_path(Path(args.fold_dir), fold)),
                    "--checkpoint",
                    str(ckpt_dir / "best.pt"),
                    "--output",
                    str(val_csv),
                    "--batch_size",
                    str(args.predict_batch_size),
                ]
                if args.bf16:
                    oof_cmd.append("--bf16")
                if args.save_probs:
                    oof_cmd.append("--save_probs")
                if not run_maybe(oof_cmd, args.dry_run, args.skip_failed):
                    failed.append(f"{candidate.out}/{candidate.task_key}/fold{fold}/oof")
                    candidate_failed = True
                    break
            fold_oof_predictions.append(val_csv)

            out_csv = fold_prediction_path(candidate, pred_dir, fold)
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
                if args.save_probs:
                    pred_cmd.append("--save_probs")
                if not run_maybe(pred_cmd, args.dry_run, args.skip_failed):
                    failed.append(f"{candidate.out}/{candidate.task_key}/fold{fold}/test")
                    candidate_failed = True
                    break
            fold_predictions.append(out_csv)

        if candidate_failed:
            print(f"[skip candidate] {candidate.out}/{candidate.task_key}")
            continue

        model_val_csv = model_oof_prediction_path(candidate, pred_dir)
        if not args.force_vote and model_val_csv.exists():
            print(f"[skip OOF merge] {model_val_csv}")
        elif not args.dry_run:
            build_oof_prediction(candidate, pred_dir, fold_oof_predictions, model_val_csv)
        else:
            print(f"[dry run] build OOF {model_val_csv}")

        model_csv = model_prediction_path(candidate, pred_dir)
        if not args.force_vote and model_csv.exists():
            print(f"[skip fold vote] {model_csv}")
        else:
            vote_cmd = [
                sys.executable,
                "scripts/make_hybrid_submission.py",
                "--base",
                args.base_prediction,
                "--output",
                str(model_csv),
                "--vote",
                f"{candidate.task}={','.join(str(path) for path in fold_predictions)}",
            ]
            if not run_maybe(vote_cmd, args.dry_run, args.skip_failed):
                failed.append(f"{candidate.out}/{candidate.task_key}/fold_vote")
                continue
        model_predictions.append((candidate, model_csv))

    vote_sources = {task_key: [] for task_key in TASKS}
    for candidate, model_csv in model_predictions:
        vote_sources[candidate.task_key].append(str(model_csv))

    final_cmd = [
        sys.executable,
        "scripts/make_hybrid_submission.py",
        "--base",
        args.base_prediction,
        "--output",
        args.submission,
    ]
    for task_key, task in TASKS.items():
        sources = vote_sources[task_key]
        if sources:
            final_cmd.extend(["--vote", f"{task}={','.join(sources)}"])
    run(final_cmd, args.dry_run)

    print("\nCV candidates used:")
    for candidate in selected_candidates:
        print(f"  {candidate.task:24s} {candidate.alias:12s} {candidate.model}")
    if failed:
        print("\nFailed/skipped steps:")
        for item in failed:
            print(f"  {item}")
    print(f"\nsubmission: {args.submission}")


if __name__ == "__main__":
    main()
