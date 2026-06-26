#!/usr/bin/env python
"""Run single-task fine-tuning experiments for multiple backbones.

This is a thin orchestrator around train_multitask.py.  It trains one task head
at a time and selects the best checkpoint by that task's validation macro-F1.
Existing checkpoints are skipped by default.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


MODELS = {
    "qwen": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "out": "qwen3emb",
        "pooling": "last",
    },
    "qwen3emb4b": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "out": "qwen3emb4b",
        "pooling": "last",
        "large": True,
    },
    "qwen3emb8b": {
        "name": "Qwen/Qwen3-Embedding-8B",
        "out": "qwen3emb8b",
        "pooling": "last",
        "large": True,
    },
    "bge": {
        "name": "BAAI/bge-m3",
        "out": "bge_m3",
        "pooling": "mean",
    },
    "e5": {
        "name": "intfloat/multilingual-e5-large",
        "out": "multilingual_e5_large",
        "pooling": "mean",
    },
    "e5_instruct": {
        "name": "intfloat/multilingual-e5-large-instruct",
        "out": "multilingual_e5_large_instruct",
        "pooling": "mean",
    },
    "gte_qwen_1_5b": {
        "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "out": "gte_qwen2_1_5b_instruct",
        "pooling": "last",
        "large": True,
    },
    "gte_qwen_7b": {
        "name": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "out": "gte_qwen2_7b_instruct",
        "pooling": "last",
        "large": True,
    },
    "roberta": {
        "name": "hfl/chinese-roberta-wwm-ext-large",
        "out": "chinese_roberta_wwm_ext_large",
        "pooling": "cls",
    },
    "macbert": {
        "name": "hfl/chinese-macbert-large",
        "out": "chinese_macbert_large",
        "pooling": "cls",
    },
    "lert": {
        "name": "hfl/chinese-lert-large",
        "out": "chinese_lert_large",
        "pooling": "cls",
    },
    "electra": {
        "name": "hfl/chinese-electra-180g-large-discriminator",
        "out": "chinese_electra_180g_large_discriminator",
        "pooling": "cls",
    },
    "mdeberta": {
        "name": "microsoft/mdeberta-v3-base",
        "out": "mdeberta_v3_base",
        "pooling": "cls",
    },
    "xlmr": {
        "name": "FacebookAI/xlm-roberta-large",
        "out": "xlm_roberta_large",
        "pooling": "cls",
    },
    "ernie": {
        "name": "nghuyong/ernie-3.0-base-zh",
        "out": "ernie_3_base_zh",
        "pooling": "cls",
    },
    "mengzi": {
        "name": "Langboat/mengzi-bert-base",
        "out": "mengzi_bert_base",
        "pooling": "cls",
    },
    "deberta_zh": {
        "name": "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese",
        "out": "erlangshen_deberta_v2_710m_chinese",
        "pooling": "cls",
        "large": True,
    },
    "glm4_9b_0414": {
        "name": "THUDM/GLM-4-9B-0414",
        "out": "glm4_9b_0414",
        "pooling": "last",
        "large": True,
    },
    "glm4_9b_chat": {
        "name": "THUDM/GLM-4-9B-Chat",
        "out": "glm4_9b_chat",
        "pooling": "last",
        "large": True,
    },
}

TASKS = {
    "promise": "promise_status",
    "evidence_status": "evidence_status",
    "evidence_quality": "evidence_quality",
    "timeline": "verification_timeline",
}


def split_csv_arg(value: str, choices: dict[str, object]) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = sorted(set(items) - set(choices))
    if unknown:
        raise SystemExit(f"unknown values: {', '.join(unknown)}")
    return items


def split_task_arg(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = sorted(set(items) - set(TASKS))
    if unknown:
        raise SystemExit(f"unknown task values: {', '.join(unknown)}")
    return items


def split_model_arg(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise SystemExit("at least one model is required")
    unknown = [item for item in items if item not in MODELS and "/" not in item]
    if unknown:
        raise SystemExit(
            "unknown model aliases: "
            + ", ".join(unknown)
            + ". Use a known alias or a Hugging Face model id like org/model."
        )
    return items


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").lower()


def resolve_model(model_key: str) -> dict[str, object]:
    if model_key in MODELS:
        return MODELS[model_key]
    return {
        "name": model_key,
        "out": sanitize_model_id(model_key),
        "pooling": "last",
        "large": True,
    }


def best_record(history_path: Path, task: str) -> dict[str, object] | None:
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return None
    return max(history, key=lambda row: row.get("macro", {}).get(task, -1.0))


def print_summary(results: list[dict[str, object]]) -> None:
    if not results:
        return

    print()
    print("=" * 88)
    print("Single-task validation summary")
    print("=" * 88)
    print(
        f"{'task':24s} {'model':36s} {'score':>8s} {'epoch':>5s} {'checkpoint'}"
    )
    print("-" * 120)
    for row in sorted(results, key=lambda x: (str(x["task_key"]), str(x["model_key"]))):
        score = row.get("score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "N/A"
        epoch = row.get("epoch", "N/A")
        print(
            f"{str(row['task']):24s} {str(row['model_key']):36s} "
            f"{score_text:>8s} {str(epoch):>5s} {row['output_dir']}"
        )

    print()
    print("Best by task")
    print("-" * 120)
    for task_key in TASKS:
        task_rows = [
            row for row in results
            if row["task_key"] == task_key and isinstance(row.get("score"), float)
        ]
        if not task_rows:
            continue
        best = max(task_rows, key=lambda row: row["score"])
        print(
            f"{str(best['task']):24s} {str(best['model_key']):36s} "
            f"{best['score']:.4f}  epoch={best['epoch']}  {best['output_dir']}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/vpesg_4k_train_1000.csv")
    p.add_argument("--val_csv", default="data/vpesg4k_val_1000.csv")
    p.add_argument("--output_root", default="checkpoints")
    p.add_argument("--models", default="qwen,bge,e5,roberta")
    p.add_argument("--tasks", default="promise,evidence_status,evidence_quality,timeline")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--class_weights", action="store_true")
    p.add_argument(
        "--safe_large",
        action="store_true",
        help="freeze large backbones and load them in bfloat16 to reduce VRAM use",
    )
    p.add_argument(
        "--group_by_model",
        action="store_true",
        help="save as OUTPUT_ROOT/MODEL_ALIAS/TASK instead of OUTPUT_ROOT/single_MODEL_TASK",
    )
    p.add_argument("--force", action="store_true", help="rerun even if history.json and best.pt exist")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    model_keys = split_model_arg(args.models)
    task_keys = split_task_arg(args.tasks)
    results: list[dict[str, object]] = []

    for model_key in model_keys:
        model_cfg = resolve_model(model_key)
        for task_key in task_keys:
            task = TASKS[task_key]
            if args.group_by_model:
                output_dir = Path(args.output_root) / str(model_cfg["out"]) / task_key
            else:
                output_dir = Path(args.output_root) / f"single_{model_cfg['out']}_{task_key}"
            if (
                not args.force
                and (output_dir / "history.json").exists()
                and (output_dir / "best.pt").exists()
            ):
                print(f"[skip] {output_dir}")
                record = best_record(output_dir / "history.json", task)
                results.append({
                    "model_key": model_key,
                    "task_key": task_key,
                    "task": task,
                    "output_dir": str(output_dir),
                    "score": record.get("macro", {}).get(task) if record else None,
                    "epoch": record.get("epoch") if record else None,
                })
                continue

            cmd = [
                sys.executable,
                "train_multitask.py",
                "--csv",
                args.csv,
                "--val_csv",
                args.val_csv,
                "--output_dir",
                str(output_dir),
                "--model",
                str(model_cfg["name"]),
                "--pooling",
                str(model_cfg["pooling"]),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--grad_accum",
                str(args.grad_accum),
                "--seed",
                str(args.seed),
                "--train_tasks",
                task,
                "--select_metric",
                task,
            ]
            if args.bf16:
                cmd.append("--bf16")
            if args.class_weights:
                cmd.append("--class_weights")
            if args.safe_large and model_cfg.get("large"):
                cmd.extend(["--freeze_backbone", "--backbone_dtype", "bfloat16"])

            print("\n" + "=" * 88)
            print(" ".join(cmd))
            print("=" * 88)
            if not args.dry_run:
                subprocess.run(cmd, check=True)

            record = best_record(output_dir / "history.json", task)
            results.append({
                "model_key": model_key,
                "task_key": task_key,
                "task": task,
                "output_dir": str(output_dir),
                "score": record.get("macro", {}).get(task) if record else None,
                "epoch": record.get("epoch") if record else None,
            })

    print_summary(results)


if __name__ == "__main__":
    main()
