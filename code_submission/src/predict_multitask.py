"""
Run inference for the VeriPromiseESG multi-task checkpoint.

Example:
    python predict_multitask.py \
        --csv data/vpesg4k_val_1000.csv \
        --checkpoint checkpoints/qwen3emb_multitask/best.pt \
        --output predictions/qwen3emb_val_predictions.csv \
        --batch_size 1 --bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from train_multitask import (
    ID2LABEL,
    LABELS,
    TASKS,
    Collator,
    MultiTaskQwen,
    apply_hierarchy,
    format_eval,
    normalize_label,
)


class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["data"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        for task in TASKS:
            item[task] = 0
        return item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwen3emb_multitask/best.pt")
    p.add_argument("--output", type=str, default="predictions/qwen3emb_predictions.csv")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="enable bf16 autocast on CUDA")
    p.add_argument(
        "--hierarchy",
        action="store_true",
        help="enable hierarchical consistency post-processing",
    )
    p.add_argument(
        "--save_probs",
        action="store_true",
        help="append per-label softmax probability columns such as evidence_quality__Clear",
    )
    return p.parse_args()

def compute_metrics(df: pd.DataFrame, preds: Dict[str, List[str]]) -> Dict[str, Any] | None:
    if not all(task in df.columns for task in TASKS):
        return None

    macro: Dict[str, float] = {}
    per_class: Dict[str, List[float]] = {}
    for task in TASKS:
        y_true = [LABELS[task].index(normalize_label(v)) for v in df[task]]
        y_pred = [LABELS[task].index(v) for v in preds[task]]
        labels = list(range(len(LABELS[task])))
        macro[task] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_class[task] = [
            float(x)
            for x in f1_score(
                y_true,
                y_pred,
                labels=labels,
                average=None,
                zero_division=0,
            )
        ]

    weights = {
        "promise_status": 0.20,
        "evidence_status": 0.30,
        "evidence_quality": 0.35,
        "verification_timeline": 0.15,
    }
    weighted = float(sum(weights[task] * macro[task] for task in TASKS))
    return {"macro": macro, "per_class": per_class, "weighted": weighted, "val_loss": 0.0}


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    model_name = train_args.get("model", "Qwen/Qwen3-Embedding-0.6B")
    max_length = args.max_length or int(train_args.get("max_length", 512))
    dropout = float(train_args.get("dropout", 0.1))
    pooling = train_args.get("pooling", "last")
    backbone_dtype = train_args.get("backbone_dtype", "float32")
    freeze_backbone = bool(train_args.get("freeze_backbone", False))

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if "data" not in df.columns:
        raise ValueError(f"{args.csv} must contain a data column")

    tokenizer_dir = checkpoint_path.parent / "tokenizer"
    tokenizer_source = str(tokenizer_dir) if tokenizer_dir.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if pooling == "last" else "right"

    ds = InferenceDataset(df, tokenizer, max_length=max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskQwen(
        model_name,
        dropout=dropout,
        pooling=pooling,
        backbone_dtype=backbone_dtype,
        freeze_backbone=freeze_backbone,
    )
    if ckpt.get("heads_only"):
        model.heads.load_state_dict(ckpt["model_state"]["heads"])
    else:
        model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    use_bf16 = bool(args.bf16 and device.type == "cuda")
    pred_ids_by_task: Dict[str, List[int]] = {task: [] for task in TASKS}
    probs_by_task: Dict[str, List[List[float]]] = {task: [] for task in TASKS}
    with torch.no_grad():
        for padded, _labels in tqdm(loader, desc="predict", dynamic_ncols=True):
            input_ids = padded["input_ids"].to(device, non_blocking=True)
            attention_mask = padded["attention_mask"].to(device, non_blocking=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            for task in TASKS:
                if args.save_probs:
                    probs_by_task[task].extend(torch.softmax(logits[task], dim=-1).cpu().tolist())
                pred_ids = logits[task].argmax(dim=-1).cpu().tolist()
                pred_ids_by_task[task].extend(pred_ids)
    if args.hierarchy:
        pred_ids_by_task = apply_hierarchy(pred_ids_by_task)
    preds = {
        task: [ID2LABEL[task][idx] for idx in pred_ids_by_task[task]]
        for task in TASKS
    }

    pred_df = df.copy()
    for task in TASKS:
        pred_df[task] = preds[task]
        if args.save_probs:
            for label_idx, label in enumerate(LABELS[task]):
                pred_df[f"{task}__{label}"] = [
                    float(row[label_idx]) for row in probs_by_task[task]
                ]
    pred_df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"predictions saved to {out_path}")

    metrics = compute_metrics(df, preds)
    if metrics is not None:
        metrics_path = out_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"weighted={metrics['weighted']:.4f}")
        print(format_eval(metrics))
        print(f"metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
