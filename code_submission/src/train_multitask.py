"""
VeriPromiseESG 2026 — multi-task fine-tuning script.

Backbone : Qwen/Qwen3-Embedding-0.6B (last-token pooling, left padding).
Input    : the `data` column of the training CSV.
Tasks    : 4 single-label classification heads, equal-weighted CE losses.

  ┌────────────────────────┬──────────────────────────────────────────────────────────┐
  │ head                   │ classes                                                  │
  ├────────────────────────┼──────────────────────────────────────────────────────────┤
  │ promise_status         │ No, Yes                                                  │
  │ evidence_status        │ N/A, No, Yes                                             │
  │ evidence_quality       │ N/A, Clear, Not Clear, Misleading            (clarity)   │
  │ verification_timeline  │ N/A, already, within_2_years,                            │
  │                        │ between_2_and_5_years, more_than_5_years                 │
  └────────────────────────┴──────────────────────────────────────────────────────────┘


Typical run:
    python train_multitask.py \
        --csv data/vpesg_4k_train_1000.csv \
        --val_csv data/vpesg4k_val_1000.csv \
        --output_dir checkpoints/qwen3emb_multitask \
        --epochs 5 --batch_size 8 --grad_accum 2 --bf16
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ----------------------------- Label maps -----------------------------

LABELS: Dict[str, List[str]] = {
    "promise_status":        ["No", "Yes"],
    "evidence_status":       ["N/A", "No", "Yes"],
    "evidence_quality":      ["N/A", "Clear", "Not Clear", "Misleading"],
    "verification_timeline": ["N/A", "already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}
TASKS: List[str] = list(LABELS.keys())
LABEL2ID = {t: {l: i for i, l in enumerate(ls)} for t, ls in LABELS.items()}
ID2LABEL = {t: {i: l for i, l in enumerate(ls)} for t, ls in LABELS.items()}

# Competition scoring weights (CLAUDE.md §2). Used for the headline metric
# and best-checkpoint selection — not for the training loss (loss is equal-weighted).
TASK_WEIGHTS: Dict[str, float] = {
    "promise_status":        0.20,
    "evidence_status":       0.30,
    "evidence_quality":      0.35,
    "verification_timeline": 0.15,
}


def normalize_label(val: Any) -> str:
    """Map NaN / empty / variants to the canonical 'N/A' token."""
    if val is None:
        return "N/A"
    try:
        if pd.isna(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a"}:
        return "N/A"
    return s

def apply_hierarchy(preds: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Enforce task consistency implied by the competition labels."""
    fixed = {t: list(vals) for t, vals in preds.items()}
    for i in range(len(fixed["promise_status"])):
        promise = ID2LABEL["promise_status"][fixed["promise_status"][i]]
        evidence = ID2LABEL["evidence_status"][fixed["evidence_status"][i]]

        if promise == "No":
            fixed["evidence_status"][i] = LABEL2ID["evidence_status"]["N/A"]
            fixed["evidence_quality"][i] = LABEL2ID["evidence_quality"]["N/A"]
            fixed["verification_timeline"][i] = LABEL2ID["verification_timeline"]["N/A"]
            continue

        if evidence in {"N/A", "No"}:
            fixed["evidence_quality"][i] = LABEL2ID["evidence_quality"]["N/A"]
    return fixed


def build_loss_fns(
    df: pd.DataFrame,
    device: torch.device,
    use_class_weights: bool,
    tasks: List[str] | None = None,
) -> Dict[str, nn.Module]:
    """Build per-task class-weighted CE losses for macro-F1 oriented training."""
    losses: Dict[str, nn.Module] = {}
    tasks = tasks or TASKS
    for task in tasks:
        if not use_class_weights:
            losses[task] = nn.CrossEntropyLoss()
            continue

        label_ids = torch.tensor(
            [LABEL2ID[task][normalize_label(v)] for v in df[task]],
            dtype=torch.long,
        )
        counts = torch.bincount(label_ids, minlength=len(LABELS[task])).float()
        weights = counts.sum() / (counts.clamp_min(1.0) * len(LABELS[task]))
        weights = weights / weights.mean()
        losses[task] = nn.CrossEntropyLoss(weight=weights.to(device))
        print(
            f"  {task:24s} class weights: "
            + ", ".join(f"{label}={weights[i]:.2f}" for i, label in enumerate(LABELS[task]))
        )
    return losses


def multitask_loss(
    logits: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    loss_fns: Dict[str, nn.Module],
    use_task_weights: bool,
    evidence_quality_weight: float,
    tasks: List[str] | None = None,
) -> torch.Tensor:
    tasks = tasks or TASKS
    task_scales = {t: 1.0 for t in tasks}
    task_scales["evidence_quality"] = evidence_quality_weight
    if use_task_weights:
        return len(tasks) * sum(
            task_scales[t] * TASK_WEIGHTS[t] * loss_fns[t](logits[t], target[t])
            for t in tasks
        )
    return sum(task_scales[t] * loss_fns[t](logits[t], target[t]) for t in tasks)


# ----------------------------- Dataset --------------------------------
class ESGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.texts = df["data"].astype(str).tolist()
        self.label_ids = {
            t: [LABEL2ID[t][normalize_label(v)] for v in df[t]] for t in TASKS
        }
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
        for t in TASKS:
            item[t] = self.label_ids[t][idx]
        return item

@dataclass
class Collator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, batch: List[Dict[str, Any]]):
        feats = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        padded = self.tokenizer.pad(
            feats,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = {t: torch.tensor([b[t] for b in batch], dtype=torch.long) for t in TASKS}
        return padded, labels


# ----------------------------- Model ----------------------------------

def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Qwen3-Embedding convention: last non-pad token. Works for left- or right-padding.
    left_padded = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    seq_idx = attention_mask.sum(dim=1) - 1
    bsz = last_hidden.shape[0]
    return last_hidden[torch.arange(bsz, device=last_hidden.device), seq_idx]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "last":
        return last_token_pool(last_hidden, attention_mask)
    if pooling == "mean":
        return mean_pool(last_hidden, attention_mask)
    if pooling == "cls":
        return last_hidden[:, 0]
    raise ValueError(f"unknown pooling: {pooling}")


class MultiTaskQwen(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        pooling: str = "last",
        backbone_dtype: str = "float32",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if backbone_dtype not in dtype_map:
            raise ValueError(f"unknown backbone_dtype: {backbone_dtype}")
        self.pooling = pooling
        self.force_position_ids = "gte-multilingual-base" in model_name.lower()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "gte-Qwen2" in model_name and not hasattr(config, "rope_theta"):
            # Some transformers versions drop this custom config field, while
            # Alibaba's remote Qwen2 model code expects it during init.
            config.rope_theta = 1_000_000.0
        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
        }
        if backbone_dtype != "auto":
            model_kwargs["torch_dtype"] = dtype_map[backbone_dtype]
        self.backbone = AutoModel.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            t: nn.Linear(hidden, len(LABELS[t])) for t in TASKS
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.force_position_ids:
            seq_len = input_ids.size(1)
            kwargs["position_ids"] = torch.arange(
                seq_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(input_ids.size(0), -1)
        out = self.backbone(**kwargs)
        pooled = pool_hidden(out.last_hidden_state, attention_mask, self.pooling)
        pooled = self.dropout(pooled)
        pooled = pooled.to(next(self.heads.parameters()).dtype)
        return {t: head(pooled) for t, head in self.heads.items()}


# ----------------------------- Train utils ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
#val
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fns: Dict[str, nn.Module],
    use_bf16: bool,
    use_hierarchy: bool,
    use_task_weights: bool,
    evidence_quality_weight: float,
    train_tasks: List[str] | None = None,
) -> Dict[str, Any]:
    model.eval()
    preds = {t: [] for t in TASKS}
    targs = {t: [] for t in TASKS}
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  val", leave=False, dynamic_ncols=True)
    for padded, labels in pbar:
        input_ids = padded["input_ids"].to(device, non_blocking=True)
        attention_mask = padded["attention_mask"].to(device, non_blocking=True)
        target = {t: labels[t].to(device, non_blocking=True) for t in TASKS}

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = multitask_loss(
                    logits,
                    target,
                    loss_fns,
                    use_task_weights,
                    evidence_quality_weight,
                    tasks=train_tasks,
                )
        else:
            logits = model(input_ids, attention_mask)
            loss = multitask_loss(
                logits,
                target,
                loss_fns,
                use_task_weights,
                evidence_quality_weight,
                tasks=train_tasks,
            )

        total_loss += float(loss.detach())
        n_batches += 1
        for t in TASKS:
            preds[t].extend(logits[t].argmax(dim=-1).cpu().tolist())
            targs[t].extend(labels[t].tolist())

    if use_hierarchy:
        preds = apply_hierarchy(preds)

    val_loss = total_loss / max(1, n_batches)
    macro: Dict[str, float] = {}
    per_class: Dict[str, List[float]] = {}
    for t in TASKS:
        n_classes = len(LABELS[t])
        macro[t] = float(f1_score(targs[t], preds[t], average="macro", zero_division=0))
        per_class[t] = [
            float(x) for x in f1_score(
                targs[t], preds[t],
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            )
        ]
    weighted = float(sum(TASK_WEIGHTS[t] * macro[t] for t in TASKS))
    return {"val_loss": val_loss, "macro": macro, "per_class": per_class, "weighted": weighted}

def format_eval(metrics: Dict[str, Any]) -> str:
    lines = []
    for t in TASKS:
        cls = LABELS[t]
        pcs = "  ".join(f"{c}={s:.3f}" for c, s in zip(cls, metrics["per_class"][t]))
        lines.append(f"  {t:24s} macro={metrics['macro'][t]:.4f}  |  {pcs}")
    return "\n".join(lines)


def parse_task_list(raw: str) -> List[str]:
    tasks = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in tasks if x not in TASKS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown tasks: {bad}; choose from {TASKS}")
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    return tasks


# ----------------------------- Main -----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/vpesg_4k_train_1000.csv")
    p.add_argument("--val_csv", type=str, default="data/vpesg4k_val_1000.csv")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output_dir", type=str, default="checkpoints/qwen3emb_multitask")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5, help="backbone learning rate")
    p.add_argument("--head_lr", type=float, default=1e-4, help="classification head learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=0, help="early-stop after N non-improving epochs; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="enable bf16 autocast on CUDA")
    p.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing")
    p.add_argument(
        "--pooling",
        choices=["last", "mean", "cls"],
        default="last",
        help="pooling strategy for the encoder hidden states",
    )
    p.add_argument(
        "--backbone_dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
        help="dtype used when loading the backbone; float32 is safest for full fine-tuning",
    )
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="train only the classification heads; useful for large encoders on limited VRAM",
    )
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--train_tasks",
        type=parse_task_list,
        default=TASKS,
        help="comma-separated task heads to train, e.g. evidence_quality",
    )
    p.add_argument(
        "--select_metric",
        choices=["weighted", *TASKS],
        default="weighted",
        help="metric used to choose best.pt; use the task name for single-task runs",
    )
    p.add_argument(
        "--task_loss_weights",
        action="store_true",
        help="weight task losses by official scoring weights",
    )
    p.add_argument(
        "--class_weights",
        action="store_true",
        help="use inverse-frequency class weights for each task",
    )
    p.add_argument(
        "--evidence_quality_weight",
        type=float,
        default=1.0,
        help="multiply the evidence_quality training loss by this value",
    )
    p.add_argument(
        "--hierarchy",
        action="store_true",
        help="enable hierarchical consistency post-processing during validation",
    )
    p.add_argument(
        "--no_eval_save_last",
        action="store_true",
        help="disable validation and save the final epoch as best.pt; useful for final train+val training",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_tasks: List[str] = args.train_tasks
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    missing = [c for c in ["data", *TASKS] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {args.csv}: {missing}")
    for t in TASKS:
        df[t] = df[t].map(normalize_label)

    val_df = None
    if args.no_eval_save_last:
        train_df = df
        print("validation disabled: saving final epoch as best.pt")
    elif args.val_csv:
        val_df = pd.read_csv(args.val_csv, encoding="utf-8-sig")
        missing = [c for c in ["data", *TASKS] if c not in val_df.columns]
        if missing:
            raise ValueError(f"missing columns in {args.val_csv}: {missing}")
        for t in TASKS:
            val_df[t] = val_df[t].map(normalize_label)
        train_df = df
        print(f"using fixed validation set: {args.val_csv}")
    else:
        # stratify on promise_status — binary, always populated
        train_df, val_df = train_test_split(
            df,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=df["promise_status"],
        )
    print(f"train={len(train_df)}  val={0 if val_df is None else len(val_df)}")
    print(f"training tasks: {', '.join(train_tasks)}")
    print(
        "checkpoint selection: final epoch"
        if args.no_eval_save_last
        else f"best checkpoint metric: {args.select_metric}"
    )
    for t in TASKS:
        dist = train_df[t].value_counts().to_dict()
        print(f"  {t:24s} train dist: {dist}")

    # ---- tokenizer / model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if args.pooling == "last" else "right"

    train_ds = ESGDataset(train_df, tokenizer, max_length=args.max_length)
    collator = Collator(tokenizer)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=pin_mem,
    )
    val_loader = None
    if val_df is not None:
        val_ds = ESGDataset(val_df, tokenizer, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers, pin_memory=pin_mem,
        )

    model = MultiTaskQwen(
        args.model,
        dropout=args.dropout,
        pooling=args.pooling,
        backbone_dtype=args.backbone_dtype,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "config"):
            model.backbone.config.use_cache = False

    # ---- optimizer ----
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    backbone_params = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.heads.named_parameters())
    optim_groups = []
    if backbone_params:
        optim_groups.extend([
            {"params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
             "lr": args.lr, "weight_decay": 0.0},
        ])
    optim_groups.extend([
        {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
         "lr": args.head_lr, "weight_decay": 0.0},
    ])
    optimizer = torch.optim.AdamW(optim_groups)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    if args.class_weights:
        print("class-weighted losses:")
    loss_fns = build_loss_fns(
        train_df,
        device,
        use_class_weights=args.class_weights,
        tasks=train_tasks,
    )

    use_bf16 = bool(args.bf16 and device.type == "cuda")
    best_weighted = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    epochs_without_improvement = 0

    def save_checkpoint(epoch: int, metrics: Dict[str, Any] | None = None) -> None:
        model_state = (
            {"heads": model.heads.state_dict()}
            if args.freeze_backbone
            else model.state_dict()
        )
        ckpt = {
            "model_state": model_state,
            "heads_only": bool(args.freeze_backbone),
            "labels": LABELS,
            "task_weights": TASK_WEIGHTS,
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out_dir / "best.pt")
        tokenizer.save_pretrained(out_dir / "tokenizer")

    # ---- train ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )
        for step, (padded, labels) in enumerate(pbar):
            input_ids = padded["input_ids"].to(device, non_blocking=True)
            attention_mask = padded["attention_mask"].to(device, non_blocking=True)
            target = {t: labels[t].to(device, non_blocking=True) for t in TASKS}

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask)
                    loss = multitask_loss(
                        logits,
                        target,
                        loss_fns,
                        args.task_loss_weights,
                        args.evidence_quality_weight,
                        tasks=train_tasks,
                    )
            else:
                logits = model(input_ids, attention_mask)
                loss = multitask_loss(
                    logits,
                    target,
                    loss_fns,
                    args.task_loss_weights,
                    args.evidence_quality_weight,
                    tasks=train_tasks,
                )

            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach())
            n_steps += 1
            pbar.set_postfix(train_loss=f"{running / n_steps:.4f}")

        train_loss = running / max(1, n_steps)
        if args.no_eval_save_last:
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "select_metric": "final_epoch",
                "select_score": None,
            }
            history.append(record)
            print(f"[epoch {epoch}] train_loss={train_loss:.4f}")
            continue

        assert val_loader is not None
        metrics = evaluate(
            model,
            val_loader,
            device,
            loss_fns,
            use_bf16,
            use_hierarchy=args.hierarchy,
            use_task_weights=args.task_loss_weights,
            evidence_quality_weight=args.evidence_quality_weight,
            train_tasks=train_tasks,
        )
        select_score = (
            metrics["weighted"]
            if args.select_metric == "weighted"
            else metrics["macro"][args.select_metric]
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "weighted": metrics["weighted"],
            "select_metric": args.select_metric,
            "select_score": select_score,
            "macro": metrics["macro"],
            "per_class": metrics["per_class"],
        }
        history.append(record)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f}  "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"weighted={metrics['weighted']:.4f}  "
            f"{args.select_metric}={select_score:.4f}"
        )
        print(format_eval(metrics))

        if select_score > best_weighted:
            best_weighted = select_score
            best_epoch = epoch
            best_metrics = metrics
            save_checkpoint(epoch, metrics)
            print(
                f"  ↳ new best ({args.select_metric}={best_weighted:.4f}) "
                f"saved to {out_dir / 'best.pt'}"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(
                    f"  no improvement for {epochs_without_improvement}/"
                    f"{args.patience} epochs"
                )
                if epochs_without_improvement >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break

    if args.no_eval_save_last:
        best_epoch = history[-1]["epoch"] if history else args.epochs
        best_weighted = float("nan")
        save_checkpoint(int(best_epoch), None)
        print(f"final epoch checkpoint saved to {out_dir / 'best.pt'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 72)
    if args.no_eval_save_last:
        print(f"saved final epoch={best_epoch}  validation=disabled")
    else:
        print(f"best epoch={best_epoch}  {args.select_metric}={best_weighted:.4f}")
    if best_metrics:
        print(format_eval(best_metrics))
    print(f"checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
