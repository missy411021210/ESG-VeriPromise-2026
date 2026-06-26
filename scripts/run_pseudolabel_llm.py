"""
Run a local Hugging Face chat LLM over pseudo-label prompts.

The input is the JSONL created by scripts/make_pseudolabel_data.py
export-prompts. The output JSONL can be passed to:

    python scripts/make_pseudolabel_data.py build-csv \
        --prompts data/pseudolabel_prompts.jsonl \
        --responses data/pseudolabel_responses.jsonl \
        --output data/pseudolabeled_external.csv \
        --source_model Qwen/Qwen3-14B

This script supports resume: if the output file already contains request_id
values, those prompts are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for obj in iter_jsonl(path):
        request_id = obj.get("request_id")
        if request_id:
            done.add(str(request_id))
    return done


def render_chat(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_qwen_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    prompt = render_chat(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_qwen_thinking(text)


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = list(iter_jsonl(prompts_path))
    if args.limit:
        prompts = prompts[: args.limit]
    done_ids = load_done_ids(out_path) if args.resume else set()
    prompts = [p for p in prompts if str(p.get("request_id", "")) not in done_ids]

    if not prompts:
        print(f"nothing to do; all selected prompts already exist in {out_path}")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if args.bf16 else "auto"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("warning: CUDA is not available; Qwen3-14B CPU inference will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quantization_config = None
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load_in_4bit requires CUDA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device_map if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()

    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for prompt_obj in tqdm(prompts, desc="pseudo-label", dynamic_ncols=True):
            request_id = str(prompt_obj.get("request_id", ""))
            messages = prompt_obj["messages"]
            try:
                content = generate_one(model, tokenizer, messages, device, args)
                record = {
                    "request_id": request_id,
                    "model": args.model,
                    "content": content,
                }
            except Exception as exc:  # noqa: BLE001 - preserve progress.
                record = {
                    "request_id": request_id,
                    "model": args.model,
                    "error": repr(exc),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"wrote responses to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/pseudolabel_prompts.jsonl")
    parser.add_argument("--output", default="data/pseudolabel_responses.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--limit", type=int, default=0, help="0 means all prompts")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
