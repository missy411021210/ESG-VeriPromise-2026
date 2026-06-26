"""
Create LLM pseudo-label prompts for collected ESG paragraphs and build CSVs.

This script is for external/source-PDF paragraphs collected by
collect_esg_paragraphs.py. It does not use the test set and does not perform
manual correction. If an external LLM is used, disclose the model name/version,
purpose, and approximate contribution in the competition report.

Examples:
    python scripts/make_pseudolabel_data.py export-prompts \
        --paragraphs data/external_esg_paragraphs.csv \
        --output data/pseudolabel_prompts.jsonl \
        --limit 500

    python scripts/make_pseudolabel_data.py build-csv \
        --prompts data/pseudolabel_prompts.jsonl \
        --responses data/pseudolabel_responses.jsonl \
        --output data/pseudolabeled_external.csv \
        --source_model "Qwen3-8B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


OFFICIAL_COLUMNS = [
    "id",
    "data",
    "esg_type",
    "promise_status",
    "promise_string",
    "verification_timeline",
    "evidence_status",
    "evidence_string",
    "evidence_quality",
    "company",
    "ticker",
    "page_number",
    "pdf_url",
    "company_source",
]

LABELS = {
    "promise_status": {"No", "Yes"},
    "evidence_status": {"N/A", "No", "Yes"},
    "evidence_quality": {"N/A", "Clear", "Not Clear", "Misleading"},
    "verification_timeline": {
        "N/A",
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    },
}


def normalize_label(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "n/a", "na", "null"}:
        return "N/A"
    return text


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


def build_prompt(row: pd.Series, request_id: str) -> Dict[str, Any]:
    paragraph = str(row["paragraph"]).strip()
    user_prompt = f"""
你是 VeriPromiseESG 2026 的繁體中文 ESG 承諾驗證標註助手。請只根據下方段落標註四個任務。

段落：
{paragraph}

標籤定義：
- promise_status: 段落是否表達企業承諾。只能是 No 或 Yes。
- evidence_status: 承諾是否附有具體執行計畫或佐證。只能是 N/A、No、Yes。
- evidence_quality: 證據清晰度。只能是 N/A、Clear、Not Clear、Misleading。
- verification_timeline: 承諾時程。只能是 N/A、already、within_2_years、between_2_and_5_years、more_than_5_years。

標註規則：
- 若 promise_status 是 No，promise_string 填 "N/A"，且 verification_timeline、evidence_status、evidence_quality、evidence_string 全部填 "N/A"。
- 若 evidence_status 是 No 或 N/A，evidence_string 與 evidence_quality 必須填 "N/A"。
- promise_string 必須是段落中可直接找到的片段。
- evidence_string 必須是段落中可直接找到的片段；若沒有證據填 "N/A"。
- Misleading 表示證據看似支持承諾，但存在錯置、過度推論、避重就輕、無法支撐承諾，或與承諾不直接相關。
- 只輸出 JSON，不要 Markdown，不要額外解釋。

輸出 JSON schema：
{{
  "esg_type": "E 或 S 或 G",
  "promise_status": "...",
  "promise_string": "...",
  "verification_timeline": "...",
  "evidence_status": "...",
  "evidence_string": "...",
  "evidence_quality": "..."
}}
""".strip()
    metadata = {
        "source_pdf_url": str(row.get("source_pdf_url", "")),
        "company": str(row.get("company", "")),
        "ticker": str(row.get("ticker", "")),
        "source_page_number": int(row.get("source_page_number", 0) or 0),
        "paragraph": paragraph,
    }
    return {
        "request_id": request_id,
        "metadata": metadata,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You label Traditional Chinese ESG report paragraphs for a "
                    "promise verification classifier. Return valid JSON only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }


def export_prompts(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.paragraphs, encoding="utf-8-sig")
    if "paragraph" not in df.columns:
        raise ValueError(f"{args.paragraphs} must contain a paragraph column")
    if args.min_chars:
        df = df[df["paragraph"].astype(str).str.len() >= args.min_chars]
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, row in df.reset_index(drop=True).iterrows():
            request_id = f"pseudo-{idx + 1:06d}"
            f.write(json.dumps(build_prompt(row, request_id), ensure_ascii=False) + "\n")
    print(f"wrote {len(df)} pseudo-label prompts to {out_path}")


def extract_text_response(obj: Dict[str, Any]) -> str:
    if any(k in obj for k in LABELS):
        return json.dumps(obj, ensure_ascii=False)
    for key in ["output_text", "content", "response", "text"]:
        if isinstance(obj.get(key), str):
            return obj[key]
    message = obj.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    raise ValueError("could not find response text in object")


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def validate_labels(row: Dict[str, Any]) -> Tuple[bool, str]:
    for col in ["esg_type", *LABELS.keys()]:
        if col not in row:
            return False, f"missing {col}"
    if str(row["esg_type"]).strip() not in {"E", "S", "G"}:
        return False, "invalid esg_type"
    for col, valid in LABELS.items():
        label = normalize_label(row[col])
        if label not in valid:
            return False, f"invalid {col}={label!r}"

    promise = normalize_label(row["promise_status"])
    evidence = normalize_label(row["evidence_status"])
    quality = normalize_label(row["evidence_quality"])
    timeline = normalize_label(row["verification_timeline"])
    if promise == "No" and (evidence != "N/A" or quality != "N/A" or timeline != "N/A"):
        return False, "hierarchy violation: promise_status=No requires downstream N/A"
    if evidence in {"N/A", "No"} and quality != "N/A":
        return False, "hierarchy violation: no evidence requires evidence_quality=N/A"
    return True, "ok"


def load_prompt_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    mapping = {}
    for obj in iter_jsonl(path):
        request_id = str(obj.get("request_id", ""))
        if request_id:
            mapping[request_id] = obj.get("metadata", {})
    return mapping


def build_csv(args: argparse.Namespace) -> None:
    prompt_meta = load_prompt_metadata(Path(args.prompts))
    out_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for raw in iter_jsonl(Path(args.responses)):
        request_id = str(raw.get("request_id", ""))
        if not request_id or request_id not in prompt_meta:
            skipped.append(f"{request_id or '<missing>'}: no matching prompt metadata")
            continue
        metadata = prompt_meta[request_id]
        try:
            labels = parse_llm_json(extract_text_response(raw))
        except Exception as exc:  # noqa: BLE001 - report and keep processing.
            skipped.append(f"{request_id}: parse error: {exc}")
            continue
        ok, reason = validate_labels(labels)
        if not ok:
            skipped.append(f"{request_id}: {reason}")
            continue

        row = {col: "" for col in OFFICIAL_COLUMNS}
        row.update(
            {
                "id": args.start_id + len(out_rows),
                "data": str(metadata.get("paragraph", "")).strip(),
                "esg_type": str(labels["esg_type"]).strip(),
                "promise_status": normalize_label(labels["promise_status"]),
                "promise_string": str(labels.get("promise_string", "N/A")).strip() or "N/A",
                "verification_timeline": normalize_label(labels["verification_timeline"]),
                "evidence_status": normalize_label(labels["evidence_status"]),
                "evidence_string": str(labels.get("evidence_string", "N/A")).strip() or "N/A",
                "evidence_quality": normalize_label(labels["evidence_quality"]),
                "company": str(metadata.get("company", "external_pdf")),
                "ticker": str(metadata.get("ticker", "0000")),
                "page_number": int(metadata.get("source_page_number", 0) or 0),
                "pdf_url": str(metadata.get("source_pdf_url", "")),
                "company_source": f"PDF pseudo-label; source_model={args.source_model}",
            }
        )
        out_rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows, columns=OFFICIAL_COLUMNS)
    out_df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {len(out_df)} pseudo-labeled rows to {out_path}")
    if len(out_df):
        print(out_df[["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]].value_counts().head(20).to_string())
    if skipped:
        log_path = out_path.with_suffix(".skipped.txt")
        log_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"skipped {len(skipped)} responses; details saved to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-prompts", help="write JSONL prompts for pseudo-labeling")
    export.add_argument("--paragraphs", default="data/external_esg_paragraphs.csv")
    export.add_argument("--output", default="data/pseudolabel_prompts.jsonl")
    export.add_argument("--limit", type=int, default=0, help="0 means all rows")
    export.add_argument("--min_chars", type=int, default=80)
    export.set_defaults(func=export_prompts)

    build = sub.add_parser("build-csv", help="validate LLM responses and write official-format CSV")
    build.add_argument("--prompts", required=True)
    build.add_argument("--responses", required=True)
    build.add_argument("--output", default="data/pseudolabeled_external.csv")
    build.add_argument("--source_model", required=True)
    build.add_argument("--start_id", type=int, default=800001)
    build.set_defaults(func=build_csv)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
