"""
Create and validate LLM-based augmentation data for VeriPromiseESG.

This script intentionally does not label or alter the test set. It supports a
two-step, auditable workflow:

1. export-prompts: write JSONL prompt requests for a chosen LLM.
2. build-csv: convert the LLM JSON/JSONL responses into a training CSV with the
   same columns as the official data.

If external LLMs are used, disclose the model name/version, purpose, and
approximate contribution in the competition report.

Examples:
    python scripts/make_llm_aug_data.py export-prompts \
        --train_csv data/vpesg_4k_train_1000.csv \
        --output data/llm_aug_prompts.jsonl \
        --targets misleading=80 within_2_years=60 evidence_no=40

    python scripts/make_llm_aug_data.py build-csv \
        --train_csv data/vpesg_4k_train_1000.csv \
        --responses data/llm_aug_responses.jsonl \
        --output data/llm_aug_candidates.csv \
        --source_model "Qwen3-8B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import random
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

DEFAULT_TARGETS = {
    "promise_no": 100,
    "not_clear": 100,
    "evidence_no_diverse": 80,
    "misleading": 80,
    "governance_misleading": 40,
    "within_2_years": 60,
    "evidence_no": 40,
}

TARGET_SPECS = {
    "promise_no": {
        "label_constraints": {
            "promise_status": "No",
            "verification_timeline": "N/A",
            "evidence_status": "N/A",
            "evidence_quality": "N/A",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須像真實永續報告內容，但不能表達未來承諾、"
            "目標、預計作為或可驗證承諾。可描述既有政策、制度介紹、過去成果、理念、風險揭露、"
            "章節背景或一般管理機制。下游欄位必須全部為 N/A。"
        ),
    },
    "not_clear": {
        "label_constraints": {
            "promise_status": "Yes",
            "evidence_status": "Yes",
            "evidence_quality": "Not Clear",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須包含企業承諾，也要有一些佐證或措施，"
            "但佐證不夠清楚：例如缺少量化基準、範圍、時程、負責單位、執行結果、覆蓋率、"
            "或只描述方向與制度而沒有足夠細節。注意這不是 Misleading；證據方向大致相關，"
            "只是清晰度不足，因此 evidence_quality 必須是 Not Clear。"
        ),
    },
    "evidence_no_diverse": {
        "label_constraints": {
            "promise_status": "Yes",
            "evidence_status": "No",
            "evidence_quality": "N/A",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須包含企業承諾，但不得提供具體佐證、數據、"
            "制度細節、時程、負責單位或執行成果。請讓主題多樣化，涵蓋環境、社會、治理，"
            "例如低碳供應鏈、員工友善職場、資訊透明、風險管理、人權政策、資源循環等。"
        ),
    },
    "misleading": {
        "label_constraints": {
            "promise_status": "Yes",
            "evidence_status": "Yes",
            "evidence_quality": "Misleading",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須包含企業承諾，也必須包含看似支持承諾的證據，"
            "但證據與承諾之間存在誤導性、錯置、過度推論、避重就輕或無法支撐結論的問題。"
        ),
    },
    "within_2_years": {
        "label_constraints": {
            "promise_status": "Yes",
            "verification_timeline": "within_2_years",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須包含明確承諾，且時程應落在兩年內，"
            "例如明年、2025 年、2026 年底前、未來 18 個月內等短期承諾。"
        ),
    },
    "governance_misleading": {
        "label_constraints": {
            "promise_status": "Yes",
            "evidence_status": "Yes",
            "evidence_quality": "Misleading",
            "verification_timeline": "already",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落，主題必須是公司治理、董事會或高階管理階層薪酬，"
            "並描述薪酬、限制員工權利新股、獎酬制度或 KPI 與 ESG 指標連結。段落必須看似有制度"
            "或佐證，但實際上只說明薪酬治理機制、指標名稱或程序，無法直接證明永續承諾已被驗證，"
            "因此 evidence_quality 必須是 Misleading。"
        ),
    },
    "evidence_no": {
        "label_constraints": {
            "promise_status": "Yes",
            "evidence_status": "No",
            "evidence_quality": "N/A",
        },
        "instruction": (
            "產生一段繁體中文 ESG 報告段落。段落必須包含企業承諾，但不能提供具體執行方案、數據、"
            "制度、時程、負責單位或可驗證佐證。"
        ),
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


def parse_targets(raw_targets: List[str] | None) -> Dict[str, int]:
    if not raw_targets:
        return dict(DEFAULT_TARGETS)
    parsed: Dict[str, int] = {}
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError(f"target must use name=count format, got: {raw}")
        name, count = raw.split("=", 1)
        name = name.strip()
        if name not in TARGET_SPECS:
            raise ValueError(f"unknown target {name!r}; valid targets: {sorted(TARGET_SPECS)}")
        parsed[name] = int(count)
    return parsed


def clean_training_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in OFFICIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing official columns: {missing}")
    for col in LABELS:
        df[col] = df[col].map(normalize_label)
    return df[OFFICIAL_COLUMNS].copy()


def examples_for_target(df: pd.DataFrame, target: str, n: int, rng: random.Random) -> List[Dict[str, str]]:
    spec = TARGET_SPECS[target]["label_constraints"]
    mask = pd.Series(True, index=df.index)
    for col, label in spec.items():
        mask &= df[col].map(normalize_label).eq(label)
    subset = df[mask]

    must_include = pd.DataFrame()
    if target == "misleading" and not subset.empty:
        must_include = subset.head(1)

    # Misleading has only one row in the released training split, so include
    # close negatives to teach the boundary without requiring more positives.
    if len(subset) < n and target == "misleading":
        close = df[
            df["promise_status"].eq("Yes")
            & df["evidence_status"].eq("Yes")
            & df["evidence_quality"].isin(["Clear", "Not Clear"])
        ]
        subset = pd.concat([subset, close], ignore_index=True)

    if subset.empty:
        subset = df[df["promise_status"].eq("Yes")]
    sample_size = min(max(0, n - len(must_include)), len(subset))
    rows = subset.sample(n=sample_size, random_state=rng.randint(0, 10_000_000))
    if not must_include.empty:
        rows = pd.concat([must_include, rows], ignore_index=True).drop_duplicates(subset=["data"]).head(n)
    return [
        {
            "data": str(row["data"]),
            "promise_status": normalize_label(row["promise_status"]),
            "evidence_status": normalize_label(row["evidence_status"]),
            "evidence_quality": normalize_label(row["evidence_quality"]),
            "verification_timeline": normalize_label(row["verification_timeline"]),
        }
        for _, row in rows.iterrows()
    ]


def build_prompt(target: str, request_id: str, examples: List[Dict[str, str]]) -> Dict[str, Any]:
    spec = TARGET_SPECS[target]
    labels_json = json.dumps(spec["label_constraints"], ensure_ascii=False)
    examples_json = json.dumps(examples, ensure_ascii=False, indent=2)
    user_prompt = f"""
你是 VeriPromiseESG 2026 資料擴增助手。請產生 1 筆新的、非真實公司、非照抄範例的繁體中文 ESG 段落與標籤。

目標類型：{target}
目標說明：{spec["instruction"]}
必須符合的標籤限制：{labels_json}

可用標籤：
- promise_status: No, Yes
- evidence_status: N/A, No, Yes
- evidence_quality: N/A, Clear, Not Clear, Misleading
- verification_timeline: N/A, already, within_2_years, between_2_and_5_years, more_than_5_years

請遵守：
- 不要使用真實公司名稱、真實網址、真實股票代號。
- 不要複製或只改寫範例句子。
- data 必須自然像永續報告段落，長度約 80 到 220 個中文字。
- promise_string 與 evidence_string 必須是 data 內可找到的片段；若不適用請填 "N/A"。
- 若 promise_status 為 "No"，下游欄位必須為 "N/A"。
- 若 evidence_status 為 "No" 或 "N/A"，evidence_quality 必須為 "N/A"。
- 只輸出 JSON，不要 Markdown，不要額外解釋。

輸出 JSON schema：
{{
  "data": "...",
  "esg_type": "E 或 S 或 G",
  "promise_status": "...",
  "promise_string": "...",
  "verification_timeline": "...",
  "evidence_status": "...",
  "evidence_string": "...",
  "evidence_quality": "..."
}}

參考範例：
{examples_json}
""".strip()

    return {
        "request_id": request_id,
        "target": target,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate auditable synthetic training data for a Traditional Chinese "
                    "ESG promise verification classifier. Return valid JSON only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }


def export_prompts(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    df = clean_training_df(Path(args.train_csv))
    targets = parse_targets(args.targets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for target, count in targets.items():
            for idx in range(count):
                request_id = f"{target}-{idx + 1:04d}"
                examples = examples_for_target(df, target, args.examples_per_prompt, rng)
                prompt = build_prompt(target, request_id, examples)
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"wrote {n_written} prompt requests to {out_path}")
    print("next: send each messages array to your chosen LLM and save JSONL responses")


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


def extract_text_response(obj: Dict[str, Any]) -> str:
    if all(k in obj for k in ["data", "promise_status", "evidence_quality"]):
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


def validate_aug_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    for col in ["data", "esg_type", *LABELS.keys()]:
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
    if len(str(row["data"]).strip()) < 20:
        return False, "data too short"
    return True, "ok"


def build_csv(args: argparse.Namespace) -> None:
    train_df = clean_training_df(Path(args.train_csv))
    existing_texts = set(train_df["data"].astype(str).str.strip())
    out_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen_texts = set()

    for raw in iter_jsonl(Path(args.responses)):
        request_id = str(raw.get("request_id", f"line-{len(out_rows) + len(skipped) + 1}"))
        try:
            parsed = parse_llm_json(extract_text_response(raw))
        except Exception as exc:  # noqa: BLE001 - report and keep processing.
            skipped.append(f"{request_id}: parse error: {exc}")
            continue

        ok, reason = validate_aug_row(parsed)
        if not ok:
            skipped.append(f"{request_id}: {reason}")
            continue

        data_text = str(parsed["data"]).strip()
        if data_text in existing_texts:
            skipped.append(f"{request_id}: duplicate of official train row")
            continue
        if data_text in seen_texts:
            skipped.append(f"{request_id}: duplicate synthetic row")
            continue
        seen_texts.add(data_text)

        row = {col: "" for col in OFFICIAL_COLUMNS}
        row.update(
            {
                "id": args.start_id + len(out_rows),
                "data": data_text,
                "esg_type": str(parsed["esg_type"]).strip(),
                "promise_status": normalize_label(parsed["promise_status"]),
                "promise_string": str(parsed.get("promise_string", "N/A")).strip() or "N/A",
                "verification_timeline": normalize_label(parsed["verification_timeline"]),
                "evidence_status": normalize_label(parsed["evidence_status"]),
                "evidence_string": str(parsed.get("evidence_string", "N/A")).strip() or "N/A",
                "evidence_quality": normalize_label(parsed["evidence_quality"]),
                "company": "llm_aug",
                "ticker": "0000",
                "page_number": 0,
                "pdf_url": "",
                "company_source": f"LLM synthetic augmentation; source_model={args.source_model}",
            }
        )
        out_rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows, columns=OFFICIAL_COLUMNS)
    out_df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"wrote {len(out_df)} validated rows to {out_path}")
    if len(out_df):
        print(out_df[["evidence_quality", "verification_timeline", "evidence_status"]].value_counts().to_string())
    if skipped:
        log_path = out_path.with_suffix(".skipped.txt")
        log_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"skipped {len(skipped)} responses; details saved to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-prompts", help="write JSONL prompt requests for LLM augmentation")
    export.add_argument("--train_csv", default="data/vpesg_4k_train_1000.csv")
    export.add_argument("--output", default="data/llm_aug_prompts.jsonl")
    export.add_argument("--targets", nargs="*", help="target=count pairs, e.g. misleading=80")
    export.add_argument("--examples_per_prompt", type=int, default=3)
    export.add_argument("--seed", type=int, default=42)
    export.set_defaults(func=export_prompts)

    build = sub.add_parser("build-csv", help="validate LLM responses and write augmentation CSV")
    build.add_argument("--train_csv", default="data/vpesg_4k_train_1000.csv")
    build.add_argument("--responses", required=True)
    build.add_argument("--output", default="data/llm_aug_candidates.csv")
    build.add_argument("--source_model", required=True)
    build.add_argument("--start_id", type=int, default=900001)
    build.set_defaults(func=build_csv)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
