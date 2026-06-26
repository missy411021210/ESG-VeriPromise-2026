"""
Collect candidate ESG paragraphs from official-source sustainability PDFs.

The default inputs are the released train/validation CSV files. This script does
not read or label the test set. It downloads each unique pdf_url into a local
cache, extracts page text, splits text into paragraph-like chunks, and keeps
chunks with ESG promise keywords for later LLM pseudo-labeling.

Dependency:
    pip install pypdf

Example:
    python scripts/collect_esg_paragraphs.py \
        --csvs data/vpesg_4k_train_1000.csv data/vpesg4k_val_1000.csv \
        --output data/external_esg_paragraphs.csv \
        --cache_dir data/pdf_cache \
        --max_pdfs 10
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


DEFAULT_KEYWORDS = [
    "承諾",
    "目標",
    "預計",
    "預期",
    "將於",
    "未來",
    "持續",
    "致力於",
    "推動",
    "逐步",
    "達成",
    "提升",
    "降低",
    "淨零",
    "減碳",
    "供應鏈",
    "導入",
    "完成",
    "兩年內",
    "明年",
    "短期",
    "18個月",
    "18 個月",
]


def load_pdf_urls(csv_paths: List[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = [c for c in ["pdf_url", "company", "ticker"] if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        subset = df[["pdf_url", "company", "ticker"]].dropna(subset=["pdf_url"]).copy()
        subset["source_csv"] = str(csv_path)
        rows.append(subset)
    combined = pd.concat(rows, ignore_index=True)
    combined["pdf_url"] = combined["pdf_url"].astype(str).str.strip()
    combined = combined[combined["pdf_url"].ne("")]
    return combined.drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)


def load_existing_texts(csv_paths: List[Path]) -> set[str]:
    texts = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "data" not in df.columns:
            continue
        texts.update(normalize_for_dedupe(x) for x in df["data"].dropna().astype(str))
    return texts


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip()


def cached_pdf_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.pdf"


def download_pdf(url: str, dest: Path, timeout: int, sleep_seconds: float) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 VeriPromiseESG research script "
                "(downloads official-source sustainability PDFs)"
            )
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            print(f"skip non-PDF response: {url}")
            return False
        dest.write_bytes(data)
        time.sleep(sleep_seconds)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"download failed: {url} ({exc})")
        return False


def extract_pdf_pages(pdf_path: Path) -> Iterable[Dict[str, object]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required. Install it with: pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - keep processing other pages.
            print(f"extract failed: {pdf_path} page {page_idx} ({exc})")
            text = ""
        yield {"page_number": page_idx, "text": text}


def split_paragraphs(text: str, min_chars: int, max_chars: int) -> List[str]:
    text = text.replace("\u3000", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        if len(line) <= 2:
            continue
        current.append(line)
        current_len += len(line)
        joined = " ".join(current)
        ends_sentence = bool(re.search(r"[。！？；:]$", line))
        if current_len >= min_chars and (ends_sentence or current_len >= max_chars):
            chunks.extend(slice_long_chunk(joined, min_chars, max_chars))
            current = []
            current_len = 0
    if current:
        chunks.extend(slice_long_chunk(" ".join(current), min_chars, max_chars))
    return chunks


def slice_long_chunk(text: str, min_chars: int, max_chars: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    sentences = re.split(r"(?<=[。！？；])", text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and len(current) >= min_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence
    if len(current.strip()) >= min_chars:
        chunks.append(current.strip())
    return chunks


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text]


def collect(args: argparse.Namespace) -> None:
    csv_paths = [Path(p) for p in args.csvs]
    sources = load_pdf_urls(csv_paths)
    existing_texts = load_existing_texts(csv_paths)
    if args.max_pdfs:
        sources = sources.head(args.max_pdfs)

    keywords = args.keywords or DEFAULT_KEYWORDS
    cache_dir = Path(args.cache_dir)
    seen = set(existing_texts)
    rows: List[Dict[str, object]] = []

    for idx, source in sources.iterrows():
        url = str(source["pdf_url"])
        pdf_path = cached_pdf_path(url, cache_dir)
        print(f"[{idx + 1}/{len(sources)}] {source['company']} {source['ticker']}")
        if args.download:
            ok = download_pdf(url, pdf_path, timeout=args.timeout, sleep_seconds=args.sleep)
            if not ok:
                continue
        elif not pdf_path.exists():
            print(f"missing cached PDF, skip: {pdf_path}")
            continue

        for page in extract_pdf_pages(pdf_path):
            for paragraph in split_paragraphs(
                str(page["text"]),
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            ):
                hits = keyword_hits(paragraph, keywords)
                if args.require_keyword and not hits:
                    continue
                key = normalize_for_dedupe(paragraph)
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source_pdf_url": url,
                        "company": source["company"],
                        "ticker": source["ticker"],
                        "source_page_number": page["page_number"],
                        "paragraph": paragraph,
                        "keyword_hits": "|".join(hits),
                        "char_len": len(paragraph),
                        "source_csv": source["source_csv"],
                    }
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        rows,
        columns=[
            "source_pdf_url",
            "company",
            "ticker",
            "source_page_number",
            "paragraph",
            "keyword_hits",
            "char_len",
            "source_csv",
        ],
    )
    out_df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {len(out_df)} candidate paragraphs to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=["data/vpesg_4k_train_1000.csv", "data/vpesg4k_val_1000.csv"],
        help="official labeled CSVs to use as PDF source lists; do not pass test CSV",
    )
    parser.add_argument("--output", default="data/external_esg_paragraphs.csv")
    parser.add_argument("--cache_dir", default="data/pdf_cache")
    parser.add_argument("--max_pdfs", type=int, default=0, help="0 means all unique PDFs")
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=700)
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--require_keyword", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
