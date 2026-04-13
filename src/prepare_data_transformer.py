from __future__ import annotations

import argparse
import html
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from config import (
        MAX_DOCS,
        TRACE_DIR,
        TRANSFORMER_PREP_TRACE_FILE,
        TRANSFORMER_NON_ENGLISH_FILE,
        REVIEW_TRANSFORMER_PROCESSED_FILE,
        YELP_REVIEW_FILE,
    )
except ImportError:
    from .config import (
        MAX_DOCS,
        TRACE_DIR,
        TRANSFORMER_PREP_TRACE_FILE,
        TRANSFORMER_NON_ENGLISH_FILE,
        REVIEW_TRANSFORMER_PROCESSED_FILE,
        YELP_REVIEW_FILE,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_CTRL = re.compile(r"[\r\n\t\x00-\x1f\x7f]")
RE_EXTRA_SPACE = re.compile(r"\s{2,}")
RE_ASCII_WORD = re.compile(r"[A-Za-z]{3,}")

DEFAULT_TEXT_FIELDS = ("text", "review_text", "content", "body", "review")


@dataclass
class PipelineStats:
    started_at: str
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    input_file: str | None = None
    output_file: str | None = None
    trace_file: str | None = None
    records_seen: int = 0
    valid_json_records: int = 0
    malformed_json_records: int = 0
    missing_text_records: int = 0
    non_english_records: int = 0
    empty_clean_records: int = 0
    records_written: int = 0
    max_docs: int | None = None
    trace_limit: int = 20


def get_sentiment(rating: float) -> str:
    if rating >= 4.0:
        return "positive"
    if rating == 3.0:
        return "neutral"
    return "negative"


def normalize_whitespace(text: str) -> str:
    return RE_EXTRA_SPACE.sub(" ", text).strip()


def detect_text_field(record: dict[str, Any], preferred_field: str | None = None) -> str:
    if preferred_field and preferred_field in record:
        value = record.get(preferred_field)
        if value is not None:
            return str(value)

    for field in DEFAULT_TEXT_FIELDS:
        value = record.get(field)
        if value is not None:
            return str(value)

    return ""


def is_english(text: str, min_ascii_ratio: float = 0.75, min_words: int = 5) -> bool:
    """
    Lightweight heuristic only. No language model, no heavy NLP.
    """
    if not text:
        return False

    total_chars = len(text)
    if total_chars == 0:
        return False

    ascii_chars = sum(1 for char in text if ord(char) < 128)
    ascii_ratio = ascii_chars / total_chars
    if ascii_ratio < min_ascii_ratio:
        return False

    ascii_word_count = len(RE_ASCII_WORD.findall(text))
    if ascii_word_count >= min_words:
        return True

    if total_chars <= 60 and ascii_ratio >= 0.8 and ascii_word_count >= 1:
        return True

    if ascii_ratio >= 0.95 and ascii_word_count >= 1:
        return True

    return False


def clean_base_text(text: str) -> str:
    """
    Minimal cleaning for transformer retrieval:
    - decode HTML entities
    - strip HTML tags
    - remove URLs
    - remove control characters
    - normalize unicode
    - collapse extra whitespace

    Important:
    No stemming, no stopword removal, no POS tagging, no lemmatization,
    no lowercasing, and no token filtering.
    """
    cleaned = html.unescape(text or "")

    if "<" in cleaned and ">" in cleaned:
        try:
            cleaned = BeautifulSoup(cleaned, "html.parser").get_text(separator=" ")
        except Exception:
            cleaned = RE_HTML_TAG.sub(" ", cleaned)

    cleaned = RE_URL.sub(" ", cleaned)
    cleaned = RE_CTRL.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = normalize_whitespace(cleaned)

    return cleaned


def get_record_id(record: dict[str, Any], fallback_id: int) -> str:
    for field in ("review_id",):
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return str(fallback_id)


def build_trace_block(trace_item: dict[str, Any]) -> str:
    return json.dumps(trace_item, ensure_ascii=False) + "\n"


def build_processed_record(
    record: dict[str, Any],
    output_id: int,
    review_line_no: int,
    preferred_text_field: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_text = detect_text_field(record, preferred_field=preferred_text_field)
    review_id = get_record_id(record, fallback_id=review_line_no)

    if not raw_text.strip():
        return None, {
            "review_line_no": review_line_no,
            "review_id": review_id,
            "status": "skipped",
            "reason": "missing_text",
        }

    if not is_english(raw_text):
        return None, {
            "review_line_no": review_line_no,
            "review_id": review_id,
            "status": "skipped",
            "reason": "non_english",
        }

    clean_text = clean_base_text(raw_text)
    if not clean_text:
        return None, {
            "review_line_no": review_line_no,
            "review_id": review_id,
            "status": "skipped",
            "reason": "empty_clean_text",
        }

    rating_value = record.get("stars", record.get("rating", 0))
    try:
        rating = float(rating_value)
    except (TypeError, ValueError):
        rating = 0.0

    doc_id = str(output_id)

    output = {
        "doc_id": doc_id,
        "text": clean_text,
        "rating": rating,
        "sentiment": get_sentiment(rating),
    }

    trace_item = {
        "doc_id": doc_id,
        "review_id": review_id,
        "status": "written",
        "rating": rating,
        "sentiment": output["sentiment"],
        "raw_char_count": len(raw_text),
        "raw_word_count": len(raw_text.split()),
        "clean_char_count": len(clean_text),
        "clean_word_count": len(clean_text.split()),
    }

    return output, trace_item


def write_summary(summary_file: Path, stats: PipelineStats) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)


def process_data(
    input_file: Path = YELP_REVIEW_FILE,
    output_file: Path = REVIEW_TRANSFORMER_PROCESSED_FILE,
    trace_file: Path | None = TRANSFORMER_PREP_TRACE_FILE,
    summary_file: Path | None = None,
    max_docs: int | None = MAX_DOCS,
    trace_limit: int = 20,
    preferred_text_field: str | None = None,
) -> Path:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if summary_file is None:
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"
    summary_file = Path(summary_file)

    if trace_file is not None:
        trace_file = Path(trace_file)
        trace_file.parent.mkdir(parents=True, exist_ok=True)

    non_english_trace = Path(TRANSFORMER_NON_ENGLISH_FILE)
    non_english_trace.parent.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats(
        started_at=datetime.now(timezone.utc).isoformat(),
        input_file=str(input_file),
        output_file=str(output_file),
        trace_file=str(trace_file) if trace_file is not None else None,
        max_docs=max_docs,
        trace_limit=trace_limit,
    )

    t0 = time.perf_counter()

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        with (
            open(trace_file, "w", encoding="utf-8") if trace_file is not None else _null_context()
        ) as ftrace, open(non_english_trace, "w", encoding="utf-8") as fnoneng:
            for line_no, line in enumerate(tqdm(fin, desc="Preparing transformer data"), start=1):
                if max_docs is not None and stats.valid_json_records >= max_docs:
                    break

                raw_line = line.strip()
                if not raw_line:
                    continue

                stats.records_seen += 1

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    stats.malformed_json_records += 1
                    continue

                if not isinstance(record, dict):
                    stats.malformed_json_records += 1
                    continue

                stats.valid_json_records += 1

                output, trace_item = build_processed_record(
                    record=record,
                    output_id=stats.records_written,
                    review_line_no=line_no,
                    preferred_text_field=preferred_text_field,
                )

                if output is None:
                    reason = trace_item.get("reason") if trace_item else None
                    if reason == "missing_text":
                        stats.missing_text_records += 1
                    elif reason == "non_english":
                        stats.non_english_records += 1
                        try:
                            fnoneng.write(raw_line + "\n")
                        except Exception:
                            pass
                    elif reason == "empty_clean_text":
                        stats.empty_clean_records += 1
                    continue

                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                stats.records_written += 1

                if ftrace is not None and stats.records_written <= trace_limit and trace_item is not None:
                    ftrace.write(build_trace_block(trace_item))

    stats.finished_at = datetime.now(timezone.utc).isoformat()
    stats.elapsed_seconds = round(time.perf_counter() - t0, 4)

    write_summary(summary_file, stats)

    LOGGER.info("Records seen: %s", stats.records_seen)
    LOGGER.info("Valid JSON records: %s", stats.valid_json_records)
    LOGGER.info("Written records: %s", stats.records_written)
    LOGGER.info("Malformed JSON records: %s", stats.malformed_json_records)
    LOGGER.info("Missing text records: %s", stats.missing_text_records)
    LOGGER.info("Non-English records: %s", stats.non_english_records)
    LOGGER.info("Empty clean records: %s", stats.empty_clean_records)
    LOGGER.info("Saved processed file to %s", output_file)
    LOGGER.info("Saved summary file to %s", summary_file)
    if trace_file is not None:
        LOGGER.info("Saved trace preview to %s", trace_file)

    return output_file


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Yelp reviews for transformer-based retrieval with minimal text cleaning."
    )
    parser.add_argument("--input", type=Path, default=YELP_REVIEW_FILE)
    parser.add_argument("--output", type=Path, default=REVIEW_TRANSFORMER_PROCESSED_FILE)
    parser.add_argument("--trace", type=Path, default=TRANSFORMER_PREP_TRACE_FILE)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--trace-limit", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS)
    parser.add_argument("--preferred-text-field", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_file = None if str(args.trace).strip().lower() == "none" else args.trace

    process_data(
        input_file=args.input,
        output_file=args.output,
        trace_file=trace_file,
        summary_file=args.summary,
        max_docs=args.max_docs,
        trace_limit=args.trace_limit,
        preferred_text_field=args.preferred_text_field,
    )


if __name__ == "__main__":
    main()