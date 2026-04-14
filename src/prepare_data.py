from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

try:
    from config import (
        MAX_DOCS,
        REVIEW_PROCESSED_FILE,
        REVIEW_RAW_FILE,
        TRACE_DIR,
        TRACE_FILE,
        YELP_REVIEW_FILE,
    )
except ImportError:
    from .config import (
        MAX_DOCS,
        REVIEW_PROCESSED_FILE,
        REVIEW_RAW_FILE,
        TRACE_DIR,
        TRACE_FILE,
        YELP_REVIEW_FILE,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
RE_ALPHA = re.compile(r"[^a-z\s]")
RE_EXTRA_SPACE = re.compile(r"\s{2,}")
RE_ASCII_WORD = re.compile(r"[A-Za-z]{3,}")

DEFAULT_TEXT_FIELDS = ("text", "review_text", "content", "body", "review")

# ---------------------------------------------------------------------------
# Lazy singletons (initialised once per process)
# ---------------------------------------------------------------------------
STOP_WORDS: set[str] | None = None
LEMMATIZER: WordNetLemmatizer | None = None


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_runtime_resources() -> None:
    global STOP_WORDS, LEMMATIZER

    if STOP_WORDS is None:
        # Keep negation words — they carry meaning in BM25/TF-IDF queries too.
        STOP_WORDS = set(stopwords.words("english")) - {"not", "never", "none", "nothing"}

    if LEMMATIZER is None:
        LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_sentiment(rating: float) -> str:
    if rating >= 4.0:
        return "positive"
    if rating == 3.0:
        return "neutral"
    return "negative"


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the corresponding WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_whitespace(text: str) -> str:
    return RE_EXTRA_SPACE.sub(" ", text).strip()


def detect_text_field(record: dict[str, Any], preferred_field: str | None = None) -> str:
    """Return the text value from the first matching field in *record*."""
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
    """Lightweight heuristic: accept text that is predominantly ASCII."""
    if not text:
        return False

    total_chars = len(text)
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    ascii_ratio = ascii_chars / total_chars

    if ascii_ratio < min_ascii_ratio:
        return False

    ascii_word_count = len(RE_ASCII_WORD.findall(text))

    if ascii_word_count >= min_words:
        return True

    # Short but clearly ASCII (e.g. "Great food!", "Awesome place")
    if total_chars <= 60 and ascii_ratio >= 0.8 and ascii_word_count >= 1:
        return True

    if ascii_ratio >= 0.95 and ascii_word_count >= 1:
        return True

    return False


def get_record_id(record: dict[str, Any], fallback_id: int) -> str:
    for field in ("review_id", "doc_id", "id"):
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return str(fallback_id)


# ---------------------------------------------------------------------------
# BM25 / TF-IDF preprocessing
#
# Strategy (differs from transformer prep):
#   • All POS tags are kept — BM25/TF-IDF relies on term frequency across the
#     full vocabulary, so pruning to only NN/VB/JJ/RB would discard signal.
#   • Stopwords are removed; negations are preserved.
#   • Every surviving token is lemmatised using its actual POS tag so that
#     "running" → "run", "better" → "good", etc.
#   • Minimum token length ≥ 2 (excludes single-char noise while keeping
#     meaningful two-letter words such as "ok").
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Return a lemmatised, stopword-filtered string suitable for BM25/TF-IDF."""
    init_runtime_resources()
    assert STOP_WORDS is not None
    assert LEMMATIZER is not None

    # 1. Lowercase and strip non-alpha characters.
    normalised = text.replace("\n", " ").lower()
    normalised = RE_ALPHA.sub(" ", normalised)
    normalised = normalize_whitespace(normalised)

    # 2. Tokenise.
    tokens = word_tokenize(normalised)

    # 3. Remove stopwords and very short tokens.
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]

    if not tokens:
        return ""

    # 4. POS-tag and lemmatise — keep ALL POS tags (BM25/TF-IDF strategy).
    tagged = pos_tag(tokens)
    lemmas = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]

    return " ".join(lemmas)


# ---------------------------------------------------------------------------
# Core record builder
# ---------------------------------------------------------------------------
def build_processed_record(
    record: dict[str, Any],
    output_id: int,
    review_line_no: int,
    preferred_text_field: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Returns (output_record, trace_item).
    If the record should be skipped, output_record is None.
    """
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

    clean_text = preprocess_text(raw_text)
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
        "raw_token_count": len(raw_text.split()),
        "clean_token_count": len(clean_text.split()),
    }

    return output, trace_item


def build_trace_block(trace_item: dict[str, Any]) -> str:
    return json.dumps(trace_item, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary(summary_file: Path, stats: PipelineStats) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Null context (mirrors prepare_data_transformer.py)
# ---------------------------------------------------------------------------
class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_data(
    input_file: Path = YELP_REVIEW_FILE,
    output_file: Path = REVIEW_PROCESSED_FILE,
    trace_file: Path | None = TRACE_FILE,
    summary_file: Path | None = None,
    non_english_file: Path | None = None,
    max_docs: int | None = MAX_DOCS,
    trace_limit: int = 20,
    preferred_text_field: str | None = None,
) -> Path:
    """
    Read raw Yelp JSONL, apply BM25/TF-IDF preprocessing and write clean JSONL.

    Non-English records are optionally written to *non_english_file* for
    inspection.  A JSON summary of pipeline statistics is always written next
    to *output_file*.
    """
    init_runtime_resources()

    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if summary_file is None:
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"
    summary_file = Path(summary_file)

    if trace_file is not None:
        trace_file = Path(trace_file)
        trace_file.parent.mkdir(parents=True, exist_ok=True)

    if non_english_file is None:
        non_english_file = output_file.parent / f"{output_file.stem}_non_english.jsonl"
    non_english_file = Path(non_english_file)
    non_english_file.parent.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats(
        started_at=datetime.now(timezone.utc).isoformat(),
        input_file=str(input_file),
        output_file=str(output_file),
        trace_file=str(trace_file) if trace_file is not None else None,
        max_docs=max_docs,
        trace_limit=trace_limit,
    )

    t0 = time.perf_counter()

    with (
        open(input_file, "r", encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
        (open(trace_file, "w", encoding="utf-8") if trace_file is not None else _null_context()) as ftrace,
        open(non_english_file, "w", encoding="utf-8") as fnoneng,
    ):
        for line_no, line in enumerate(tqdm(fin, desc="Preparing BM25/TF-IDF data"), start=1):
            if max_docs is not None and stats.records_written >= max_docs:
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

    LOGGER.info("Records seen          : %s", stats.records_seen)
    LOGGER.info("Valid JSON records     : %s", stats.valid_json_records)
    LOGGER.info("Written records       : %s", stats.records_written)
    LOGGER.info("Malformed JSON        : %s", stats.malformed_json_records)
    LOGGER.info("Missing text          : %s", stats.missing_text_records)
    LOGGER.info("Non-English           : %s", stats.non_english_records)
    LOGGER.info("Empty after cleaning  : %s", stats.empty_clean_records)
    LOGGER.info("Output file           : %s", output_file)
    LOGGER.info("Summary file          : %s", summary_file)
    if trace_file is not None:
        LOGGER.info("Trace preview         : %s", trace_file)

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Yelp reviews for BM25 / TF-IDF retrieval."
    )
    parser.add_argument("--input", type=Path, default=YELP_REVIEW_FILE)
    parser.add_argument("--output", type=Path, default=REVIEW_PROCESSED_FILE)
    parser.add_argument("--trace", type=Path, default=TRACE_FILE)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--non-english", type=Path, default=None)
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
        non_english_file=args.non_english,
        max_docs=args.max_docs,
        trace_limit=args.trace_limit,
        preferred_text_field=args.preferred_text_field,
    )


if __name__ == "__main__":
    main()