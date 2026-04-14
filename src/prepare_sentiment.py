"""
Prepare Yelp reviews for sentiment-aware retrieval.

Processing strategy (differs from BM25/TF-IDF in prepare_data.py):
  • Only sentiment-bearing POS tags are kept: adjectives (JJ*), adverbs (RB*),
    verbs (VB*), and nouns (NN*) — these carry the most lexical sentiment signal.
  • Negation words (not, never, none, nothing, no, neither, nor) are preserved and
    attached to the following token as a bigram (e.g. "not_good") so polarity is not lost.
  • Tokens are filtered against a domain keyword list covering food/service/ambience/value.
  • Every surviving token is lemmatised using its actual POS tag.
  • Minimum token length ≥ 2.

Output fields per record:
    doc_id, text (processed), rating, sentiment,
    sentiment_words (list), keyword_count

Run:
    python src/prepare_sentiment.py
    python src/prepare_sentiment.py --input data/raw/reviews.jsonl --max-docs 50000
"""

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
        REVIEW_RAW_FILE,
        REVIEW_SENTIMENT_FILE,
        TRACE_DIR,
    )
except ImportError:
    from .config import (
        MAX_DOCS,
        REVIEW_RAW_FILE,
        REVIEW_SENTIMENT_FILE,
        TRACE_DIR,
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

# POS tag prefixes that carry sentiment/keyword signal
SENTIMENT_POS_PREFIXES = ("JJ", "RB", "VB", "NN")

# Negation words: preserved AND used to form bigrams with the next token
NEGATION_WORDS = {"not", "never", "none", "nothing", "no", "neither", "nor"}

# Domain keywords always kept regardless of POS (food/service/ambience/value)
DOMAIN_KEYWORDS: set[str] = {
    # food quality
    "fresh", "stale", "tasty", "bland", "delicious", "disgusting", "crispy",
    "soggy", "greasy", "flavourful", "flavorful", "spicy", "sweet", "sour",
    "salty", "bitter", "overcooked", "undercooked", "raw", "hot", "cold",
    "warm", "frozen", "juicy", "dry", "tender", "tough", "chewy", "crunchy",
    # service
    "friendly", "rude", "attentive", "slow", "fast", "quick", "efficient",
    "unprofessional", "helpful", "knowledgeable", "courteous", "dismissive",
    "welcoming", "inattentive", "responsive", "polite", "impolite",
    # ambience / experience
    "clean", "dirty", "noisy", "quiet", "cozy", "cramped", "spacious",
    "crowded", "busy", "empty", "comfortable", "uncomfortable", "lively",
    "dull", "romantic", "casual", "upscale", "fancy", "rustic", "modern",
    # value
    "expensive", "cheap", "affordable", "overpriced", "worth", "value",
    "reasonable", "pricey", "costly",
    # general sentiment anchors
    "love", "hate", "enjoy", "disappoint", "recommend", "avoid", "return",
    "regret", "impress", "surprise", "expect", "satisfy", "complain",
    "excellent", "terrible", "wonderful", "awful", "amazing", "horrible",
    "great", "bad", "good", "best", "worst", "mediocre", "decent", "ok",
    "okay", "fine", "poor", "average", "outstanding", "unacceptable",
}

# ---------------------------------------------------------------------------
# Lazy singletons (initialised once per process)
# ---------------------------------------------------------------------------
STOP_WORDS: set[str] | None = None
LEMMATIZER: WordNetLemmatizer | None = None
SENTIMENT_LEXICON: set[str] | None = None  # domain keywords only


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
    global STOP_WORDS, LEMMATIZER, SENTIMENT_LEXICON

    if STOP_WORDS is None:
        # Keep negation words — they flip polarity
        STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

    if LEMMATIZER is None:
        LEMMATIZER = WordNetLemmatizer()

    if SENTIMENT_LEXICON is None:
        # Use only domain-specific keywords; no VADER/opinion lexicon.
        SENTIMENT_LEXICON = set(DOMAIN_KEYWORDS)
        LOGGER.info("Sentiment lexicon size: %d tokens", len(SENTIMENT_LEXICON))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expand_contractions(text: str) -> str:
    text = text.lower()

    contractions = {
        "can't": "can not",
        "won't": "will not",
        "n't": " not",

        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",

        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "what's": "what is",
        "who's": "who is",
        "where's": "where is",
        "how's": "how is",

        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",

        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",

        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "they'll": "they will",

        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
    }

    for key, value in contractions.items():
        text = text.replace(key, value)

    return text


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
# Sentiment-focused preprocessing
#
# Strategy:
#   1. Expand contractions.
#   2. Lowercase + strip non-alpha characters.
#   3. Tokenise with NLTK word_tokenize.
#   4. Attach negation: "not good" -> "not_good"; negation scope = 1 token.
#   5. POS-tag; keep only sentiment-bearing tags (JJ*, RB*, VB*, NN*) plus
#      any token that appears in the domain keyword list.
#   6. Remove stopwords (negation bigrams are already formed).
#   7. Lemmatise each surviving token using its actual POS tag.
#   8. Minimum token length >= 2 (post-lemma).
# ---------------------------------------------------------------------------
def _attach_negations(tokens: list[str]) -> list[str]:
    """
    Replace "NEG token" pairs with a "NEG_token" bigram.
    Negation scope is limited to one token.
    """
    result: list[str] = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{token}_{tokens[i + 1]}")
            skip_next = True
        else:
            result.append(token)
    return result


def preprocess_text_sentiment(text: str) -> tuple[str, list[str]]:
    """
    Return (processed_text, sentiment_words_list).

    processed_text  — lemmatised sentiment/keyword tokens joined by spaces.
    sentiment_words — subset of tokens that matched the domain keyword list.
    """
    init_runtime_resources()
    assert STOP_WORDS is not None
    assert LEMMATIZER is not None
    assert SENTIMENT_LEXICON is not None

    # 1. Expand contractions first.
    normalised = expand_contractions(text)

    # 2. Lowercase and strip non-alpha characters.
    normalised = normalised.replace("\n", " ").lower()
    normalised = RE_ALPHA.sub(" ", normalised)
    normalised = normalize_whitespace(normalised)

    # 3. Tokenise.
    tokens = word_tokenize(normalised)

    # 4. Attach negation bigrams before stopword removal.
    tokens = _attach_negations(tokens)

    # 5. Remove stopwords.
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]

    if not tokens:
        return "", []

    # 6. POS-tag.
    tagged = pos_tag(tokens)

    # 7. Keep tokens that are either:
    #    a) a sentiment-bearing POS tag, OR
    #    b) present in the domain lexicon.
    #    Negation bigrams always pass through.
    kept: list[tuple[str, str]] = []
    for word, tag in tagged:
        is_negation_bigram = "_" in word
        has_sentiment_pos = tag.startswith(SENTIMENT_POS_PREFIXES)
        base_word = word.split("_")[-1] if is_negation_bigram else word
        in_lexicon = base_word in SENTIMENT_LEXICON or word in SENTIMENT_LEXICON
        if is_negation_bigram or has_sentiment_pos or in_lexicon:
            kept.append((word, tag))

    if not kept:
        return "", []

    # 8. Lemmatise using actual POS tags; preserve negation bigram structure.
    lemmas: list[str] = []
    sentiment_words: list[str] = []

    for word, tag in kept:
        if "_" in word:
            # Negation bigram — lemmatise the content half only.
            neg_part, content_part = word.split("_", 1)
            lemma_content = LEMMATIZER.lemmatize(content_part, get_wordnet_pos(tag))
            lemma = f"{neg_part}_{lemma_content}"
        else:
            lemma = LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))

        if len(lemma) < 2:
            continue

        lemmas.append(lemma)

        # Track which tokens came from the domain lexicon.
        base = lemma.split("_")[-1] if "_" in lemma else lemma
        if base in SENTIMENT_LEXICON:
            sentiment_words.append(lemma)

    return " ".join(lemmas), sentiment_words


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

    clean_text, sentiment_words = preprocess_text_sentiment(raw_text)
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
        "sentiment_words": sentiment_words,
        "keyword_count": len(clean_text.split()),
    }

    trace_item = {
        "doc_id": doc_id,
        "review_id": review_id,
        "status": "written",
        "rating": rating,
        "sentiment": output["sentiment"],
        "raw_token_count": len(raw_text.split()),
        "clean_token_count": len(clean_text.split()),
        "sentiment_word_count": len(sentiment_words),
        "sample_sentiment_words": sentiment_words[:10],
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
# Null context
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
    input_file: Path = REVIEW_RAW_FILE,
    output_file: Path = REVIEW_SENTIMENT_FILE,
    trace_file: Path | None = None,
    summary_file: Path | None = None,
    non_english_file: Path | None = None,
    max_docs: int | None = MAX_DOCS,
    trace_limit: int = 20,
    preferred_text_field: str | None = None,
) -> Path:
    """
    Read raw Yelp JSONL, apply sentiment/keyword preprocessing and write clean JSONL.

    Each output record contains:
        doc_id, text (processed), rating, sentiment, sentiment_words, keyword_count

    Non-English records are optionally written to *non_english_file* for
    inspection. A JSON summary of pipeline statistics is always written next
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
    else:
        default_trace = TRACE_DIR / "sentiment_trace.jsonl"
        default_trace.parent.mkdir(parents=True, exist_ok=True)
        trace_file = default_trace

    if non_english_file is None:
        non_english_file = output_file.parent / f"{output_file.stem}_non_english.jsonl"
    non_english_file = Path(non_english_file)
    non_english_file.parent.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats(
        started_at=datetime.now(timezone.utc).isoformat(),
        input_file=str(input_file),
        output_file=str(output_file),
        trace_file=str(trace_file),
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
        for line_no, line in enumerate(tqdm(fin, desc="Preparing sentiment data"), start=1):
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
    LOGGER.info("Valid JSON records    : %s", stats.valid_json_records)
    LOGGER.info("Written records       : %s", stats.records_written)
    LOGGER.info("Malformed JSON        : %s", stats.malformed_json_records)
    LOGGER.info("Missing text          : %s", stats.missing_text_records)
    LOGGER.info("Non-English           : %s", stats.non_english_records)
    LOGGER.info("Empty after cleaning  : %s", stats.empty_clean_records)
    LOGGER.info("Output file           : %s", output_file)
    LOGGER.info("Summary file          : %s", summary_file)
    LOGGER.info("Trace preview         : %s", trace_file)

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Yelp reviews for sentiment-aware retrieval."
    )
    parser.add_argument("--input", type=Path, default=REVIEW_RAW_FILE)
    parser.add_argument("--output", type=Path, default=REVIEW_SENTIMENT_FILE)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--non-english", type=Path, default=None)
    parser.add_argument("--trace-limit", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS)
    parser.add_argument("--preferred-text-field", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_file = None if args.trace and str(args.trace).strip().lower() == "none" else args.trace

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