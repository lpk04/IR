from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm

try:
    from config import DATA_DIR, YELP_REVIEW_FILE
except ImportError:
    from .config import DATA_DIR, YELP_REVIEW_FILE


DEFAULT_TEXT_FIELDS = ("text", "review", "review_text", "content", "body", "comment")
DEFAULT_ID_FIELDS = ("review_id")
DEFAULT_RATING_FIELDS = ("rating", "stars", "score")
DEFAULT_TOP_LEVEL_FIELDS = (
    "business_id",
    "user_id",
    "date",
    "sentiment",
    "useful",
    "funny",
    "cool",
)
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


# =========================
# PARSE ARGUMENTS
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=YELP_REVIEW_FILE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "metadata")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--id-fields", type=str, default="review_id")
    return parser.parse_args()


# =========================
# PATH HELPERS
# =========================
def get_output_paths(input_file: Path, output_dir: Path) -> dict[str, Path]:
    dataset_name = input_file.stem
    return {
        "metadata": output_dir / f"{dataset_name}_metadata.jsonl",
        "summary": output_dir / f"{dataset_name}_summary.json",
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_id_fields(raw_value: str | Sequence[str] | None) -> list[str]:
    if raw_value is None:
        return list(DEFAULT_ID_FIELDS)

    if isinstance(raw_value, str):
        fields = [item.strip() for item in raw_value.split(",")]
        return [field for field in fields if field] or list(DEFAULT_ID_FIELDS)

    return [str(field).strip() for field in raw_value if str(field).strip()]


# =========================
# TEXT HELPERS
# =========================
def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def estimate_sentence_count(text: str) -> int:
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return len(parts)


def resolve_text_field(record: dict[str, Any], preferred_field: str = "") -> str | None:
    if preferred_field and preferred_field in record:
        return preferred_field

    for field in DEFAULT_TEXT_FIELDS:
        if field in record:
            return field

    if preferred_field:
        return preferred_field

    return None


def resolve_text_value(record: dict[str, Any], preferred_field: str = "") -> tuple[str, str | None]:
    field_name = resolve_text_field(record, preferred_field)

    if field_name is None:
        return "", None

    value = record.get(field_name, "")
    if value is None:
        return "", field_name

    if isinstance(value, str):
        return value, field_name

    return str(value), field_name


# =========================
# FIELD EXTRACTION
# =========================
def resolve_review_id(record: dict[str, Any], review_line_no: int, id_fields: Sequence[str]) -> tuple[str, str | None]:
    for field in id_fields:
        value = record.get(field)
        if value not in (None, ""):
            return str(value), field

    return f"line_{review_line_no}", None


def resolve_rating(record: dict[str, Any]) -> float | int | str | None:
    for field in DEFAULT_RATING_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            return value
    return None


def copy_scalar_fields(record: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}

    for field in fields:
        value = record.get(field)
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            extracted[field] = value

    return extracted


def collect_extra_fields(
    record: dict[str, Any],
    excluded_fields: set[str],
) -> dict[str, Any]:
    extras: dict[str, Any] = {}

    for key, value in record.items():
        if key in excluded_fields:
            continue

        if isinstance(value, (int, float, bool)):
            extras[key] = value
            continue

        if isinstance(value, str) and len(value) <= 200:
            extras[key] = value

    return extras


# =========================
# METADATA BUILDERS
# =========================
def build_invalid_metadata(review_line_no: int, dataset_name: str, error_message: str) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "review_line_no": review_line_no,
        "review_id": f"line_{review_line_no}",
        "id_field": None,
        "valid_json": False,
        "parse_error": error_message,
        "text_field": None,
        "has_text": False,
        "is_empty_text": True,
        "char_count": 0,
        "token_count": 0,
        "unique_token_count": 0,
        "avg_token_length": 0.0,
        "sentence_count_est": 0,
        "rating": None,
    }


def build_metadata_record(
    record: dict[str, Any],
    review_line_no: int,
    dataset_name: str,
    preferred_text_field: str,
    id_fields: Sequence[str],
) -> dict[str, Any]:
    review_id, id_field = resolve_review_id(record, review_line_no, id_fields)
    text_value, text_field = resolve_text_value(record, preferred_text_field)

    normalized_text = text_value.strip()
    tokens = tokenize_text(normalized_text)
    unique_tokens = set(tokens)

    metadata = {
        "dataset": dataset_name,
        "review_line_no": review_line_no,
        "review_id": review_id,
        "id_field": id_field,
        "valid_json": True,
        "text_field": text_field,
        "has_text": text_field is not None,
        "is_empty_text": not bool(normalized_text),
        "char_count": len(normalized_text),
        "token_count": len(tokens),
        "unique_token_count": len(unique_tokens),
        "avg_token_length": round(sum(len(token) for token in tokens) / len(tokens), 4) if tokens else 0.0,
        "sentence_count_est": estimate_sentence_count(normalized_text) if normalized_text else 0,
        "rating": resolve_rating(record),
    }

    metadata.update(copy_scalar_fields(record, DEFAULT_TOP_LEVEL_FIELDS))

    excluded_fields = {text_field} if text_field else set()
    excluded_fields.update(DEFAULT_TOP_LEVEL_FIELDS)
    excluded_fields.update(DEFAULT_RATING_FIELDS)
    excluded_fields.update(id_fields)

    extra_fields = collect_extra_fields(record, excluded_fields)
    if extra_fields:
        metadata["extra_fields"] = extra_fields

    return metadata


# =========================
# SUMMARY HELPERS
# =========================
def create_summary(input_file: Path, preferred_text_field: str, id_fields: Sequence[str]) -> dict[str, Any]:
    return {
        "input_file": str(input_file),
        "dataset": input_file.stem,
        "preferred_text_field": preferred_text_field or None,
        "id_fields": list(id_fields),
        "records_seen": 0,
        "valid_records": 0,
        "malformed_records": 0,
        "missing_text_records": 0,
        "empty_text_records": 0,
        "records_with_tokens": 0,
        "total_tokens": 0,
        "max_token_count": 0,
        "min_token_count": None,
        "rating_distribution": Counter(),
    }


def update_summary(summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    summary["records_seen"] += 1

    if not metadata["valid_json"]:
        summary["malformed_records"] += 1
        return

    summary["valid_records"] += 1

    if not metadata["has_text"]:
        summary["missing_text_records"] += 1
    elif metadata["is_empty_text"]:
        summary["empty_text_records"] += 1

    token_count = int(metadata["token_count"])
    summary["total_tokens"] += token_count

    if token_count > 0:
        summary["records_with_tokens"] += 1
        summary["max_token_count"] = max(summary["max_token_count"], token_count)

        min_token_count = summary["min_token_count"]
        if min_token_count is None:
            summary["min_token_count"] = token_count
        else:
            summary["min_token_count"] = min(min_token_count, token_count)

    rating = metadata.get("rating")
    if rating not in (None, ""):
        summary["rating_distribution"][str(rating)] += 1


def finalize_summary(summary: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    valid_records = summary["valid_records"]
    summary["output_metadata_file"] = str(output_paths["metadata"])
    summary["output_summary_file"] = str(output_paths["summary"])
    summary["average_token_count"] = round(
        summary["total_tokens"] / valid_records,
        4,
    ) if valid_records else 0.0
    summary["rating_distribution"] = dict(summary["rating_distribution"])
    return summary


# =========================
# MAIN GENERATION
# =========================
def generate_metadata(
    input_file: Path = YELP_REVIEW_FILE,
    output_dir: Path = DATA_DIR / "metadata",
    text_field: str = "",
    id_fields: Sequence[str] | None = None,
) -> dict[str, Path]:
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    id_fields = parse_id_fields(id_fields)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_paths = get_output_paths(input_file, output_dir)
    ensure_dir(output_paths["metadata"].parent)

    summary = create_summary(input_file, text_field, id_fields)

    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_paths["metadata"], "w", encoding="utf-8"
    ) as fout:
        for review_line_no, line in enumerate(tqdm(fin, desc="Generating metadata"), start=1):
            raw_line = line.strip()

            if not raw_line:
                metadata = build_invalid_metadata(
                    review_line_no=review_line_no,
                    dataset_name=input_file.stem,
                    error_message="empty_line",
                )
            else:
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    metadata = build_invalid_metadata(
                        review_line_no=review_line_no,
                        dataset_name=input_file.stem,
                        error_message=f"json_decode_error: {exc.msg}",
                    )
                else:
                    if isinstance(record, dict):
                        metadata = build_metadata_record(
                            record=record,
                            review_line_no=review_line_no,
                            dataset_name=input_file.stem,
                            preferred_text_field=text_field,
                            id_fields=id_fields,
                        )
                    else:
                        metadata = build_invalid_metadata(
                            review_line_no=review_line_no,
                            dataset_name=input_file.stem,
                            error_message="record_is_not_an_object",
                        )

            fout.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            update_summary(summary, metadata)

    summary = finalize_summary(summary, output_paths)

    with open(output_paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_paths


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    paths = generate_metadata(
        input_file=args.input,
        output_dir=args.output_dir,
        text_field=args.text_field,
        id_fields=args.id_fields,
    )

    print(f"Saved metadata -> {paths['metadata']}")
    print(f"Saved summary -> {paths['summary']}")
