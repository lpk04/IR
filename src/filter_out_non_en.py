from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


RE_ASCII_WORD = re.compile(r"[A-Za-z]{3,}")


def is_english(text: str, min_ascii_ratio: float = 0.75, min_words: int = 5) -> bool:
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_text(record: Dict[str, Any]) -> str:
    for key in ("text", "raw_text", "processed_text"):
        if key in record and record[key]:
            return str(record[key])
    return ""


def filter_english(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if is_english(extract_text(r))]


# ⭐ NEW: reindex function
def reindex(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for new_id, record in enumerate(records):
        record["doc_id"] = str(new_id)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter non-English reviews and reindex doc_id.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/yelp_reviews_100000_changed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yelp_reviews_100000_en.jsonl"),
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)

    # 1. filter
    filtered = filter_english(records)

    # 2. reindex
    reindexed = reindex(filtered)

    # 3. save
    write_jsonl(args.output, reindexed)

    print(f"Total records: {len(records)}")
    print(f"English kept: {len(filtered)}")
    print(f"After reindex: {len(reindexed)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()