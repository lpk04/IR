from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_INPUT = Path("data/processed/yelp_reviews_100000_merged.jsonl")
DEFAULT_OUTPUT = Path("data/processed/yelp_reviews_100000_merged_no_neu.jsonl")


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
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_neutral(record: Dict[str, Any]) -> bool:
    sentiment = str(record.get("sentiment", "")).strip().lower()
    return sentiment in {"neu", "neutral", "neut", "neutrality"}


def filter_out_neutral_reviews(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if not is_neutral(record)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter out neutral reviews from a JSONL file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL file.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    filtered = filter_out_neutral_reviews(records)
    write_jsonl(args.output, filtered)

    removed = len(records) - len(filtered)
    print(f"Input records:  {len(records)}")
    print(f"Removed neutral: {removed}")
    print(f"Output records: {len(filtered)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()