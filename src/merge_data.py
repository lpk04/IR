from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import (
    PROCESSED_DATA_DIR,
    REVIEW_PROCESSED_FILE,
    REVIEW_MERGERED_PROCESSED_FILE, 
)

EN_DATA_DIR = PROCESSED_DATA_DIR / "yelp_reviews_100000_en.jsonl"

def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
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


def build_raw_index(raw_records: List[dict]) -> Tuple[Dict[str, dict], List[dict]]:
    by_doc_id: Dict[str, dict] = {}
    ordered: List[dict] = []

    for record in raw_records:
        ordered.append(record)
        doc_id = record.get("doc_id")
        if doc_id is not None:
            by_doc_id[str(doc_id)] = record
    return by_doc_id, ordered


def get_text(record: Optional[dict]) -> str:
    if not record:
        return ""
    for key in ("text", "processed_text", "raw_text"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def merge_records(raw_records: List[dict], processed_records: List[dict]) -> List[dict]:
    raw_by_doc_id, raw_ordered = build_raw_index(raw_records)

    merged: List[dict] = []
    used_raw_ids = set()
    raw_seq_idx = 0

    for processed_rec in processed_records:
        proc_doc_id = processed_rec.get("doc_id")
        proc_doc_id_str = str(proc_doc_id) if proc_doc_id is not None else None

        raw_rec: Optional[dict] = None

        if proc_doc_id_str is not None and proc_doc_id_str in raw_by_doc_id:
            raw_rec = raw_by_doc_id[proc_doc_id_str]
            used_raw_ids.add(proc_doc_id_str)
        else:
            while raw_seq_idx < len(raw_ordered):
                candidate = raw_ordered[raw_seq_idx]
                raw_seq_idx += 1
                candidate_id = candidate.get("doc_id")
                candidate_id_str = str(candidate_id) if candidate_id is not None else None
                if candidate_id_str is not None and candidate_id_str in used_raw_ids:
                    continue
                raw_rec = candidate
                if candidate_id_str is not None:
                    used_raw_ids.add(candidate_id_str)
                break

        merged_rec = {
            "doc_id": proc_doc_id_str,
            "raw_text": get_text(raw_rec),
            "processed_text": get_text(processed_rec),
            "rating": processed_rec.get("rating"),
            "sentiment": processed_rec.get("sentiment"),
        }

        for key, value in processed_rec.items():
            if key in {"text", "processed_text", "raw_text"}:
                continue
            if key not in merged_rec:
                merged_rec[key] = value

        merged.append(merged_rec)

    return merged


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge raw and processed JSONL files.")
    parser.add_argument("--raw", type=Path, default=EN_DATA_DIR)
    parser.add_argument("--processed", type=Path, default=REVIEW_PROCESSED_FILE)
    parser.add_argument("--output", type=Path, default=REVIEW_MERGERED_PROCESSED_FILE)
    args = parser.parse_args()

    raw_records = read_jsonl(args.raw)
    processed_records = read_jsonl(args.processed)

    merged_records = merge_records(raw_records, processed_records)
    write_jsonl(args.output, merged_records)

    print(f"Raw records: {len(raw_records)}")
    print(f"Processed records: {len(processed_records)}")
    print(f"Merged records: {len(merged_records)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()