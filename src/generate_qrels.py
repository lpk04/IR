from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import DATA_DIR, RESULTS_DIR, LOG_FILE, QRELS_FILE

# Keep these paths for compatibility with your existing pipeline.
QRELS_KEYWORD = RESULTS_DIR / "qrels_keyword.txt"
QRELS_COUNT = RESULTS_DIR / "qrels_count.txt"
QRELS_RATIO = RESULTS_DIR / "qrels_ratio.txt"

# Default location produced by build_candidate_pool.py
DEFAULT_CANDIDATE_POOL = DATA_DIR / "candidate_pool.csv"


def load_candidate_pool(path: Path) -> list[dict[str, Any]]:
    """
    Load candidate pool rows from CSV.

    Expected columns include at least:
        qid, doc_id

    Optional columns:
        query, rating, sentiment, sources, ranks, label, text, is_random
    """
    if not path.exists():
        raise FileNotFoundError(f"Candidate pool file not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Candidate pool CSV has no header: {path}")

        required = {"qid", "doc_id"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Candidate pool is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            if not row.get("qid") or not row.get("doc_id"):
                continue
            rows.append(row)

    return rows


def parse_label(value: Any, default: int = 0) -> int:
    """
    Convert the candidate pool label field to an integer qrel.

    Accepts:
    - numeric strings: "0", "1", "2", "3"
    - empty / missing values -> default
    """
    if value is None:
        return default

    text = str(value).strip()
    if text == "":
        return default

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def build_qrels(rows: list[dict[str, Any]], use_label_column: bool = True) -> list[tuple[str, str, int]]:
    """
    Build TREC qrels from the candidate pool.

    If a `label` column exists, use it as the relevance judgment.
    Otherwise, fall back to 0.
    """
    qrels: list[tuple[str, str, int]] = []

    for row in rows:
        qid = str(row.get("qid", "")).strip()
        doc_id = str(row.get("doc_id", "")).strip()
        if not qid or not doc_id:
            continue

        rel = 0
        if use_label_column and "label" in row:
            rel = parse_label(row.get("label"), default=0)

        # Clamp to the expected 0-3 range.
        rel = max(0, min(3, rel))
        qrels.append((qid, doc_id, rel))

    return qrels


def save_qrels(path: Path, qrels: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_id, rel in qrels:
            f.write(f"{qid} 0 {doc_id} {rel}\n")


def write_log(path: Path, rows: list[dict[str, Any]], qrels: list[tuple[str, str, int]]) -> None:
    per_query = defaultdict(int)
    rel_hist = defaultdict(int)

    for qid, _, rel in qrels:
        per_query[qid] += 1
        rel_hist[rel] += 1

    lines = []
    lines.append(f"candidate_pool_rows={len(rows)}\n")
    lines.append(f"qrels_rows={len(qrels)}\n")
    lines.append("per_query_counts:\n")
    for qid in sorted(per_query):
        lines.append(f"  {qid}: {per_query[qid]}\n")
    lines.append("relevance_histogram:\n")
    for rel in sorted(rel_hist):
        lines.append(f"  rel={rel}: {rel_hist[rel]}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_qrels(candidate_pool_path: Path, out_path: Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🔄 Loading candidate pool...")
    rows = load_candidate_pool(candidate_pool_path)

    print("⚙️ Building qrels from candidate pool labels...")
    qrels = build_qrels(rows, use_label_column=True)

    if out_path is None:
        out_path = QRELS_FILE

    print(f"💾 Saving canonical qrels → {out_path}")
    save_qrels(out_path, qrels)

    # Keep the extra files for compatibility with existing scripts.
    # They are identical copies of the canonical qrels unless you later
    # decide to generate separate label schemes.
    print(f"💾 Saving compatibility qrels → {QRELS_KEYWORD}")
    save_qrels(QRELS_KEYWORD, qrels)

    print(f"💾 Saving compatibility qrels → {QRELS_COUNT}")
    save_qrels(QRELS_COUNT, qrels)

    print(f"💾 Saving compatibility qrels → {QRELS_RATIO}")
    save_qrels(QRELS_RATIO, qrels)

    print(f"📝 Saving log → {LOG_FILE}")
    write_log(LOG_FILE, rows, qrels)

    print("✅ DONE!")
    print(f"📁 Candidate pool: {candidate_pool_path}")
    print(f"📁 Qrels: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qrels from an existing candidate pool CSV.")
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=DEFAULT_CANDIDATE_POOL,
        help="Path to candidate_pool.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=QRELS_FILE,
        help="Output qrels file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_qrels(candidate_pool_path=args.candidate_pool, out_path=args.out)


if __name__ == "__main__":
    main()
