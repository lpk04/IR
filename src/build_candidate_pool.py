"""
Build a candidate pool for manual relevance labeling.

For each query in `queries.txt`, this script collects:
- top 20 from BM25              (run/runs_search_bm25/)
- top 20 from TF-IDF            (run/runs_search_tfidf/)
- top 20 from Transformer       (run/runs_search_transformer/)
- top 20 from RRF               (run/runs_rrf/)
- 10 random documents from the corpus

It merges duplicates, preserves source/rank provenance, and writes:
    data/candidate_pool.csv

Run:
    python src/build_candidate_pool.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# ===================== CONFIG =====================

ROOT = Path(__file__).resolve().parent.parent

QUERIES_FILE = ROOT / "queries.txt"

# Use cleaned raw file as the corpus source for candidate pool display
CORPUS_FILE = ROOT / "data" / "raw" / "yelp_reviews_100000_changed.jsonl"

# Run directories/files for each retriever
RUNS_BM25_DIR = ROOT / "run" / "runs_search_bm25"
RUNS_TFIDF_DIR = ROOT / "run" / "runs_search_tfidf"
RUNS_TRANSFORMER_DIR = ROOT / "run" / "runs_search_transformer"
RUNS_RRF_DIR = ROOT / "run" / "runs_rrf"

OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "candidate_pool.csv"


# ===================== UTILITIES =====================

def load_queries(path: Path) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            parts = raw.split("\t", maxsplit=1)
            if len(parts) == 2:
                qid, query = parts[0].strip(), parts[1].strip()
            else:
                qid, query = f"Q{i}", parts[0].strip()

            if not qid:
                qid = f"Q{i}"

            queries.append((qid, query))
    return queries


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def load_corpus(path: Path) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    """
    Load the cleaned corpus from JSONL.

    Returns:
        corpus_doc_ids: ordered list of doc IDs
        doc_meta: doc_id -> metadata dict
        line_to_doc_id: 1-based line number (string) -> doc_id
    """
    corpus_doc_ids: list[str] = []
    doc_meta: dict[str, dict[str, Any]] = {}
    line_to_doc_id: dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(d, dict):
                continue

            doc_id = d.get("doc_id", "")
            if doc_id in (None, ""):
                doc_id = str(len(corpus_doc_ids))

            doc_id = str(doc_id)

            text = normalize_text(d.get("text", ""))
            rating = d.get("rating", None)
            sentiment = d.get("sentiment", None)

            corpus_doc_ids.append(doc_id)
            line_to_doc_id[str(line_no)] = doc_id
            doc_meta[doc_id] = {
                "text": text,
                "rating": rating,
                "sentiment": sentiment,
            }

    return corpus_doc_ids, doc_meta, line_to_doc_id


def parse_run_file(path: Path) -> dict[str, dict[str, int]]:
    """
    Parse a TREC-style run file.

    Returns:
        qid -> {docid -> best_rank}
    """
    qid_doc_rank: dict[str, dict[str, int]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            qid = parts[0]
            docid = parts[2]
            try:
                rank = int(parts[3])
            except ValueError:
                rank = 10**9

            existing = qid_doc_rank[qid].get(docid)
            if existing is None or rank < existing:
                qid_doc_rank[qid][docid] = rank

    return qid_doc_rank


def parse_run_collection(path: Path) -> dict[str, dict[str, int]]:
    """
    If `path` is a file, parse it directly.
    If `path` is a directory, parse all .txt files inside and keep best rank per doc.
    """
    merged: dict[str, dict[str, int]] = defaultdict(dict)

    if not path.exists():
        return merged

    if path.is_file():
        source = parse_run_file(path)
        for qid, doc_ranks in source.items():
            for docid, rank in doc_ranks.items():
                existing = merged[qid].get(docid)
                if existing is None or rank < existing:
                    merged[qid][docid] = rank
        return merged

    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".txt"):
            continue
        fp = path / fname
        source = parse_run_file(fp)
        for qid, doc_ranks in source.items():
            for docid, rank in doc_ranks.items():
                existing = merged[qid].get(docid)
                if existing is None or rank < existing:
                    merged[qid][docid] = rank

    return merged


def resolve_doc_id(docid: str, line_to_doc_id: dict[str, str], doc_meta: dict[str, dict[str, Any]]) -> str:
    """
    Resolve docid from run files to corpus doc_id.

    Handles:
    - direct doc_id matches
    - line-number based ids
    """
    if docid in doc_meta:
        return docid

    if docid in line_to_doc_id:
        return line_to_doc_id[docid]

    stripped = str(docid).strip()
    if stripped in doc_meta:
        return stripped

    if stripped in line_to_doc_id:
        return line_to_doc_id[stripped]

    return stripped


def top_k_unique_from_run(
    run_map: dict[str, dict[str, int]],
    qid: str,
    top_k: int,
    line_to_doc_id: dict[str, str],
    doc_meta: dict[str, dict[str, Any]],
) -> list[tuple[str, int]]:
    """
    Return top_k unique docs for a given qid, sorted by rank ascending.
    """
    doc_ranks = run_map.get(qid, {})
    items = []

    for docid, rank in doc_ranks.items():
        resolved = resolve_doc_id(docid, line_to_doc_id, doc_meta)
        items.append((resolved, rank))

    items.sort(key=lambda x: (x[1], x[0]))

    seen = set()
    top_docs: list[tuple[str, int]] = []
    for doc_id, rank in items:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        top_docs.append((doc_id, rank))
        if len(top_docs) >= top_k:
            break

    return top_docs


def stable_rng_for_qid(qid: str) -> random.Random:
    """
    Deterministic RNG per query id.
    """
    digest = hashlib.md5(qid.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def auto_label_candidate(info: dict[str, Any], is_random: bool = False) -> int:
    """
    Heuristic graded relevance label suggestion.

    0 = not relevant
    1 = little relevant
    2 = relevant
    3 = very relevant
    """
    if is_random:
        return 0

    sources = info.get("sources", set())
    ranks = info.get("ranks", {})
    source_count = len(sources)
    best_rank = min(ranks.values()) if ranks else 10**9

    if source_count >= 3 or (source_count >= 2 and best_rank <= 10):
        return 3
    if source_count >= 2 or best_rank <= 20:
        return 2
    if source_count >= 1:
        return 1
    return 0


# ===================== CANDIDATE COLLECTION =====================

def collect_candidates(
    queries: list[tuple[str, str]],
    corpus_doc_ids: list[str],
    doc_meta: dict[str, dict[str, Any]],
    line_to_doc_id: dict[str, str],
    top_bm25: int = 20,
    top_tfidf: int = 20,
    top_transformer: int = 20,
    top_rrf: int = 20,
    random_docs: int = 10,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Build per-query candidate pools from:
    - BM25 top 20          (runs_search_bm25/)
    - TF-IDF top 20        (runs_search_tfidf/)
    - Transformer top 20   (runs_search_transformer/)
    - RRF top 20           (runs_rrf/)
    - 10 random docs from corpus
    """
    candidate_sources = {
        "BM25":        (RUNS_BM25_DIR,       top_bm25),
        "TFIDF":       (RUNS_TFIDF_DIR,      top_tfidf),
        "TRANSFORMER": (RUNS_TRANSFORMER_DIR, top_transformer),
        "RRF":         (RUNS_RRF_DIR,         top_rrf),
    }

    # qid -> doc_id -> candidate info
    candidates: dict[str, dict[str, dict[str, Any]]] = {
        qid: {} for qid, _ in queries
    }

    parsed_runs: dict[str, dict[str, dict[str, int]]] = {}

    for source_label, (source_path, _) in candidate_sources.items():
        parsed_runs[source_label] = parse_run_collection(source_path)

    for qid, _query in queries:
        pool = candidates[qid]

        # Merge runs from each system
        for source_label, (_, k) in candidate_sources.items():
            top_docs = top_k_unique_from_run(
                run_map=parsed_runs[source_label],
                qid=qid,
                top_k=k,
                line_to_doc_id=line_to_doc_id,
                doc_meta=doc_meta,
            )

            for doc_id, rank in top_docs:
                entry = pool.setdefault(
                    doc_id,
                    {
                        "sources": set(),
                        "ranks": {},
                        "is_random": False,
                    },
                )
                entry["sources"].add(source_label)
                entry["ranks"][source_label] = rank

        # Add 10 random corpus documents not already in the pool
        rng = stable_rng_for_qid(qid)
        existing_doc_ids = set(pool.keys())
        available = [doc_id for doc_id in corpus_doc_ids if doc_id not in existing_doc_ids]

        if available:
            sample_size = min(random_docs, len(available))
            sampled = rng.sample(available, sample_size)

            for doc_id in sampled:
                entry = pool.setdefault(
                    doc_id,
                    {
                        "sources": set(),
                        "ranks": {},
                        "is_random": True,
                    },
                )
                entry["sources"].add("RANDOM")
                entry["ranks"]["RANDOM"] = None
                entry["is_random"] = True

    return candidates


# ===================== OUTPUT =====================

def sort_pool_items(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """
    Sort by:
    1. more sources first
    2. lower best rank
    3. random docs later
    4. doc_id
    """
    def sort_key(item: tuple[str, dict[str, Any]]):
        doc_id, info = item
        sources = info.get("sources", set())
        ranks = info.get("ranks", {})
        is_random = bool(info.get("is_random", False))
        numeric_ranks = [r for r in ranks.values() if isinstance(r, int)]
        best_rank = min(numeric_ranks) if numeric_ranks else 10**9
        return (-len(sources), is_random, best_rank, doc_id)

    return sorted(items, key=sort_key)


def write_candidate_csv(
    queries: list[tuple[str, str]],
    candidates: dict[str, dict[str, dict[str, Any]]],
    doc_meta: dict[str, dict[str, Any]],
    out_csv_path: Path,
) -> None:
    fields = [
        "qid",
        "query",
        "doc_id",
        "rating",
        "sentiment",
        "sources",
        "ranks",
        "label",
        "is_random",
        "text",
    ]

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for qid, query in queries:
            pool = candidates.get(qid, {})
            ordered = sort_pool_items(list(pool.items()))

            for doc_id, info in ordered:
                meta = doc_meta.get(str(doc_id), {})
                text = normalize_text(meta.get("text", ""))
                rating = meta.get("rating", None)
                sentiment = meta.get("sentiment", None)
                is_random = bool(info.get("is_random", False))

                writer.writerow({
                    "qid": qid,
                    "query": query,
                    "doc_id": doc_id,
                    "rating": rating,
                    "sentiment": sentiment,
                    "sources": ";".join(sorted(info.get("sources", set()))),
                    "ranks": safe_json_dumps(info.get("ranks", {})),
                    "label": auto_label_candidate(info, is_random=is_random),
                    "is_random": int(is_random),
                    "text": text,
                })


# ===================== MAIN =====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate pool for manual relevance labeling.")
    parser.add_argument("--queries", type=Path, default=QUERIES_FILE)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--bm25-top", type=int, default=20)
    parser.add_argument("--tfidf-top", type=int, default=20)
    parser.add_argument("--transformer-top", type=int, default=20)
    parser.add_argument("--rrf-top", type=int, default=20)
    parser.add_argument("--random-docs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🔄 Loading queries...")
    queries = load_queries(args.queries)

    print("📚 Loading corpus...")
    corpus_doc_ids, doc_meta, line_to_doc_id = load_corpus(args.corpus)

    print("🔍 Collecting candidates from runs...")
    candidates = collect_candidates(
        queries=queries,
        corpus_doc_ids=corpus_doc_ids,
        doc_meta=doc_meta,
        line_to_doc_id=line_to_doc_id,
        top_bm25=args.bm25_top,
        top_tfidf=args.tfidf_top,
        top_transformer=args.transformer_top,
        top_rrf=args.rrf_top,
        random_docs=args.random_docs,
    )

    print(f"💾 Writing candidate CSV → {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_candidate_csv(queries, candidates, doc_meta, args.out)

    print("✅ Done.")
    print("   The CSV includes BM25, TF-IDF, Transformer, RRF, and random corpus documents per query.")
    print("   The `label` column is only a heuristic suggestion for manual review.")


if __name__ == "__main__":
    main()