from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import (
    QUERY_TEXT_FILE,
    RUNS_RRF_DIR,
    TRACE_DIR,
    RUNS_SEARCH_BM25_DIR,
    RUNS_SEARCH_TRANSFORMER_DIR,
    get_bm25_paths,
)
from prepare_data import preprocess_text

RRF_K = 60
HYBRID_TRACE_DIR = TRACE_DIR / "search_hybrid"


# =========================
# PARSE ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to save.")
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help="RRF constant used to fuse rankings.",
    )
    parser.add_argument(
        "--transformer-top-k",
        type=int,
        default=100,
        help="Top-k to read from each transformer run file.",
    )
    return parser.parse_args()


# =========================
# ENSURE DIR
# =========================
def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {path}")


# =========================
# LOAD QUERIES
# =========================
def load_queries():
    queries = []
    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                qid, query = parts
            else:
                qid, query = f"Q{i}", line.strip()
            if query:
                queries.append((qid, query))
    return queries


# =========================
# LOAD BM25 INDEX
# =========================
def load_bm25_index(k1, b):
    paths = get_bm25_paths(k1, b)

    with open(paths["model"], "rb") as f:
        bm25 = pickle.load(f)

    with open(paths["ids"], "rb") as f:
        doc_ids = pickle.load(f)

    return bm25, doc_ids


# =========================
# LOAD RUN FILES
# =========================
def load_top_k_run(file_path: Path, k: int) -> Dict[str, List[str]]:
    """
    Parse a TREC-style run file and keep top-k docids per query.
    """
    top_docs = defaultdict(list)

    if not file_path.exists():
        return top_docs

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            qid = parts[0]
            doc_id = parts[2]

            if len(top_docs[qid]) < k:
                top_docs[qid].append(doc_id)

    return top_docs


def load_all_transformer_runs(top_k: int) -> Dict[str, Dict[str, List[str]]]:
    """
    Load every .txt run file in RUNS_SEARCH_TRANSFORMER_DIR.

    Returns:
        {
            "file_name_1.txt": {"Q1": [docid, ...], ...},
            "file_name_2.txt": {"Q1": [docid, ...], ...},
            ...
        }
    """
    all_runs: Dict[str, Dict[str, List[str]]] = {}

    if not RUNS_SEARCH_TRANSFORMER_DIR.exists():
        return all_runs

    for fname in sorted(os.listdir(RUNS_SEARCH_TRANSFORMER_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = RUNS_SEARCH_TRANSFORMER_DIR / fname
        all_runs[fname] = load_top_k_run(path, top_k)

    return all_runs


# =========================
# SCORE HELPERS
# =========================
def rank_documents(scores, doc_ids, top_k):
    if len(doc_ids) == 0:
        return []

    k = min(int(top_k), len(doc_ids))
    idx = np.argsort(scores)[::-1][:k]
    return [(str(doc_ids[i]), float(scores[i])) for i in idx]


def reciprocal_rank_fusion(rankings, rrf_k):
    """
    rankings: list of ranked lists, where each ranked list is:
        [(doc_id, score), (doc_id, score), ...]
    """
    fused_scores = defaultdict(float)

    for ranked_list in rankings:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1.0 / (rrf_k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# =========================
# RUN SEARCH
# =========================
def run_search(k1=1.2, b=0.75, top_k=100, rrf_k=RRF_K, transformer_top_k=100):
    ensure_dir(RUNS_RRF_DIR)
    ensure_dir(HYBRID_TRACE_DIR)

    bm25, bm25_doc_ids = load_bm25_index(k1, b)
    transformer_runs = load_all_transformer_runs(transformer_top_k)
    queries = load_queries()

    run_name = f"hybrid_bm25_{k1}_{b}_all_transformers_rrf"
    run_path = RUNS_RRF_DIR / f"{run_name}.txt"
    trace_path = HYBRID_TRACE_DIR / f"{run_name}.txt"

    with open(run_path, "w", encoding="utf-8") as run_f, open(trace_path, "w", encoding="utf-8") as trace_f:
        trace_f.write(
            f"===== SEARCH HYBRID BM25 + ALL TRANSFORMERS (k1={k1}, b={b}, rrf_k={rrf_k}) =====\n\n"
        )

        for qid, query in queries:
            # BM25 ranking
            bm25_query = preprocess_text(query)
            bm25_tokens = bm25_query.split()
            bm25_scores = bm25.get_scores(bm25_tokens)
            bm25_ranked = rank_documents(bm25_scores, bm25_doc_ids, top_k)

            # Transformer rankings from every run file
            transformer_ranked_lists = []
            transformer_debug = []

            for fname, run_map in transformer_runs.items():
                docs = run_map.get(qid, [])
                ranked = [(doc_id, float(top_k - rank)) for rank, doc_id in enumerate(docs, start=1)]
                # The score value here is only used for debugging / trace.
                # RRF uses the rank position, not the score magnitude.
                transformer_ranked_lists.append([(doc_id, score) for doc_id, score in ranked])
                transformer_debug.append((fname, ranked))

            # Fuse BM25 with all transformer rankings
            all_rankings = [bm25_ranked] + transformer_ranked_lists
            fused_ranked = reciprocal_rank_fusion(all_rankings, rrf_k)

            for rank, (doc_id, fused_score) in enumerate(fused_ranked[:top_k], start=1):
                run_f.write(f"{qid} Q0 {doc_id} {rank} {fused_score:.6f} HYBRID\n")

            # Trace
            trace_f.write(f"Query: {query}\n")
            trace_f.write(f"BM25 clean: {bm25_query}\n")
            trace_f.write(f"BM25 tokens: {bm25_tokens}\n")

            trace_f.write("Top BM25:\n")
            for rank, (doc_id, score) in enumerate(bm25_ranked[:5], start=1):
                trace_f.write(f"  Rank {rank}: Doc {doc_id} | Score: {score:.4f}\n")

            for fname, ranked in transformer_debug:
                trace_f.write(f"Top Transformer ({fname}):\n")
                for rank, (doc_id, score) in enumerate(ranked[:5], start=1):
                    trace_f.write(f"  Rank {rank}: Doc {doc_id} | RankScore: {score:.4f}\n")

            trace_f.write("Top Hybrid:\n")
            for rank, (doc_id, score) in enumerate(fused_ranked[:5], start=1):
                trace_f.write(f"  Rank {rank}: Doc {doc_id} | Score: {score:.6f}\n")

            trace_f.write("\n" + "-" * 50 + "\n\n")

    return {"run": run_path, "trace": trace_path}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()
    paths = run_search(
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        transformer_top_k=args.transformer_top_k,
    )
    print(f"Saved run -> {paths['run']}")
    print(f"Saved trace -> {paths['trace']}")