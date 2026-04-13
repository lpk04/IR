import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    QUERY_TEXT_FILE,
    RUNS_RRF_DIR,
    TRACE_DIR,
    get_bm25_paths,
    get_transformer_paths,
)
from prepare_data import preprocess_text
from search_transformer import (
    DEFAULT_MODEL_NAME,
    load_index as load_transformer_index,
    preprocess_project_query,
)


RRF_K = 60
HYBRID_TRACE_DIR = TRACE_DIR / "search_hybrid"


# =========================
# PARSE ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to save.")
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help="RRF constant used to fuse BM25 and transformer rankings.",
    )
    return parser.parse_args()


# =========================
# ENSURE DIR
# =========================
def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Created: {path}")
    else:
        print(f"⚠️ Exists: {path}")


# =========================
# LOAD QUERIES
# =========================
def load_queries():
    queries = []

    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            qid, query = parts
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
# SCORE HELPERS
# =========================
def rank_documents(scores, doc_ids, top_k):
    if len(doc_ids) == 0:
        return []

    k = min(int(top_k), len(doc_ids))
    idx = np.argsort(scores)[::-1][:k]

    return [(str(doc_ids[i]), float(scores[i])) for i in idx]


def reciprocal_rank_fusion(bm25_ranked, transformer_ranked, rrf_k):
    fused_scores = defaultdict(float)

    for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
        fused_scores[doc_id] += 1.0 / (rrf_k + rank)

    for rank, (doc_id, _) in enumerate(transformer_ranked, start=1):
        fused_scores[doc_id] += 1.0 / (rrf_k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# =========================
# RUN SEARCH
# =========================
def run_search(k1=1.2, b=0.75, model_name=DEFAULT_MODEL_NAME, top_k=100, rrf_k=RRF_K):
    ensure_dir(RUNS_RRF_DIR)
    ensure_dir(HYBRID_TRACE_DIR)

    bm25, bm25_doc_ids = load_bm25_index(k1, b)
    transformer_searcher = load_transformer_index(get_transformer_paths(model_name))
    queries = load_queries()

    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    run_path = RUNS_RRF_DIR / f"hybrid_bm25_{k1}_{b}_transformer_{safe_model_name}.txt"
    trace_path = HYBRID_TRACE_DIR / f"search_hybrid_bm25_{k1}_{b}_transformer_{safe_model_name}.txt"

    with open(run_path, "w", encoding="utf-8") as run_f, open(
        trace_path, "w", encoding="utf-8"
    ) as trace_f:
        trace_f.write(
            f"===== SEARCH HYBRID BM25 + TRANSFORMER (k1={k1}, b={b}, model={model_name}, rrf_k={rrf_k}) =====\n\n"
        )

        for qid, query in queries:
            bm25_query = preprocess_text(query)
            bm25_tokens = bm25_query.split()

            bm25_scores = bm25.get_scores(bm25_tokens)
            bm25_ranked = rank_documents(bm25_scores, bm25_doc_ids, top_k)

            transformer_query = preprocess_project_query(query)
            transformer_results = transformer_searcher.search(transformer_query, top_k=top_k)
            transformer_ranked = [
                (result.doc_id, float(result.score)) for result in transformer_results
            ]

            fused_ranked = reciprocal_rank_fusion(bm25_ranked, transformer_ranked, rrf_k)

            for rank, (doc_id, fused_score) in enumerate(fused_ranked[:top_k], start=1):
                run_f.write(
                    f"{qid} Q0 {doc_id} {rank} {fused_score:.6f} HYBRID\n"
                )

            trace_f.write(f"Query: {query}\n")
            trace_f.write(f"BM25 clean: {bm25_query}\n")
            trace_f.write(f"BM25 tokens: {bm25_tokens}\n")
            trace_f.write(f"Transformer clean: {transformer_query}\n")

            trace_f.write("Top BM25:\n")
            for rank, (doc_id, score) in enumerate(bm25_ranked[:5], start=1):
                trace_f.write(f"  Rank {rank}: Doc {doc_id} | Score: {score:.4f}\n")

            trace_f.write("Top Transformer:\n")
            for rank, (doc_id, score) in enumerate(transformer_ranked[:5], start=1):
                trace_f.write(f"  Rank {rank}: Doc {doc_id} | Score: {score:.4f}\n")

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
        model_name=args.model,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )
    print(f"Saved run -> {paths['run']}")
    print(f"Saved trace -> {paths['trace']}")
