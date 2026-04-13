import argparse
import json
import pickle
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except Exception:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except Exception:
        CrossEncoder = None

from config import (
    REVIEW_PROCESSED_FILE,
    QUERY_TEXT_FILE,
    RUNS_SEARCH_TRANSFORMER_DIR,
    TRANSFORMER_SEARCH_TRACE_DIR,
    get_transformer_paths,
)

from search_transformer import load_index, load_queries, preprocess_project_query


DEFAULT_BI_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_processed_docs() -> dict:
    docs = {}
    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                docs[str(d.get("doc_id"))] = d.get("text", "")
            except Exception:
                continue
    return docs


def rerank(
    model_name: str = DEFAULT_BI_MODEL,
    reranker_model: str = DEFAULT_RERANKER,
    candidate_size: int = 100,
    top_k: int = 100,
    device: str | None = None,
) -> None:
    index_paths = get_transformer_paths(model_name)

    safe_bi = model_name.replace("/", "_").replace("-", "_")
    safe_r = reranker_model.replace("/", "_").replace("-", "_")
    name = f"transformer_rerank_{safe_bi}_{safe_r}_c{candidate_size}"

    run_file = RUNS_SEARCH_TRANSFORMER_DIR / f"{name}.txt"
    trace_file = TRANSFORMER_SEARCH_TRACE_DIR / f"{name}.txt"

    ensure_dir(RUNS_SEARCH_TRANSFORMER_DIR)
    ensure_dir(TRANSFORMER_SEARCH_TRACE_DIR)

    print("📥 Loading transformer index...")
    searcher = load_index(index_paths)

    print("📥 Loading reranker model...")
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder not available - install sentence-transformers package")

    reranker = CrossEncoder(reranker_model, device=device)

    queries = load_queries()
    processed_docs = load_processed_docs()

    with open(run_file, "w", encoding="utf-8") as run_f, open(trace_file, "w", encoding="utf-8") as trace_f:
        trace_f.write(f"===== TRANSFORMER RERANKER ({model_name} -> {reranker_model}) =====\n\n")

        for qid, q in queries:
            q_for_search = preprocess_project_query(q)

            # retrieve candidates from bi-encoder index
            candidates = searcher.search(q_for_search, top_k=candidate_size)
            if not candidates:
                continue

            pairs = []
            doc_ids = []
            orig_scores = []

            for cand in candidates:
                doc_text = processed_docs.get(cand.doc_id, cand.document)
                pairs.append((q, doc_text))
                doc_ids.append(cand.doc_id)
                orig_scores.append(cand.score)

            # predict rerank scores
            rerank_scores = reranker.predict(pairs, batch_size=32)

            items = list(zip(doc_ids, orig_scores, rerank_scores))
            items.sort(key=lambda x: x[2], reverse=True)

            final_k = min(top_k, len(items))
            for rank_idx, (doc_id, orig_score, rscore) in enumerate(items[:final_k], start=1):
                run_f.write(f"{qid} Q0 {doc_id} {rank_idx} {rscore:.4f} CROSS_{safe_r}\n")

            trace_f.write(f"Query: {q}\n")
            for rank_idx, (doc_id, orig_score, rscore) in enumerate(items[:10], start=1):
                trace_f.write(f"  Rank {rank_idx}: Doc {doc_id} | Orig: {orig_score:.4f} | Rerank: {rscore:.4f}\n")

            trace_f.write("\n" + "-" * 50 + "\n\n")

    print(f"✅ Saved → {run_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_BI_MODEL, help="bi-encoder model for initial retrieval")
    parser.add_argument("--reranker", type=str, default=DEFAULT_RERANKER, help="cross-encoder reranker model")
    parser.add_argument("--candidates", type=int, default=100, help="number of candidates to rerank")
    parser.add_argument("--top-k", type=int, default=100, help="number of final results to write")
    parser.add_argument("--device", type=str, default=None, help="device for reranker (e.g., cpu or cuda)")

    args = parser.parse_args()
    rerank(args.model, args.reranker, args.candidates, args.top_k, device=args.device)
