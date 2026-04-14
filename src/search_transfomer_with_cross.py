from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import CrossEncoder
except Exception:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except Exception:
        CrossEncoder = None

from config import (
    REVIEW_PROCESSED_FILE,
    RUNS_SEARCH_TRANSFORMER_DIR,
    TRANSFORMER_SEARCH_TRACE_DIR,
    get_transformer_paths,
)

from search_transformer_no_cross import load_index, load_queries, preprocess_project_query

DEFAULT_BI_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_processed_docs() -> dict[str, str]:
    """
    Load doc_id -> text from the processed Yelp JSONL file.

    Expected schema per line:
        {"doc_id": "...", "text": "...", ...}
    """
    docs: dict[str, str] = {}
    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue

            doc_id = str(d.get("doc_id", "")).strip()
            text = str(d.get("text", "")).strip()
            if doc_id:
                docs[doc_id] = text
    return docs


def rerank(
    model_name: str = DEFAULT_BI_MODEL,
    reranker_model: str = DEFAULT_RERANKER,
    candidate_size: int = 100,
    top_k: int = 100,
    device: str | None = None,
) -> None:
    """
    Dense retrieval using all-MiniLM-L6-v2, then cross-encoder reranking
    using ms-marco-MiniLM-L-6-v2.

    The dense encoder is used to fetch candidates from the existing index.
    The cross encoder scores query-document pairs and reorders the pool.
    """
    index_paths = get_transformer_paths(model_name)

    safe_bi = model_name.replace("/", "_").replace("-", "_")
    safe_r = reranker_model.replace("/", "_").replace("-", "_")
    name = f"transformer_dense_cross_{safe_bi}_{safe_r}_c{candidate_size}"

    run_file = RUNS_SEARCH_TRANSFORMER_DIR / f"{name}.txt"
    trace_file = TRANSFORMER_SEARCH_TRACE_DIR / f"{name}.txt"

    ensure_dir(RUNS_SEARCH_TRANSFORMER_DIR)
    ensure_dir(TRANSFORMER_SEARCH_TRACE_DIR)

    print("📥 Loading dense transformer index...")
    searcher = load_index(index_paths)

    print("📥 Loading reranker model...")
    if CrossEncoder is None:
        raise RuntimeError(
            "CrossEncoder not available. Install sentence-transformers with cross-encoder support."
        )
    reranker = CrossEncoder(reranker_model, device=device)

    print("📥 Loading queries...")
    queries = load_queries()

    print("📥 Loading processed documents...")
    processed_docs = load_processed_docs()

    with open(run_file, "w", encoding="utf-8") as run_f, open(trace_file, "w", encoding="utf-8") as trace_f:
        trace_f.write(
            f"===== DENSE + CROSS RERANKER ({model_name} -> {reranker_model}) =====\n\n"
        )

        for qid, q in queries:
            q_for_search = preprocess_project_query(q)

            # First-stage retrieval from dense index.
            candidates = searcher.search(q_for_search, top_k=candidate_size)
            if not candidates:
                continue

            pairs: list[tuple[str, str]] = []
            doc_ids: list[str] = []
            orig_scores: list[float] = []

            for cand in candidates:
                doc_id = str(cand.doc_id)
                doc_text = processed_docs.get(doc_id, getattr(cand, "document", ""))
                pairs.append((q, doc_text))
                doc_ids.append(doc_id)
                orig_scores.append(float(cand.score))

            # Cross-encoder reranking.
            rerank_scores = reranker.predict(pairs, batch_size=32)
            rerank_scores = [float(s) for s in rerank_scores]

            items = list(zip(doc_ids, orig_scores, rerank_scores))
            items.sort(key=lambda x: x[2], reverse=True)

            final_k = min(top_k, len(items))
            for rank_idx, (doc_id, orig_score, rscore) in enumerate(items[:final_k], start=1):
                run_f.write(f"{qid} Q0 {doc_id} {rank_idx} {rscore:.4f} CROSS_{safe_r}\n")

            trace_f.write(f"Query: {q}\n")
            for rank_idx, (doc_id, orig_score, rscore) in enumerate(items[:10], start=1):
                trace_f.write(
                    f"  Rank {rank_idx}: Doc {doc_id} | Orig: {orig_score:.4f} | Rerank: {rscore:.4f}\n"
                )
            trace_f.write("\n" + "-" * 60 + "\n\n")

    print(f"✅ Saved → {run_file}")
    print(f"✅ Trace → {trace_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BI_MODEL,
        help="bi-encoder model for initial retrieval",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=DEFAULT_RERANKER,
        help="cross-encoder reranker model",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="number of dense candidates to rerank",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="number of final results to write",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device for reranker (e.g., cpu or cuda)",
    )

    args = parser.parse_args()
    rerank(
        model_name=args.model,
        reranker_model=args.reranker,
        candidate_size=args.candidates,
        top_k=args.top_k,
        device=args.device,
    )
