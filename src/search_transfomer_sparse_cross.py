from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except Exception:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except Exception:
        CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

from config import (
    REVIEW_PROCESSED_FILE,
    QUERY_TEXT_FILE,
    RUNS_SEARCH_TRANSFORMER_DIR,
    TRANSFORMER_SEARCH_TRACE_DIR,
    get_transformer_paths,
)

from search_transformer_no_cross import load_index, load_queries

DEFAULT_BI_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SparseCandidate:
    doc_id: str
    document: str
    score: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_processed_docs() -> tuple[list[str], dict[str, str]]:
    """
    Load cleaned documents from REVIEW_PROCESSED_FILE.

    Returns:
        ordered_doc_ids: list[str]
        doc_texts: dict[str, str]
    """
    ordered_doc_ids: list[str] = []
    doc_texts: dict[str, str] = {}

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                doc_id = str(d.get("doc_id"))
                text = d.get("text", "")
                if doc_id and doc_id != "None":
                    ordered_doc_ids.append(doc_id)
                    doc_texts[doc_id] = text
            except Exception:
                continue

    return ordered_doc_ids, doc_texts


def sparse_tokenize(text: str) -> list[str]:
    """
    Light sparse tokenization:
    - lowercase
    - keep word tokens
    - no stemming
    - no stopword removal

    This keeps sparse retrieval usable without heavily distorting text.
    """
    return TOKEN_RE.findall((text or "").lower())


class SparseRetriever:
    def __init__(self, doc_ids: list[str], doc_texts: dict[str, str]) -> None:
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.documents = [doc_texts.get(doc_id, "") for doc_id in doc_ids]

        self.mode = None
        self.bm25 = None
        self.vectorizer = None
        self.matrix = None
        self.doc_tokens = [sparse_tokenize(text) for text in self.documents]

        if BM25Okapi is not None:
            self.mode = "bm25"
            self.bm25 = BM25Okapi(self.doc_tokens)
        elif TfidfVectorizer is not None:
            self.mode = "tfidf"
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                tokenizer=sparse_tokenize,
                preprocessor=None,
                token_pattern=None,
                ngram_range=(1, 2),
                min_df=2,
                norm="l2",
            )
            self.matrix = self.vectorizer.fit_transform(self.documents)
        else:
            raise RuntimeError(
                "No sparse backend available. Install either rank_bm25 or scikit-learn."
            )

    def search(self, query: str, top_k: int = 100) -> list[SparseCandidate]:
        if not self.documents:
            return []

        top_k = min(top_k, len(self.documents))
        if top_k <= 0:
            return []

        if self.mode == "bm25" and self.bm25 is not None:
            q_tokens = sparse_tokenize(query)
            scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)
        elif self.mode == "tfidf" and self.vectorizer is not None and self.matrix is not None:
            q_vec = self.vectorizer.transform([query])
            scores = (q_vec @ self.matrix.T).toarray().ravel().astype(np.float32)
        else:
            return []

        if scores.size == 0:
            return []

        if top_k >= scores.size:
            idx = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -top_k)[-top_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]

        results: list[SparseCandidate] = []
        for i in idx[:top_k]:
            doc_id = self.doc_ids[int(i)]
            results.append(
                SparseCandidate(
                    doc_id=doc_id,
                    document=self.doc_texts.get(doc_id, ""),
                    score=float(scores[int(i)]),
                )
            )
        return results


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-8:
        return [0.0 for _ in values]
    return [float((x - vmin) / (vmax - vmin)) for x in values]


def rerank(
    model_name: str = DEFAULT_BI_MODEL,
    reranker_model: str = DEFAULT_RERANKER,
    dense_candidate_size: int = 100,
    sparse_candidate_size: int = 100,
    top_k: int = 100,
    device: str | None = None,
) -> None:
    index_paths = get_transformer_paths(model_name)

    safe_bi = model_name.replace("/", "_").replace("-", "_")
    safe_r = reranker_model.replace("/", "_").replace("-", "_")
    name = (
        f"transformer_dense_sparse_rerank_"
        f"{safe_bi}_{safe_r}_d{dense_candidate_size}_s{sparse_candidate_size}"
    )

    run_file = RUNS_SEARCH_TRANSFORMER_DIR / f"{name}.txt"
    trace_file = TRANSFORMER_SEARCH_TRACE_DIR / f"{name}.txt"

    ensure_dir(RUNS_SEARCH_TRANSFORMER_DIR)
    ensure_dir(TRANSFORMER_SEARCH_TRACE_DIR)

    print("📥 Loading transformer index...")
    searcher = load_index(index_paths)

    print("📥 Loading sparse corpus...")
    ordered_doc_ids, processed_docs = load_processed_docs()
    sparse_retriever = SparseRetriever(ordered_doc_ids, processed_docs)

    print("📥 Loading reranker model...")
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder not available. Install sentence-transformers.")
    reranker = CrossEncoder(reranker_model, device=device)

    queries = load_queries()

    with open(run_file, "w", encoding="utf-8") as run_f, open(trace_file, "w", encoding="utf-8") as trace_f:
        trace_f.write(
            f"===== DENSE + SPARSE + CROSS RERANKER ({model_name} -> {reranker_model}) =====\n\n"
        )

        for qid, q in queries:
            # Dense retrieval: keep query close to its natural form
            dense_candidates = searcher.search(q, top_k=dense_candidate_size) or []

            # Sparse retrieval: light lexical tokenization
            sparse_candidates = sparse_retriever.search(q, top_k=sparse_candidate_size) or []

            # Merge candidates from both sources
            merged: dict[str, dict[str, object]] = {}

            def add_candidate(
                doc_id: str,
                document: str,
                dense_score: float | None = None,
                sparse_score: float | None = None,
                source: str = "",
            ) -> None:
                entry = merged.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "document": document,
                        "dense_score": None,
                        "sparse_score": None,
                        "sources": set(),
                    },
                )
                if document and not entry["document"]:
                    entry["document"] = document
                if dense_score is not None:
                    entry["dense_score"] = dense_score
                if sparse_score is not None:
                    entry["sparse_score"] = sparse_score
                if source:
                    entry["sources"].add(source)

            for cand in dense_candidates:
                add_candidate(
                    doc_id=str(cand.doc_id),
                    document=getattr(cand, "document", "") or processed_docs.get(str(cand.doc_id), ""),
                    dense_score=float(cand.score),
                    source="DENSE",
                )

            for cand in sparse_candidates:
                add_candidate(
                    doc_id=str(cand.doc_id),
                    document=cand.document or processed_docs.get(str(cand.doc_id), ""),
                    sparse_score=float(cand.score),
                    source="SPARSE",
                )

            if not merged:
                continue

            # Build a light hybrid score for ordering before reranking
            merged_items = list(merged.values())
            dense_vals = [float(x["dense_score"]) for x in merged_items if x["dense_score"] is not None]
            sparse_vals = [float(x["sparse_score"]) for x in merged_items if x["sparse_score"] is not None]

            dense_norm_map: dict[str, float] = {}
            sparse_norm_map: dict[str, float] = {}

            if dense_vals:
                dense_normed = normalize_scores(dense_vals)
                j = 0
                for item in merged_items:
                    if item["dense_score"] is not None:
                        dense_norm_map[item["doc_id"]] = dense_normed[j]
                        j += 1

            if sparse_vals:
                sparse_normed = normalize_scores(sparse_vals)
                j = 0
                for item in merged_items:
                    if item["sparse_score"] is not None:
                        sparse_norm_map[item["doc_id"]] = sparse_normed[j]
                        j += 1

            for item in merged_items:
                dnorm = dense_norm_map.get(item["doc_id"], 0.0)
                snorm = sparse_norm_map.get(item["doc_id"], 0.0)
                item["hybrid_score"] = dnorm + snorm

            # Cross-encoder reranks the merged pool
            merged_items.sort(key=lambda x: float(x["hybrid_score"]), reverse=True)

            pairs = []
            doc_ids = []
            dense_scores = []
            sparse_scores = []
            hybrid_scores = []

            for item in merged_items:
                doc_id = str(item["doc_id"])
                doc_text = str(item["document"] or processed_docs.get(doc_id, ""))
                pairs.append((q, doc_text))
                doc_ids.append(doc_id)
                dense_scores.append(item["dense_score"])
                sparse_scores.append(item["sparse_score"])
                hybrid_scores.append(float(item["hybrid_score"]))

            rerank_scores = reranker.predict(pairs, batch_size=32)
            rerank_scores = [float(s) for s in rerank_scores]

            items = list(
                zip(
                    doc_ids,
                    dense_scores,
                    sparse_scores,
                    hybrid_scores,
                    rerank_scores,
                )
            )
            items.sort(key=lambda x: x[4], reverse=True)

            final_k = min(top_k, len(items))
            for rank_idx, (doc_id, dscore, sscore, hscore, rscore) in enumerate(items[:final_k], start=1):
                run_f.write(f"{qid} Q0 {doc_id} {rank_idx} {rscore:.4f} CROSS_{safe_r}\n")

            trace_f.write(f"Query: {q}\n")
            for rank_idx, (doc_id, dscore, sscore, hscore, rscore) in enumerate(items[:10], start=1):
                trace_f.write(
                    f"  Rank {rank_idx}: Doc {doc_id} | "
                    f"Dense: {0.0 if dscore is None else float(dscore):.4f} | "
                    f"Sparse: {0.0 if sscore is None else float(sscore):.4f} | "
                    f"Hybrid: {hscore:.4f} | "
                    f"Rerank: {rscore:.4f}\n"
                )

            trace_f.write("\n" + "-" * 60 + "\n\n")

    print(f"✅ Saved → {run_file}")
    print(f"✅ Trace → {trace_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_BI_MODEL, help="bi-encoder model for initial retrieval")
    parser.add_argument("--reranker", type=str, default=DEFAULT_RERANKER, help="cross-encoder reranker model")
    parser.add_argument("--dense-candidates", type=int, default=100, help="number of dense candidates")
    parser.add_argument("--sparse-candidates", type=int, default=100, help="number of sparse candidates")
    parser.add_argument("--top-k", type=int, default=100, help="number of final results to write")
    parser.add_argument("--device", type=str, default=None, help="device for reranker (e.g., cpu or cuda)")

    args = parser.parse_args()
    rerank(
        model_name=args.model,
        reranker_model=args.reranker,
        dense_candidate_size=args.dense_candidates,
        sparse_candidate_size=args.sparse_candidates,
        top_k=args.top_k,
        device=args.device,
    )