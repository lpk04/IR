from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
import os

import logging

LOGGER = logging.getLogger(__name__)

DEFAULT_FALLBACK_MODEL = "all-MiniLM-L6-v2"

try:
    import torch
except Exception:
    torch = None

try:
    from config import (
        QUERY_TEXT_FILE,
        RUNS_SEARCH_TRANSFORMER_DIR,
        TRANSFORMER_SEARCH_TRACE_DIR,
        get_transformer_paths,
        get_transformer_run_paths,
    )
except ImportError:
    from .config import (
        QUERY_TEXT_FILE,
        RUNS_SEARCH_TRANSFORMER_DIR,
        TRANSFORMER_SEARCH_TRACE_DIR,
        get_transformer_paths,
        get_transformer_run_paths,
    )


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EPSILON = 1e-12


# =========================
# RESULT TYPE
# =========================
@dataclass
class SearchResult:
    doc_id: str
    document: str
    score: float


# =========================
# MODEL CACHE
# =========================
@lru_cache(maxsize=4)
def get_default_device() -> str:
    """Determine the default device for model inference.

    Priority: ENV TRANSFORMER_DEVICE -> CUDA (if available) -> CPU
    """
    env = os.environ.get("TRANSFORMER_DEVICE")
    if env:
        return env
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> SentenceTransformer:
    device = device or get_default_device()
    # Prefer a local model directory when available (e.g. models/all-MiniLM-L6-v2)
    local_candidates: list[Path] = []

    # If the user passed a path-like model_name and it exists, prefer it.
    maybe_path = Path(model_name)
    if maybe_path.exists():
        local_candidates.append(maybe_path)

    # Common download location used by snapshot_download in this repo
    local_models_dir = Path("models")
    local_model_path = local_models_dir / model_name
    if local_model_path.exists():
        local_candidates.append(local_model_path)

    # Try loading any local candidate first
    for candidate in local_candidates:
        try:
            LOGGER.info("Loading SentenceTransformer from local path: %s", candidate)
            model = SentenceTransformer(str(candidate), device=device, trust_remote_code=True)
            try:
                for module in model:
                    if hasattr(module, "unpad_inputs"):
                        try:
                            module.unpad_inputs = False
                            LOGGER.info("Disabled transformer.unpad_inputs for loaded model")
                            print(f"[debug] transformer.unpad_inputs set -> {getattr(module, 'unpad_inputs', None)} on {type(module)}")
                        except Exception:
                            LOGGER.debug("Could not set unpad_inputs on transformer-like module.")
                            print("[debug] could not set transformer.unpad_inputs")
                        # Also defensively wrap the underlying HF model forward to strip 'unpad_inputs'
                        try:
                            hf_model = getattr(module, "model", None)
                            if hf_model is not None and hasattr(hf_model, "forward"):
                                orig_forward = hf_model.forward

                                def _wrapped_forward(*a, **kw):
                                    if "unpad_inputs" in kw:
                                        kw.pop("unpad_inputs")
                                    return orig_forward(*a, **kw)

                                hf_model.forward = _wrapped_forward
                                print("[debug] wrapped underlying HF model.forward to strip 'unpad_inputs'")
                                try:
                                    print(f"[debug] underlying HF model config.unpad_inputs -> {getattr(hf_model.config, 'unpad_inputs', None)}")
                                except Exception:
                                    pass
                        except Exception:
                            LOGGER.debug("Could not wrap underlying HF model.forward")
                        break
            except Exception:
                LOGGER.debug("Could not disable unpad_inputs on transformer module.")
            return model
        except Exception as e_local:
            LOGGER.warning("Failed to load local model at %s: %s", candidate, str(e_local))

    # Fall back to loading by model name from the Hub
    try:
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        try:
            for module in model:
                if hasattr(module, "unpad_inputs"):
                    try:
                        module.unpad_inputs = False
                        LOGGER.info("Disabled transformer.unpad_inputs for loaded model")
                        print(f"[debug] transformer.unpad_inputs set -> {getattr(module, 'unpad_inputs', None)} on {type(module)}")
                    except Exception:
                        LOGGER.debug("Could not set unpad_inputs on transformer-like module.")
                        print("[debug] could not set transformer.unpad_inputs")
                        # Also defensively wrap the underlying HF model forward to strip 'unpad_inputs'
                        try:
                            hf_model = getattr(module, "model", None)
                            if hf_model is not None and hasattr(hf_model, "forward"):
                                orig_forward = hf_model.forward

                                def _wrapped_forward(*a, **kw):
                                    if "unpad_inputs" in kw:
                                        kw.pop("unpad_inputs")
                                    return orig_forward(*a, **kw)

                                hf_model.forward = _wrapped_forward
                                print("[debug] wrapped underlying HF model.forward to strip 'unpad_inputs'")
                                try:
                                    print(f"[debug] underlying HF model config.unpad_inputs -> {getattr(hf_model.config, 'unpad_inputs', None)}")
                                except Exception:
                                    pass
                        except Exception:
                            LOGGER.debug("Could not wrap underlying HF model.forward")
                    break
        except Exception:
            LOGGER.debug("Could not disable unpad_inputs on transformer module.")
        return model
    except Exception as e:
        # Provide guidance when gated/private HF models fail to load and try the configured fallback.
        fallback_model = os.environ.get("TRANSFORMER_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        LOGGER.warning("Could not load SentenceTransformer model '%s': %s", model_name, str(e))
        LOGGER.warning(
            "Attempting fallback model '%s'. To use gated models, authenticate with huggingface (huggingface-cli login) or set HUGGINGFACE_HUB_TOKEN.",
            fallback_model,
        )

        last_exc = e
        # Try fallback as a local path first, then as a hub name
        fallback_local = local_models_dir / fallback_model
        for attempt in (fallback_local, fallback_model):
            try:
                LOGGER.info("Trying fallback model: %s", attempt)
                return SentenceTransformer(str(attempt), device=device, trust_remote_code=True)
            except Exception as e2:
                LOGGER.warning("Fallback attempt '%s' failed: %s", attempt, str(e2))
                last_exc = e2

        LOGGER.error("All attempts to load embedding model failed.")
        raise last_exc from e


# =========================
# NORMALIZATION
# =========================
def normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.size == 0:
        return matrix.reshape(0, 0)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)

    if values.size == 0:
        return values

    norm = max(float(np.linalg.norm(values)), EPSILON)
    return values / norm


# =========================
# SEARCHER
# =========================
class TransformerSearcher:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or get_default_device()
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.normalized_embeddings = np.empty((0, 0), dtype=np.float32)

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model(self.model_name, device=self.device)

    def fit(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str] | None = None,
        batch_size: int = 32,
    ) -> "TransformerSearcher":
        raw_documents = [] if documents is None else list(documents)
        docs = [doc if isinstance(doc, str) else str(doc or "") for doc in raw_documents]

        if doc_ids is None:
            ids = [str(i) for i in range(len(docs))]
        else:
            ids = [str(doc_id) for doc_id in doc_ids]
            if len(ids) != len(docs):
                raise ValueError("doc_ids must match the number of documents.")

        self.documents = docs
        self.doc_ids = ids

        if not docs:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.normalized_embeddings = np.empty((0, 0), dtype=np.float32)
            return self

        embeddings = self.model.encode(
            docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.normalized_embeddings = normalize_matrix(self.embeddings)
        return self

    def encode_query(self, query: str) -> np.ndarray | None:
        if not isinstance(query, str):
            return None

        clean_query = query.strip()
        if not clean_query:
            return None

        embedding = self.model.encode(
            [clean_query],
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )
        return np.asarray(embedding[0], dtype=np.float32)

    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.normalized_embeddings.size == 0:
            return np.empty(0, dtype=np.float32)

        query_vector = normalize_vector(query_embedding)
        scores = self.normalized_embeddings @ query_vector
        return np.asarray(scores, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return []

        if top_k <= 0 or not self.documents:
            return []

        query_embedding = self.encode_query(query)
        if query_embedding is None:
            return []

        scores = self.cosine_similarity(query_embedding)
        if scores.size == 0:
            return []

        top_k = min(int(top_k), len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                doc_id=self.doc_ids[index],
                document=self.documents[index],
                score=float(scores[index]),
            )
            for index in top_indices
        ]

    @classmethod
    def from_precomputed(
        cls,
        documents: Sequence[str],
        embeddings: np.ndarray,
        doc_ids: Sequence[str] | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> "TransformerSearcher":
        searcher = cls(model_name=model_name)

        raw_documents = [] if documents is None else list(documents)
        docs = [doc if isinstance(doc, str) else str(doc or "") for doc in raw_documents]
        ids = [str(i) for i in range(len(docs))] if doc_ids is None else [str(doc_id) for doc_id in doc_ids]

        if len(ids) != len(docs):
            raise ValueError("doc_ids must match the number of documents.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        
        # Handle empty embeddings case
        if matrix.size == 0:
            if len(docs) != 0:
                raise ValueError("embeddings must have the same number of rows as documents (got empty embeddings but have documents).")
            matrix = np.empty((0, 0), dtype=np.float32)
        else:
            # Validate embeddings shape matches document count
            if matrix.ndim < 2:
                raise ValueError(f"embeddings must be 2-dimensional, got {matrix.ndim} dimensions.")
            if matrix.shape[0] != len(docs):
                raise ValueError(
                    f"embeddings must have the same number of rows as documents. "
                    f"Got {matrix.shape[0]} embeddings but {len(docs)} documents."
                )

        searcher.documents = docs
        searcher.doc_ids = ids
        searcher.embeddings = matrix
        searcher.normalized_embeddings = normalize_matrix(matrix)
        return searcher


# =========================
# BUILD INDEX
# =========================
def build_embedding_index(
    documents: Sequence[str],
    doc_ids: Sequence[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> TransformerSearcher:
    return TransformerSearcher(model_name=model_name).fit(
        documents,
        doc_ids=doc_ids,
        batch_size=batch_size,
    )


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Created: {path}")
    else:
        print(f"⚠️ Exists: {path}")


# =========================
# SAVE / LOAD INDEX
# =========================
def save_index(searcher: TransformerSearcher, paths: dict[str, Path]) -> None:
    for key in ("embeddings", "ids", "documents", "meta"):
        ensure_dir(paths[key].parent)

    np.save(paths["embeddings"], searcher.embeddings)

    with open(paths["ids"], "wb") as f:
        pickle.dump(searcher.doc_ids, f)

    with open(paths["documents"], "wb") as f:
        pickle.dump(searcher.documents, f)

    metadata = {
        "model_name": searcher.model_name,
        "num_documents": len(searcher.documents),
        "embedding_dim": int(searcher.embeddings.shape[1]) if searcher.embeddings.size else 0,
    }

    with open(paths["meta"], "wb") as f:
        pickle.dump(metadata, f)


def load_index(paths: dict[str, Path]) -> TransformerSearcher:
    embeddings = np.load(paths["embeddings"])

    with open(paths["ids"], "rb") as f:
        doc_ids = pickle.load(f)

    with open(paths["documents"], "rb") as f:
        documents = pickle.load(f)

    with open(paths["meta"], "rb") as f:
        metadata = pickle.load(f)

    return TransformerSearcher.from_precomputed(
        documents=documents,
        embeddings=embeddings,
        doc_ids=doc_ids,
        model_name=metadata.get("model_name", DEFAULT_MODEL_NAME),
    )


# =========================
# LOAD QUERIES
# =========================
import re

def normalize_qid(qid: str) -> str:
    match = re.search(r"\d+", qid)
    if not match:
        raise ValueError(f"Invalid query id: {qid}")
    return f"Q{match.group()}"


def load_queries() -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []

    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            raw_qid, query = parts
            qid = normalize_qid(raw_qid)

            queries.append((qid, query))

    return queries


# =========================
# QUERY PREPROCESS
# =========================
def preprocess_project_query(query: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        return ""

    try:
        try:
            from prepare_data import preprocess_text
        except ImportError:
            from .prepare_data import preprocess_text
    except Exception:
        return cleaned_query

    try:
        processed_query = preprocess_text(cleaned_query)
    except Exception:
        return cleaned_query

    return processed_query or cleaned_query


# =========================
# RUN SEARCH
# =========================
def run_search(model_name: str = DEFAULT_MODEL_NAME, top_k: int = 100) -> dict[str, Path]:
    index_paths = get_transformer_paths(model_name)
    run_paths = get_transformer_run_paths(model_name)

    ensure_dir(RUNS_SEARCH_TRANSFORMER_DIR)
    ensure_dir(TRANSFORMER_SEARCH_TRACE_DIR)

    searcher = load_index(index_paths)
    queries = load_queries()

    with open(run_paths["run"], "w", encoding="utf-8") as run_f, open(
        run_paths["trace"], "w", encoding="utf-8"
    ) as trace_f:
        trace_f.write(f"===== SEARCH TRANSFORMER ({searcher.model_name}) =====\n\n")

        for qid, query in queries:
            query_for_search = preprocess_project_query(query)
            results = searcher.search(query_for_search, top_k=top_k)

            for rank, result in enumerate(results, start=1):
                run_f.write(
                    f"{qid} Q0 {result.doc_id} {rank} {result.score:.4f} TRANSFORMER\n"
                )

            trace_f.write(f"Query: {query}\n")
            trace_f.write(f"Clean: {query_for_search}\n")

            for rank, result in enumerate(results[:5], start=1):
                trace_f.write(
                    f"  Rank {rank}: Doc {result.doc_id} | Score: {result.score:.4f}\n"
                )

            trace_f.write("\n" + "-" * 50 + "\n\n")

    return run_paths


# =========================
# PARSE ARGUMENTS
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name or local path.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to return.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_search(model_name=args.model, top_k=args.top_k)
    print(f"Saved run -> {paths['run']}")
    print(f"Saved trace -> {paths['trace']}")
