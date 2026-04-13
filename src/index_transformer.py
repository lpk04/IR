import json
import argparse
from pathlib import Path
import numpy as np

from search_transformer import build_embedding_index, save_index
from config import (
    REVIEW_TRANSFORMER_PROCESSED_FILE,
    TRANSFORMER_INDEX_DIR,
    TRANSFORMER_TRACE_DIR,
    get_transformer_paths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Model name or local path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding computation")
    parser.add_argument("--sample-size", type=int, default=100000, help="If >0, limit to first N documents")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Created: {path}")
    else:
        print(f"⚠️ Exists: {path}")


def build_index(model_name: str, batch_size: int = 8, sample_size: int = 0) -> None:
    ensure_dir(TRANSFORMER_INDEX_DIR)
    ensure_dir(TRANSFORMER_TRACE_DIR)

    paths = get_transformer_paths(model_name)

    documents: list[str] = []
    doc_ids: list[str] = []
    metadata_map: dict[str, dict] = {}

    print(f"📥 Building TRANSFORMER index: model={model_name} sample_size={sample_size} ...")

    with open(REVIEW_TRANSFORMER_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "")
            # prefer review_id if present, fall back to doc_id or the index
            doc_id = data.get("review_id", data.get("doc_id", i))
            documents.append(str(text))
            doc_ids.append(str(doc_id))
            # store metadata (everything except the full text)
            try:
                meta = {k: v for k, v in data.items() if k != "text"}
            except Exception:
                meta = {}
            metadata_map[str(doc_id)] = meta

    # Build embeddings and in-memory searcher
    searcher = build_embedding_index(documents, doc_ids=doc_ids, model_name=model_name, batch_size=batch_size)

    # Persist index files (numpy + pickles + metadata)
    save_index(searcher, paths)

    # Write a readable trace: summary + per-document rows (norm + sample)
    with open(paths["trace"], "w", encoding="utf-8") as trace_f:
        emb_dim = int(searcher.embeddings.shape[1]) if searcher.embeddings.size else 0

        trace_f.write(f"===== TRANSFORMER INDEX ({model_name}) =====\n\n")
        trace_f.write(f"Total documents: {len(doc_ids)}\n")
        trace_f.write(f"Embedding dimension: {emb_dim}\n")
        trace_f.write(f"Embeddings file: {paths['embeddings']}\n\n")

        # Summary stats for the embedding matrix
        emb = getattr(searcher, "embeddings", None)
        if emb is not None and getattr(emb, "size", 0):
            try:
                trace_f.write("Embeddings summary:\n")
                trace_f.write(f"  shape: {emb.shape}\n")
                trace_f.write(f"  dtype: {emb.dtype}\n")
                trace_f.write(f"  mean: {float(np.mean(emb)):.6f}  std: {float(np.std(emb)):.6f}  min: {float(np.min(emb)):.6f}  max: {float(np.max(emb)):.6f}\n\n")
            except Exception:
                trace_f.write("Embeddings summary: (failed to compute stats)\n\n")
        else:
            trace_f.write("Embeddings summary: (no embeddings)\n\n")

        # Per-document table: index, doc_id, norm, first-8 values, some meta
        trace_f.write("Index\tDocID\tNorm\tSample(first8)\tMeta\n")
        trace_f.write("-----\t-----\t----\t--------------\t----\n")

        for idx, doc_id in enumerate(doc_ids):
            meta = metadata_map.get(str(doc_id), {})
            if emb is not None and getattr(emb, "size", 0) and idx < emb.shape[0]:
                row = emb[idx]
                norm = float(np.linalg.norm(row))
                sample = [round(float(x), 6) for x in row[:8].tolist()]
            else:
                norm = 0.0
                sample = []

            # meta summary: include rating/sentiment/line if available
            mparts = []
            if isinstance(meta, dict):
                if "rating" in meta:
                    mparts.append(f"rating={meta['rating']}")
                if "sentiment" in meta:
                    mparts.append(f"sent={meta['sentiment']}")
                if "review_line_no" in meta:
                    mparts.append(f"line={meta['review_line_no']}")

            meta_str = ",".join(mparts) if mparts else ""

            trace_f.write(f"{idx}\t{doc_id}\t{norm:.4f}\t{sample}\t{meta_str}\n")

        trace_f.write("\nNote: full embedding matrix saved at: {}\n".format(paths["embeddings"]))

    print(f"✅ Saved embeddings -> {paths['embeddings']}")
    print(f"✅ Saved ids -> {paths['ids']}")
    print(f"✅ Saved documents -> {paths['documents']}")
    print(f"📝 Trace -> {paths['trace']}")


if __name__ == "__main__":
    args = parse_args()
    build_index(args.model, batch_size=args.batch_size, sample_size=args.sample_size)
