import json
import pickle
import argparse

from rank_bm25 import BM25Okapi

from config import (
    REVIEW_PROCESSED_FILE,
    BM25_INDEX_DIR,
    BM25_TRACE_DIR,
    get_bm25_paths
)


# =========================
# PARSE ARGUMENT
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k1", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)

    return parser.parse_args()


# =========================
# ENSURE DIR
# =========================
def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Created: {path}")
    else:
        print(f"⚠️ Exists: {path}")


# =========================
# BUILD BM25
# =========================
def build_index(k1, b):
    ensure_dir(BM25_INDEX_DIR)
    ensure_dir(BM25_TRACE_DIR)

    paths = get_bm25_paths(k1, b)

    corpus = []
    doc_ids = []

    print(f"📥 Building BM25 (k1={k1}, b={b})...")

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data["text"].split()

            corpus.append(tokens)
            doc_ids.append(data["doc_id"])

    # =========================
    # BUILD MODEL
    # =========================
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    # =========================
    # SAVE
    # =========================
    with open(paths["model"], "wb") as f:
        pickle.dump(bm25, f)

    with open(paths["ids"], "wb") as f:
        pickle.dump(doc_ids, f)

    # =========================
    # TRACE (for report)
    # =========================
    with open(paths["trace"], "w", encoding="utf-8") as f:
        f.write(f"===== BM25 INDEX (k1={k1}, b={b}) =====\n\n")

        f.write(f"Total documents: {len(doc_ids)}\n")
        f.write(f"Average doc length: {bm25.avgdl:.2f}\n\n")

        f.write("Sample documents (first 3):\n")
        for i in range(min(3, len(corpus))):
            f.write(f"Doc {i}: {' '.join(corpus[i][:10])}\n")

    print(f"✅ Saved → {paths['model']}")
    print(f"📝 Trace → {paths['trace']}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    build_index(args.k1, args.b)