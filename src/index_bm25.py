import json
import pickle
import sys

from rank_bm25 import BM25Okapi
from config import REVIEW_PROCESSED_FILE, INDEX_DIR


def build(k1, b):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    corpus = []
    ids = []

    print(f"📥 Building BM25 (k1={k1}, b={b})...")

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["text"].split())
            ids.append(data["doc_id"])

    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    with open(INDEX_DIR / f"bm25_{k1}_{b}.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(INDEX_DIR / "bm25_doc_ids.pkl", "wb") as f:
        pickle.dump(ids, f)

    print(f"✅ Saved → bm25_{k1}_{b}.pkl")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/index_bm25.py <k1> <b>")
        exit()

    k1 = float(sys.argv[1])
    b = float(sys.argv[2])

    build(k1, b)