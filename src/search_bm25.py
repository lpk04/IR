import pickle
import numpy as np
import sys

from config import INDEX_DIR, RUNS_DIR, QUERY_TEXT_FILE
from prepare import preprocess_text

RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load(k1, b):
    bm25 = pickle.load(open(INDEX_DIR / f"bm25_{k1}_{b}.pkl", "rb"))
    ids = pickle.load(open(INDEX_DIR / "bm25_doc_ids.pkl", "rb"))
    return bm25, ids


def load_queries():
    queries = []
    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            qid, q = line.strip().split("\t")
            queries.append((qid, q))
    return queries


def run(k1, b):  
    bm25, ids = load(k1, b)
    queries = load_queries()

    out_file = RUNS_DIR / f"run_bm25_{k1}_{b}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for qid, q in queries:
            tokens = preprocess_text(q).split()
            scores = bm25.get_scores(tokens)

            idx = np.argsort(scores)[::-1][:10]

            for rank, i in enumerate(idx, 1):
                f.write(f"{qid} Q0 {ids[i]} {rank} {scores[i]:.4f} BM25\n")

    print(f"✅ BM25 run → {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/search_bm25.py <k1> <b>")
        exit()

    k1 = float(sys.argv[1])
    b = float(sys.argv[2])

    run(k1, b)