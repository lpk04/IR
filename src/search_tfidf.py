import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import INDEX_DIR, RUNS_DIR, QUERY_TEXT_FILE
from prepare import preprocess_text

RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load():
    v = pickle.load(open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb"))
    X = pickle.load(open(INDEX_DIR / "tfidf_matrix.pkl", "rb"))
    ids = pickle.load(open(INDEX_DIR / "tfidf_doc_ids.pkl", "rb"))
    return v, X, ids


def load_queries():
    queries = []
    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            qid, q = line.strip().split("\t")
            queries.append((qid, q))
    return queries


def run():
    v, X, ids = load()
    queries = load_queries()

    out_file = RUNS_DIR / "run_tfidf.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for qid, q in queries:
            q_clean = preprocess_text(q)
            q_vec = v.transform([q_clean])

            scores = cosine_similarity(q_vec, X).flatten()
            idx = np.argsort(scores)[::-1][:10]

            for rank, i in enumerate(idx, 1):
                f.write(f"{qid} Q0 {ids[i]} {rank} {scores[i]:.4f} TFIDF\n")

    print(f"✅ TF-IDF run → {out_file}")


if __name__ == "__main__":
    run()