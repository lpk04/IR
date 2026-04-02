import pickle
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    QUERY_TEXT_FILE,
    RUNS_SEARCH_TFIDF_DIR,
    TFIDF_SEARCH_TRACE_DIR,
    get_tfidf_paths,
    get_tfidf_run_paths
)

from prepare_data import preprocess_text


# =========================
# PARSE ARGUMENT
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ngram", type=str, required=True)
    parser.add_argument("--sublinear", type=str, required=True)

    return parser.parse_args()


# =========================
# ENSURE DIR
# =========================
def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
    else:
        # xóa file nếu đã tồn tại
        for f in path.glob("*"):
            if f.is_file():
                f.unlink()


# =========================
# LOAD INDEX
# =========================
def load_index(paths):
    with open(paths["vectorizer"], "rb") as f:
        v = pickle.load(f)

    with open(paths["matrix"], "rb") as f:
        X = pickle.load(f)

    with open(paths["ids"], "rb") as f:
        ids = pickle.load(f)

    return v, X, ids


# =========================
# LOAD QUERIES
# =========================
def load_queries():
    queries = []
    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            qid, q = line.strip().split("\t")
            queries.append((qid, q))
    return queries


# =========================
# RUN SEARCH
# =========================
def run_search(ngram, sublinear, top_k=10):
    index_paths = get_tfidf_paths(ngram, sublinear)
    run_paths = get_tfidf_run_paths(ngram, sublinear)

    # tạo folder nếu chưa có
    ensure_dir(RUNS_SEARCH_TFIDF_DIR)
    ensure_dir(TFIDF_SEARCH_TRACE_DIR)

    v, X, ids = load_index(index_paths)
    queries = load_queries()

    with open(run_paths["run"], "w", encoding="utf-8") as run_f, \
         open(run_paths["trace"], "w", encoding="utf-8") as trace_f:

        trace_f.write(f"===== SEARCH TF-IDF {ngram} sub={sublinear} =====\n\n")

        for qid, q in queries:
            q_clean = preprocess_text(q)
            q_vec = v.transform([q_clean])

            scores = cosine_similarity(q_vec, X).flatten()
            idx = np.argsort(scores)[::-1][:top_k]

            # WRITE RUN
            for rank, i in enumerate(idx, 1):
                run_f.write(
                    f"{qid} Q0 {ids[i]} {rank} {scores[i]:.4f} TFIDF\n"
                )

            # TRACE
            trace_f.write(f"Query: {q}\n")
            trace_f.write(f"Clean: {q_clean}\n")

            for rank, i in enumerate(idx[:5], 1):
                trace_f.write(
                    f"  Rank {rank}: Doc {ids[i]} | Score: {scores[i]:.4f}\n"
                )

            trace_f.write("\n" + "-" * 50 + "\n\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    ngram = (int(args.ngram[0]), int(args.ngram[1]))
    sublinear = args.sublinear.lower() == "true"

    run_search(ngram, sublinear)