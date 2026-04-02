import pickle
import numpy as np
import argparse

from config import (
    QUERY_TEXT_FILE,
    RUNS_SEARCH_BM25_DIR,
    BM25_SEARCH_TRACE_DIR,
    get_bm25_paths,
    get_bm25_run_paths
)

from prepare_data import preprocess_text


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
    else:
        print(f"⚠️ Exists: {path}")


# =========================
# LOAD INDEX
# =========================
def load_index(paths):
    with open(paths["model"], "rb") as f:
        bm25 = pickle.load(f)

    with open(paths["ids"], "rb") as f:
        doc_ids = pickle.load(f)

    return bm25, doc_ids


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
def run_search(k1, b, top_k=10):
    index_paths = get_bm25_paths(k1, b)
    run_paths = get_bm25_run_paths(k1, b)

    ensure_dir(RUNS_SEARCH_BM25_DIR)
    ensure_dir(BM25_SEARCH_TRACE_DIR)

    bm25, doc_ids = load_index(index_paths)
    queries = load_queries()

    with open(run_paths["run"], "w", encoding="utf-8") as run_f, \
         open(run_paths["trace"], "w", encoding="utf-8") as trace_f:

        trace_f.write(f"===== SEARCH BM25 (k1={k1}, b={b}) =====\n\n")

        for qid, q in queries:
            q_clean = preprocess_text(q)
            tokens = q_clean.split()

            scores = bm25.get_scores(tokens)
            idx = np.argsort(scores)[::-1][:top_k]

            # =========================
            # WRITE RUN
            # =========================
            for rank, i in enumerate(idx, 1):
                run_f.write(
                    f"{qid} Q0 {doc_ids[i]} {rank} {scores[i]:.4f} BM25\n"
                )

            # =========================
            # TRACE
            # =========================
            trace_f.write(f"Query: {q}\n")
            trace_f.write(f"Clean: {q_clean}\n")
            trace_f.write(f"Tokens: {tokens}\n")

            for rank, i in enumerate(idx[:5], 1):
                trace_f.write(
                    f"  Rank {rank}: Doc {doc_ids[i]} | Score: {scores[i]:.4f}\n"
                )

            trace_f.write("\n" + "-" * 50 + "\n\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()
    run_search(args.k1, args.b)