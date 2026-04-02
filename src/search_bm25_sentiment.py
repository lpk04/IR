import json
import pickle
import numpy as np
import argparse

from config import (
    REVIEW_PROCESSED_FILE,
    RUNS_SEARCH_BM25_DIR,
    BM25_SEARCH_TRACE_DIR,
    get_bm25_paths,
    get_bm25_run_paths
)

from prepare_data import preprocess_text


# =========================
# SENTIMENT SCORE
# =========================
def sentiment_score(s):
    if s == "positive":
        return 1
    elif s == "neutral":
        return 0
    else:
        return -1


# =========================
# LOAD SENTIMENT MAP
# =========================
def load_sentiment():
    sentiment_map = {}

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sentiment_map[d["doc_id"]] = d["sentiment"]

    return sentiment_map


# =========================
# PARSE ARG
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k1", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)

    return parser.parse_args()


# =========================
# LOAD BM25
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
def load_queries(query_file):
    queries = []

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            qid, q = line.strip().split("\t")
            queries.append((qid, q))

    return queries


# =========================
# RUN SEARCH + RERANK
# =========================
def run_search(k1, b, alpha):
    index_paths = get_bm25_paths(k1, b)

    name = f"bm25_sentiment_{k1}_{b}_a{alpha}"
    run_file = RUNS_SEARCH_BM25_DIR / f"{name}.txt"
    trace_file = BM25_SEARCH_TRACE_DIR / f"{name}.txt"

    RUNS_SEARCH_BM25_DIR.mkdir(parents=True, exist_ok=True)
    BM25_SEARCH_TRACE_DIR.mkdir(parents=True, exist_ok=True)

    bm25, doc_ids = load_index(index_paths)
    queries = load_queries("queries.txt")
    sentiment_map = load_sentiment()

    with open(run_file, "w", encoding="utf-8") as run_f, \
         open(trace_file, "w", encoding="utf-8") as trace_f:

        trace_f.write(f"===== BM25 + Sentiment (alpha={alpha}) =====\n\n")

        for qid, q in queries:
            q_clean = preprocess_text(q)
            tokens = q_clean.split()

            bm25_scores = bm25.get_scores(tokens)

            final_scores = []

            for i, doc_id in enumerate(doc_ids):
                s = sentiment_map.get(doc_id, "neutral")
                s_score = sentiment_score(s)

                final = bm25_scores[i] + alpha * s_score
                final_scores.append(final)

            final_scores = np.array(final_scores)
            idx = np.argsort(final_scores)[::-1][:10]

            # WRITE RUN
            for rank, i in enumerate(idx, 1):
                run_f.write(
                    f"{qid} Q0 {doc_ids[i]} {rank} {final_scores[i]:.4f} BM25_SENT\n"
                )

            # TRACE
            trace_f.write(f"Query: {q}\n")

            for rank, i in enumerate(idx[:5], 1):
                trace_f.write(
                    f"  Rank {rank}: Doc {doc_ids[i]} | Score: {final_scores[i]:.4f} | Sent: {sentiment_map[doc_ids[i]]}\n"
                )

            trace_f.write("\n" + "-" * 50 + "\n\n")

    print(f"✅ BM25 + Sentiment → {run_file}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()
    run_search(args.k1, args.b, args.alpha)