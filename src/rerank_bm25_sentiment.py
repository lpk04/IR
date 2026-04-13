import json
import pickle
import numpy as np
import argparse
import joblib

from config import (
    REVIEW_PROCESSED_FILE,
    QUERY_TEXT_FILE,
    RUNS_SEARCH_BM25_DIR,
    BM25_SEARCH_TRACE_DIR,
    SENTIMENT_MODEL_PATH,
    VECTORIZER_PATH,
    get_bm25_paths
)

from prepare_data import preprocess_text


# =========================
# LOAD SENTIMENT MODEL (FIX PATH)
# =========================
def load_sentiment_model():

    model = joblib.load(SENTIMENT_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# =========================
# PREDICT QUERY SENTIMENT
# =========================
def predict_query_sentiment(query, model, vectorizer):
    q_clean = preprocess_text(query)
    q_vec = vectorizer.transform([q_clean])

    pred = model.predict(q_vec)[0]

    # normalize label
    if isinstance(pred, str):
        if pred.lower() in ["pos", "positive"]:
            label = "positive"
        else:
            label = "negative"
    else:
        label = "positive" if pred == 1 else "negative"

    # confidence check (optional nhưng nên có)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(q_vec)[0]
        if max(proba) < 0.6:
            return "neutral"

    return label


# =========================
# LOAD DOC SENTIMENT
# =========================
def load_doc_sentiment():
    sentiment_map = {}

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sentiment_map[d["doc_id"]] = d["sentiment"]

    return sentiment_map


# =========================
# LOAD BM25 INDEX
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
# RERANK FUNCTION
# =========================
def rerank(k1, b, alpha):
    index_paths = get_bm25_paths(k1, b)

    name = f"bm25_sentiment_ml_{k1}_{b}_a{alpha}"
    run_file = RUNS_SEARCH_BM25_DIR / f"{name}.txt"
    trace_file = BM25_SEARCH_TRACE_DIR / f"{name}.txt"

    RUNS_SEARCH_BM25_DIR.mkdir(parents=True, exist_ok=True)
    BM25_SEARCH_TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Loading BM25...")
    bm25, doc_ids = load_index(index_paths)

    print("📥 Loading sentiment model...")
    model, vectorizer = load_sentiment_model()

    print("📥 Loading doc sentiment...")
    sentiment_map = load_doc_sentiment()

    queries = load_queries()

    with open(run_file, "w", encoding="utf-8") as run_f, \
         open(trace_file, "w", encoding="utf-8") as trace_f:

        trace_f.write(f"===== BM25 + ML Sentiment (alpha={alpha}) =====\n\n")

        for qid, q in queries:
            q_clean = preprocess_text(q)
            tokens = q_clean.split()

            bm25_scores = bm25.get_scores(tokens)

            # 🔥 predict query sentiment
            q_sent = predict_query_sentiment(q, model, vectorizer)

            final_scores = []

            for i, doc_id in enumerate(doc_ids):
                doc_sent = sentiment_map.get(doc_id, "neutral")

                # =========================
                # SENTIMENT MATCH LOGIC
                # =========================
                if q_sent == "neutral":
                    s_boost = 0
                elif doc_sent == q_sent:
                    s_boost = 1
                else:
                    s_boost = -1

                final = bm25_scores[i] + alpha * s_boost
                final_scores.append(final)

            final_scores = np.array(final_scores)

            # 🔥 top 100 chuẩn IR
            idx = np.argsort(final_scores)[::-1][:100]

            # WRITE RUN
            for rank, i in enumerate(idx, 1):
                run_f.write(
                    f"{qid} Q0 {doc_ids[i]} {rank} {final_scores[i]:.4f} BM25_SENT_ML\n"
                )

            # TRACE
            trace_f.write(f"Query: {q} | QSent: {q_sent}\n")

            for rank, i in enumerate(idx[:10], 1):
                trace_f.write(
                    f"  Rank {rank}: Doc {doc_ids[i]} | Score: {final_scores[i]:.4f} | DocSent: {sentiment_map.get(doc_ids[i],'neutral')}\n"
                )

            trace_f.write("\n" + "-" * 50 + "\n\n")

    print(f"✅ Saved → {run_file}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--k1", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)

    args = parser.parse_args()

    rerank(args.k1, args.b, args.alpha)