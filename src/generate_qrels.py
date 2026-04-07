import json
import random
from collections import defaultdict
from pathlib import Path

from config import (
    REVIEW_PROCESSED_FILE,
    QUERY_TEXT_FILE,
    LOG_FILE,
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_BM25_DIR
)

from prepare_data import preprocess_text

# =========================
# PATH CONFIG
# =========================
RESULTS_DIR = Path(r"D:\IR\demo\results")

QRELS_KEYWORD = RESULTS_DIR / "qrels_keyword.txt"
QRELS_COUNT   = RESULTS_DIR / "qrels_count.txt"
QRELS_RATIO   = RESULTS_DIR / "qrels_ratio.txt"

BM25_FILE  = RUNS_SEARCH_BM25_DIR / "bm25_1.2_0.75.txt"   # dùng config fixed
TFIDF_FILE = RUNS_SEARCH_TFIDF_DIR / "tfidf_1_2_sub.txt"

RANDOM_K = 20
TOP_K = 50

random.seed(42)


# =========================
# LOAD QUERIES
# =========================
def load_queries():
    queries = []

    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split("\t")

            if len(parts) >= 2:
                qid, q = parts[0], parts[1]
            else:
                qid, q = f"Q{i}", line.strip()

            if q:
                queries.append((qid, q))

    return queries


# =========================
# LOAD RUN FILE (TOP K)
# =========================
def load_top_k(file_path, k):
    top_docs = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 3:
                continue

            qid = parts[0]
            doc_id = parts[2]

            if len(top_docs[qid]) < k:
                top_docs[qid].append(doc_id)

    return top_docs


# =========================
# LOAD ALL DOC IDS
# =========================
def load_all_doc_ids():
    doc_ids = []

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_ids.append(d["doc_id"])

    return doc_ids


# =========================
# LOAD ONLY NEEDED DOCS
# =========================
def load_docs(needed_doc_ids):
    tokens_cache = {}

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_id = d["doc_id"]

            if doc_id not in needed_doc_ids:
                continue

            tokens_cache[doc_id] = set(
                preprocess_text(d["text"]).split()
            )

    return tokens_cache


# =========================
# MIX TOP DOCS
# =========================
def build_candidate_set(bm25_docs, tfidf_docs, all_doc_ids):
    top_docs = defaultdict(set)

    for qid in bm25_docs:
        top_docs[qid].update(bm25_docs[qid])

    for qid in tfidf_docs:
        top_docs[qid].update(tfidf_docs[qid])

    # add random
    for qid in top_docs:
        current = top_docs[qid]
        candidates = list(set(all_doc_ids) - current)

        rand_docs = random.sample(candidates, min(RANDOM_K, len(candidates)))
        top_docs[qid].update(rand_docs)

    return top_docs


# =========================
# SAVE
# =========================
def save_qrels(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_id, rel in data:
            f.write(f"{qid} 0 {doc_id} {rel}\n")


# =========================
# MAIN
# =========================
def generate_qrels():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🔄 Loading queries...")
    queries = load_queries()

    print("🔄 Loading BM25 + TF-IDF top docs...")
    bm25_docs  = load_top_k(BM25_FILE, TOP_K)
    tfidf_docs = load_top_k(TFIDF_FILE, TOP_K)

    print("🔄 Loading all doc ids...")
    all_doc_ids = load_all_doc_ids()

    print("🔀 Building candidate set (BM25 + TF-IDF + random)...")
    top_docs = build_candidate_set(bm25_docs, tfidf_docs, all_doc_ids)

    needed_doc_ids = set()
    for qid in top_docs:
        needed_doc_ids.update(top_docs[qid])

    print(f"📦 Need {len(needed_doc_ids)} docs")

    print("🔄 Loading needed docs...")
    tokens_cache = load_docs(needed_doc_ids)

    keyword_qrels = []
    count_qrels   = []
    ratio_qrels   = []

    logs = []

    print("⚙️ Generating qrels...")

    for qid, query in queries:
        q_tokens = set(preprocess_text(query).split())

        keyword_rel = 0
        count_rel   = 0
        ratio_rel   = 0

        for doc_id in top_docs.get(qid, []):
            doc_tokens = tokens_cache.get(doc_id, set())

            common = len(q_tokens & doc_tokens)
            ratio = common / len(q_tokens) if q_tokens else 0

            # KEYWORD
            rel1 = 1 if common >= 1 else 0
            keyword_qrels.append((qid, doc_id, rel1))
            keyword_rel += rel1

            # COUNT
            k = min(2, len(q_tokens))
            rel2 = 1 if common >= k else 0
            count_qrels.append((qid, doc_id, rel2))
            count_rel += rel2

            # RATIO
            rel3 = 1 if ratio >= 0.5 else 0
            ratio_qrels.append((qid, doc_id, rel3))
            ratio_rel += rel3

        logs.append(
            f"{qid}: keyword={keyword_rel}, count={count_rel}, ratio={ratio_rel}\n"
        )

    print("💾 Saving qrels...")

    save_qrels(QRELS_KEYWORD, keyword_qrels)
    save_qrels(QRELS_COUNT, count_qrels)
    save_qrels(QRELS_RATIO, ratio_qrels)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(logs)

    print("✅ DONE!")
    print(f"📁 Saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_qrels()