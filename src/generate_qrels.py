import json

from config import (
    REVIEW_PROCESSED_FILE,
    QUERY_TEXT_FILE,
    RESULTS_DIR,
    LOG_FILE,
    QRELS_KEYWORD,
    QRELS_COUNT,
    QRELS_RATIO
)

from prepare_data import preprocess_text


# =========================
# LOAD QUERIES
# =========================
def load_queries():
    queries = []

    with open(QUERY_TEXT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split("\t")

            if len(parts) == 2:
                qid, q = parts
            else:
                qid = f"Q{i}"
                q = line.strip()

            if q:
                queries.append((qid, q))

    return queries


# =========================
# LOAD DOCS
# =========================
def load_docs():
    docs = []

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append((d["doc_id"], d["text"]))

    return docs


# =========================
# SAVE
# =========================
def save_qrels(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for qid, doc_id, rel in data:
            f.write(f"{qid} 0 {doc_id} {rel}\n")


# =========================
# MAIN GENERATE
# =========================
def generate_qrels():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_queries()
    docs = load_docs()

    keyword_qrels = []
    count_qrels   = []
    ratio_qrels   = []

    logs = []

    for qid, query in queries:
        q_tokens = set(preprocess_text(query).split())

        keyword_rel = 0
        count_rel   = 0
        ratio_rel   = 0

        for doc_id, text in docs:
            doc_tokens = set(text.split())

            common = len(q_tokens & doc_tokens)
            ratio  = common / len(q_tokens) if q_tokens else 0

            # =========================
            # 1. KEYWORD (>=1)
            # =========================
            rel1 = 1 if common >= 1 else 0
            keyword_qrels.append((qid, doc_id, rel1))
            keyword_rel += rel1

            # =========================
            # 2. COUNT
            # =========================
            rel2 = 1 if common >= max(2, int(0.5 * len(q_tokens))) else 0
            count_qrels.append((qid, doc_id, rel2))
            count_rel += rel2

            # =========================
            # 3. RATIO
            # =========================
            rel3 = 1 if ratio >= 0.7 else 0
            ratio_qrels.append((qid, doc_id, rel3))
            ratio_rel += rel3

        logs.append(
            f"{qid}: keyword={keyword_rel}, count={count_rel}, ratio={ratio_rel}\n"
        )

    # =========================
    # SAVE FILES
    # =========================
    save_qrels(QRELS_KEYWORD, keyword_qrels)
    save_qrels(QRELS_COUNT, count_qrels)
    save_qrels(QRELS_RATIO, ratio_qrels)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(logs)

    print("✅ Generated 3 types of qrels")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    generate_qrels()