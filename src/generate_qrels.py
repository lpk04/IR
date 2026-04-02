import json

from config import (
    REVIEW_PROCESSED_FILE,
    QRELS_FILE,
    QUERY_TEXT_FILE,
    RESULTS_DIR,
    LOG_FILE
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
# GENERATE QRELS
# =========================
def generate_qrels():
    queries = load_queries()
    docs = load_docs()

    logs = []

    logs.append(f"Total queries: {len(queries)}\n")
    logs.append(f"Total docs: {len(docs)}\n\n")

    qrels = []

    for qid, query in queries:
        q_tokens = set(preprocess_text(query).split())

        rel_count = 0

        for doc_id, text in docs:
            doc_tokens = set(text.split())

            common = len(q_tokens & doc_tokens)

            # =========================
            # SIMPLE RULE
            # =========================
            if common >= 1:
                rel = 1
                rel_count += 1
            else:
                rel = 0

            qrels.append((qid, doc_id, rel))

        logs.append(f"{qid}: {rel_count} relevant docs\n")

    return qrels, logs


# =========================
# SAVE FILES
# =========================
def save_qrels(qrels, logs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # save qrels (TREC format)
    with open(QRELS_FILE, "w", encoding="utf-8") as f:
        for qid, doc_id, rel in qrels:
            f.write(f"{qid} 0 {doc_id} {rel}\n")

    # save log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(logs)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    qrels, logs = generate_qrels()
    save_qrels(qrels, logs)