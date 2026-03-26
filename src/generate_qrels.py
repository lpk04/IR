import json
from config import REVIEW_PROCESSED_FILE, QRELS_FILE, QUERY_TEXT_FILE, RESULTS_DIR
from prepare import preprocess_text

LOG_FILE = RESULTS_DIR / "qrels_log.txt"


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
# GENERATE QRELS (FIXED)
# =========================
def generate_qrels():
    queries = load_queries()
    docs = load_docs()

    qrels = {}
    logs = []

    logs.append(f"Total queries: {len(queries)}\n")
    logs.append(f"Total docs: {len(docs)}\n\n")

    for qid, query in queries:
        q_tokens = set(preprocess_text(query).split())
        qrels[qid] = []

        for doc_id, text in docs:
            doc_tokens = set(text.split())

            # 🔥 FIX: dùng số token trùng thay vì overlap %
            common = len(q_tokens & doc_tokens)

            # 🔥 điều kiện chặt hơn
            if common >= max(3, int(0.7 * len(q_tokens))):
                qrels[qid].append(doc_id)

        logs.append(f"{qid}: {len(qrels[qid])} relevant docs\n")

    return qrels, logs


# =========================
# SAVE FILES
# =========================
def save_qrels(qrels, logs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # save qrels
    with open(QRELS_FILE, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for doc_id in docs:
                f.write(f"{qid}\t{doc_id}\n")

    # save log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(logs)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    qrels, logs = generate_qrels()
    save_qrels(qrels, logs)