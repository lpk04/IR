import os
import math
from collections import defaultdict

from config import (
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_BM25_DIR,
    RESULTS_DIR,
    RUNS_RRF_DIR,  # cái này khi nào kết hợp 3 cái mới dùng
)

# =========================
# QRELS FILES
# =========================
QRELS_FILES = {
    "DEFAULT": RESULTS_DIR / "qrels.txt",
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT":   RESULTS_DIR / "qrels_count.txt",
    "RATIO":   RESULTS_DIR / "qrels_ratio.txt",
}

RESULT_FILE = RESULTS_DIR / "evaluation_sentiment.txt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD QRELS (graded relevance)
# qrels[qid][doc_id] = rel_grade
# =========================
def load_qrels(file_path):
    qrels = defaultdict(dict)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            qid, _, doc_id, rel = parts[:4]

            try:
                rel = int(rel)
            except ValueError:
                continue

            qrels[qid][doc_id] = rel

    return qrels


# =========================
# LOAD RUN
# =========================
def load_run(file_path):
    run = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            qid = parts[0]
            doc_id = parts[2]
            run[qid].append(doc_id)

    return run


# =========================
# METRICS
# =========================
def precision_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for doc in retrieved_k if relevant.get(doc, 0) > 0)
    return rel / k if k > 0 else 0.0


def recall_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    total_relevant = sum(1 for _, grade in relevant.items() if grade > 0)
    if total_relevant == 0:
        return 0.0

    rel = sum(1 for doc in retrieved_k if relevant.get(doc, 0) > 0)
    return rel / total_relevant


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# =========================
# NDCG (graded relevance)
# DCG = sum((2^rel - 1) / log2(i + 1))
# =========================
def dcg_at_k(retrieved, relevant, k=10):
    dcg = 0.0

    for i, doc in enumerate(retrieved[:k], start=1):
        rel = relevant.get(doc, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)

    return dcg


def ndcg_at_k(retrieved, relevant, k=10):
    dcg = dcg_at_k(retrieved, relevant, k)

    ideal_rels = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)[:k]
    idcg = 0.0

    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


# =========================
# EVALUATE ONE RUN
# =========================
def evaluate_run(run_file, qrels, k=10):
    run = load_run(run_file)

    precisions, recalls, f1s, ndcgs = [], [], [], []

    # Evaluate over qrels queries to avoid silently skipping judged queries
    for qid, relevant in qrels.items():
        retrieved = run.get(qid, [])

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        f = f1(p, r)
        ndcg = ndcg_at_k(retrieved, relevant, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        ndcgs.append(ndcg)

    if not precisions:
        return 0.0, 0.0, 0.0, 0.0

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
        sum(ndcgs) / len(ndcgs),
    )


# =========================
# EVALUATE FOLDER
# =========================
def evaluate_folder(folder, qrels, f):
    if not folder.exists():
        return

    for file in sorted(os.listdir(folder)):
        if file.endswith(".txt"):
            run_path = folder / file

            p, r, f1_score, ndcg = evaluate_run(run_path, qrels)

            f.write(
                f"{file}:\n"
                f"  Precision@10: {p:.4f}\n"
                f"  Recall@10:    {r:.4f}\n"
                f"  F1@10:        {f1_score:.4f}\n"
                f"  NDCG@10:      {ndcg:.4f}\n\n"
            )


# =========================
# MAIN
# =========================
def evaluate_all():
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("===== EVALUATION WITH NDCG =====\n\n")

        for name, qrels_path in QRELS_FILES.items():
            if not qrels_path.exists():
                continue

            f.write(f"===== QRELS: {name} =====\n\n")

            qrels = load_qrels(qrels_path)

            f.write("---- TF-IDF ----\n\n")
            evaluate_folder(RUNS_SEARCH_TFIDF_DIR, qrels, f)

            f.write("---- BM25 ----\n\n")
            evaluate_folder(RUNS_SEARCH_BM25_DIR, qrels, f)

            # thêm của rrf khi chạy khi k sài thì note lại
            f.write("---- RRF ----\n\n")
            evaluate_folder(RUNS_RRF_DIR, qrels, f)

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"✅ Saved → {RESULT_FILE}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate_all()