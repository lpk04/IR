import os
import math
from collections import defaultdict

from config import (
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_BM25_DIR,
    RESULTS_DIR,
    RUNS_RRF_DIR, # cái này khi nào kết hợp 3 cái mới dùng
)

# =========================
# QRELS FILES
# =========================
QRELS_FILES = {
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT":   RESULTS_DIR / "qrels_count.txt",
    "RATIO":   RESULTS_DIR / "qrels_ratio.txt",
}

RESULT_FILE = RESULTS_DIR / "evaluation_ndcg.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD QRELS
# =========================
def load_qrels(file_path):
    qrels = defaultdict(set)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split()
            if int(rel) == 1:
                qrels[qid].add(doc_id)

    return qrels


# =========================
# LOAD RUN
# =========================
def load_run(file_path):
    run = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            doc_id = parts[2]
            run[qid].append(doc_id)

    return run


# =========================
# METRICS
# =========================
def precision_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for doc in retrieved_k if doc in relevant)
    return rel / k if k > 0 else 0


def recall_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for doc in retrieved_k if doc in relevant)
    return rel / len(relevant) if relevant else 0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


# =========================
# NDCG
# =========================
def dcg_at_k(retrieved, relevant, k=10):
    dcg = 0.0

    for i, doc in enumerate(retrieved[:k], start=1):
        rel = 1 if doc in relevant else 0
        dcg += rel / math.log2(i + 1)

    return dcg


def ndcg_at_k(retrieved, relevant, k=10):
    dcg = dcg_at_k(retrieved, relevant, k)

    ideal_rels = [1] * min(len(relevant), k)
    idcg = 0.0

    for i, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0


# =========================
# EVALUATE ONE RUN
# =========================
def evaluate_run(run_file, qrels, k=10):
    run = load_run(run_file)

    precisions, recalls, f1s, ndcgs = [], [], [], []

    for qid in run:
        retrieved = run[qid]
        relevant = qrels.get(qid, set())

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        f = f1(p, r)
        ndcg = ndcg_at_k(retrieved, relevant, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        ndcgs.append(ndcg)

    if not precisions:
        return 0, 0, 0, 0

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

    for file in os.listdir(folder):
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

            # thêm của ffr khi chạy khi k sài thì note lại
            f.write("---- RRF ----\n\n")
            evaluate_folder(RUNS_RRF_DIR, qrels, f)


            f.write("\n" + "=" * 60 + "\n\n")

    print(f"✅ Saved → {RESULT_FILE}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate_all()