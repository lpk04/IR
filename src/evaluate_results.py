import os
from collections import defaultdict

from config import RUNS_DIR, RESULTS_DIR, QRELS_FILE

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD QRELS
# =========================
def load_qrels():
    qrels = defaultdict(set)

    with open(QRELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            qid, doc_id = line.strip().split()
            qrels[qid].add(doc_id)

    return qrels


# =========================
# LOAD RUN FILE
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
    return rel / k


def recall_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for doc in retrieved_k if doc in relevant)
    return rel / len(relevant) if relevant else 0


def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


# =========================
# EVALUATE ONE RUN
# =========================
def evaluate_run(run_file, qrels, k=10):
    run = load_run(run_file)

    precisions = []
    recalls = []
    f1s = []

    for qid in run:
        retrieved = run[qid]
        relevant = qrels.get(qid, set())

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        f = f1(p, r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    return avg_p, avg_r, avg_f1


# =========================
# MAIN
# =========================
def evaluate_all():
    qrels = load_qrels()

    result_file = RESULTS_DIR / "evaluation.txt"

    with open(result_file, "w", encoding="utf-8") as f:

        f.write("===== EVALUATION RESULTS =====\n\n")

        for file in os.listdir(RUNS_DIR):
            if file.endswith(".txt"):
                run_path = RUNS_DIR / file

                p, r, f1_score = evaluate_run(run_path, qrels)

                line = (
                    f"{file}:\n"
                    f"  Precision@10: {p:.4f}\n"
                    f"  Recall@10:    {r:.4f}\n"
                    f"  F1@10:        {f1_score:.4f}\n\n"
                )

                print(line)
                f.write(line)

    print(f"✅ Evaluation saved → {result_file}")


if __name__ == "__main__":
    evaluate_all()