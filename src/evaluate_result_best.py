import os
import math
from collections import defaultdict
from pathlib import Path

from config import (
    RESULTS_DIR,
    RUNS_SEARCH_BM25_DIR,
    RUNS_DIR
)

# =========================
# CONFIG
# =========================
BEST_BM25_FILE = RUNS_SEARCH_BM25_DIR / "bm25_1.2_0.75.txt"
RRF_FILE       = RUNS_DIR / "runs_rrf" / "rrf_tfidf_bm25.txt"

QRELS_FILES = {
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT":   RESULTS_DIR / "qrels_count.txt",
    "RATIO":   RESULTS_DIR / "qrels_ratio.txt",
}


# =========================
# LOAD QRELS
# =========================
def load_qrels(path):
    qrels = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split()
            qrels[qid][doc_id] = int(rel)

    return qrels


# =========================
# LOAD RUN
# =========================
def load_run(path):
    run = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
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
    rel = sum(1 for d in retrieved[:k] if d in relevant)
    return rel / k


def recall_at_k(retrieved, relevant, k=10):
    rel = sum(1 for d in retrieved[:k] if d in relevant)
    return rel / len(relevant) if relevant else 0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0


def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(retrieved, qrels, k=10):
    rels = [qrels.get(doc, 0) for doc in retrieved[:k]]
    dcg_val = dcg(rels)

    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)

    return dcg_val / idcg if idcg > 0 else 0


# =========================
# EVALUATE
# =========================
def evaluate(run, qrels_all):
    P, R, F, N = [], [], [], []

    for qid in run:
        retrieved = run[qid]
        qrels = qrels_all.get(qid, {})

        relevant = {d for d, rel in qrels.items() if rel > 0}

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)
        f = f1(p, r)
        n = ndcg_at_k(retrieved, qrels)

        P.append(p)
        R.append(r)
        F.append(f)
        N.append(n)

    return (
        sum(P)/len(P),
        sum(R)/len(R),
        sum(F)/len(F),
        sum(N)/len(N)
    )


# =========================
# MAIN
# =========================
def main():
    result_file = RESULTS_DIR / "evaluation_best.txt"

    run_bm25 = load_run(BEST_BM25_FILE)
    run_rrf  = load_run(RRF_FILE)

    with open(result_file, "w", encoding="utf-8") as f:

        f.write("===== FINAL COMPARISON =====\n\n")

        for name, qrels_path in QRELS_FILES.items():
            if not qrels_path.exists():
                continue

            f.write(f"===== QRELS: {name} =====\n\n")

            qrels = load_qrels(qrels_path)

            p_b, r_b, f_b, n_b = evaluate(run_bm25, qrels)
            p_r, r_r, f_r, n_r = evaluate(run_rrf, qrels)

            f.write("---- BM25 BEST ----\n")
            f.write(f"P@10: {p_b:.4f} | R@10: {r_b:.4f} | F1: {f_b:.4f} | NDCG: {n_b:.4f}\n\n")

            f.write("---- RRF (TF-IDF + BM25) ----\n")
            f.write(f"P@10: {p_r:.4f} | R@10: {r_r:.4f} | F1: {f_r:.4f} | NDCG: {n_r:.4f}\n\n")

            # WINNER
            if n_r > n_b:
                f.write("👉 RRF WIN\n\n")
            else:
                f.write("👉 BM25 WIN\n\n")

            f.write("="*60 + "\n\n")

    print(f"✅ Done → {result_file}")


if __name__ == "__main__":
    main()