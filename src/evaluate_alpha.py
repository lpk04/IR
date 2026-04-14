import math
from collections import defaultdict
from pathlib import Path

from config import RESULTS_DIR, RUNS_SEARCH_BM25_DIR


# =========================
# CONFIG: FILES CẦN SO SÁNH
# =========================
RUN_FILES = {
    "alpha_0.0 (BM25)": RUNS_SEARCH_BM25_DIR / "bm25_1.2_0.75.txt",
    "alpha_0.2": RUNS_SEARCH_BM25_DIR / "bm25_sentiment_ml_1.2_0.75_a0.2.txt",
    "alpha_0.5": RUNS_SEARCH_BM25_DIR / "bm25_sentiment_ml_1.2_0.75_a0.5.txt",
    "alpha_1.0": RUNS_SEARCH_BM25_DIR / "bm25_sentiment_ml_1.2_0.75_a1.0.txt",
}

QRELS_FILES = {
    "DEFAULT": RESULTS_DIR / "qrels.txt",
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT":   RESULTS_DIR / "qrels_count.txt",
    "RATIO":   RESULTS_DIR / "qrels_ratio.txt",
}
OUTPUT_FILE = RESULTS_DIR / "evaluation_sentiment_alpha.txt"


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
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def recall_at_k(retrieved, relevant, k=10):
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant) if relevant else 0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0


def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(retrieved, qrels, k=10):
    rels = [qrels.get(d, 0) for d in retrieved[:k]]
    dcg_val = dcg(rels)

    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)

    return dcg_val / idcg if idcg > 0 else 0


# =========================
# EVALUATE ONE RUN
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qrels_sets = {name: load_qrels(path) for name, path in QRELS_FILES.items()}

    results = []

    for run_name, path in RUN_FILES.items():
        run = load_run(path)

        for qrels_name, qrels in qrels_sets.items():
            p, r, f1_score, ndcg = evaluate(run, qrels)
            results.append((run_name, qrels_name, p, r, f1_score, ndcg))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("===== SENTIMENT ALPHA COMPARISON =====\n\n")

        for run_name, qrels_name, p, r, f1_score, ndcg in results:
            f.write(f"{run_name} | QRELS: {qrels_name}\n")
            f.write(f"  P@10: {p:.4f}\n")
            f.write(f"  R@10: {r:.4f}\n")
            f.write(f"  F1:   {f1_score:.4f}\n")
            f.write(f"  NDCG: {ndcg:.4f}\n\n")

    print(f"✅ Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
