import os
import math
from collections import defaultdict

from config import RESULTS_DIR, RUNS_RRF_DIR

# =========================
# QRELS
# =========================
QRELS_FILES = {
    "DEFAULT": RESULTS_DIR / "qrels.txt",
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT":   RESULTS_DIR / "qrels_count.txt",
    "RATIO":   RESULTS_DIR / "qrels_ratio.txt",
}


RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD QRELS (graded)
# =========================
def load_qrels(file_path):
    qrels = defaultdict(dict)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split()
            qrels[qid][doc_id] = int(rel)

    return qrels


# =========================
# LOAD RUN
# =========================
def load_run(file_path):
    run = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
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
def precision_at_k(retrieved, relevant_set, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for d in retrieved_k if d in relevant_set)
    return rel / k if k else 0


def recall_at_k(retrieved, relevant_set, k=10):
    retrieved_k = retrieved[:k]
    rel = sum(1 for d in retrieved_k if d in relevant_set)
    return rel / len(relevant_set) if relevant_set else 0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0


# =========================
# DCG / NDCG
# =========================
def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(retrieved, qrels, k=10):
    rels = [qrels.get(doc, 0) for doc in retrieved[:k]]
    dcg_val = dcg(rels)

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal_rels)

    return dcg_val / idcg if idcg > 0 else 0


# =========================
# EVALUATE ONE RUN
# =========================
def evaluate_run(run_file, qrels_all, k=10):
    run = load_run(run_file)

    p_scores, r_scores, f_scores, n_scores = [], [], [], []

    for qid in run:
        retrieved = run[qid]
        qrels = qrels_all.get(qid, {})

        relevant_set = {doc_id for doc_id, rel in qrels.items() if rel > 0}

        p = precision_at_k(retrieved, relevant_set, k)
        r = recall_at_k(retrieved, relevant_set, k)
        f = f1(p, r)
        n = ndcg_at_k(retrieved, qrels, k)

        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
        n_scores.append(n)

    if not p_scores:
        return 0, 0, 0, 0

    return (
        sum(p_scores) / len(p_scores),
        sum(r_scores) / len(r_scores),
        sum(f_scores) / len(f_scores),
        sum(n_scores) / len(n_scores),
    )


# =========================
# EVALUATE HYBRID RUNS
# =========================
def evaluate_folder(folder, qrels):
    results = []

    if not folder.exists():
        return results

    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue
        if not file.startswith("hybrid_"):
            continue

        path = folder / file
        p, r, f1_score, ndcg = evaluate_run(path, qrels)
        results.append((file, p, r, f1_score, ndcg))

    results.sort(key=lambda x: x[4], reverse=True)
    return results


# =========================
# MAIN
# =========================
def evaluate_all():
    result_file = RESULTS_DIR / "evaluation_hybrid.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        for name, qrels_path in QRELS_FILES.items():
            if not qrels_path.exists():
                continue

            f.write(f"\n===== QRELS: {name} =====\n\n")
            qrels = load_qrels(qrels_path)

            f.write("---- HYBRID (BM25 + TRANSFORMER) ----\n")
            hybrid_results = evaluate_folder(RUNS_RRF_DIR, qrels)

            for file, p, r, f1_score, ndcg in hybrid_results:
                f.write(
                    f"{file}\n"
                    f"  P@10: {p:.4f} | R@10: {r:.4f} | F1@10: {f1_score:.4f} | NDCG@10: {ndcg:.4f}\n\n"
                )

            if hybrid_results:
                best = max(hybrid_results, key=lambda x: x[4])
                f.write("🔥 BEST HYBRID (by NDCG):\n")
                f.write(f"{best[0]} → NDCG@10: {best[4]:.4f}\n\n")

            f.write("=" * 60 + "\n")

    print(f"✅ Done → {result_file}")


if __name__ == "__main__":
    evaluate_all()
