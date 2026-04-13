import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

from config import (
    RESULTS_DIR,
    RUNS_DIR,
    RUNS_SEARCH_BM25_DIR,
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_TRANSFORMER_DIR,
)

RUN_FOLDERS = {
    "BM25": RUNS_SEARCH_BM25_DIR,
    "TFIDF": RUNS_SEARCH_TFIDF_DIR,
    "TRANSFORMER": RUNS_SEARCH_TRANSFORMER_DIR,
    "RRF/HYBRID": RUNS_DIR / "runs_rrf",
}

DEFAULT_QRELS_FILES = {
    "KEYWORD": RESULTS_DIR / "qrels_keyword.txt",
    "COUNT": RESULTS_DIR / "qrels_count.txt",
    "RATIO": RESULTS_DIR / "qrels_ratio.txt",
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
    return rel / k if k else 0


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
    p_scores, r_scores, f_scores, n_scores = [], [], [], []

    for qid in run:
        retrieved = run[qid]
        qrels = qrels_all.get(qid, {})

        relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)
        f = f1(p, r)
        n = ndcg_at_k(retrieved, qrels)

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
# DISCOVER RUN FILES
# =========================
def collect_run_files():
    run_files = []

    for label, folder in RUN_FOLDERS.items():
        if not folder.exists():
            continue

        for file_name in sorted(os.listdir(folder)):
            if not file_name.endswith(".txt"):
                continue

            run_files.append((label, folder / file_name))

    return run_files


# =========================
# PARSE ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "evaluation_best.txt"),
        help="Output evaluation file path.",
    )
    parser.add_argument(
        "--qrels",
        action="append",
        default=[],
        help="Custom qrels in the form NAME=path. Can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_qrels_files(qrels_args):
    if not qrels_args:
        return DEFAULT_QRELS_FILES

    qrels_files = {}
    for item in qrels_args:
        if "=" not in item:
            raise ValueError(f"Invalid --qrels value: {item}. Expected NAME=path")
        name, path = item.split("=", maxsplit=1)
        qrels_files[name] = Path(path)

    return qrels_files


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_file = Path(args.output)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    qrels_files = resolve_qrels_files(args.qrels)
    run_files = collect_run_files()

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("===== ALL RUN COMPARISON =====\n\n")

        if not run_files:
            f.write("No run files found.\n")
            print(f"✅ Done → {result_file}")
            return

        for qrels_name, qrels_path in qrels_files.items():
            if not qrels_path.exists():
                continue

            f.write(f"===== QRELS: {qrels_name} =====\n\n")
            qrels = load_qrels(qrels_path)

            results = []

            for system_name, run_path in run_files:
                run = load_run(run_path)
                p, r, f1_score, ndcg = evaluate(run, qrels)
                results.append((system_name, run_path.name, p, r, f1_score, ndcg))

            results.sort(key=lambda x: x[5], reverse=True)

            for system_name, file_name, p, r, f1_score, ndcg in results:
                f.write(f"[{system_name}] {file_name}\n")
                f.write(
                    f"  P@10: {p:.4f} | R@10: {r:.4f} | F1: {f1_score:.4f} | NDCG: {ndcg:.4f}\n\n"
                )

            best = results[0]
            f.write("🔥 BEST RESULT:\n")
            f.write(f"[{best[0]}] {best[1]} → NDCG: {best[5]:.4f}\n\n")
            f.write("=" * 60 + "\n\n")

    print(f"✅ Done → {result_file}")


if __name__ == "__main__":
    main()
