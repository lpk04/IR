import os
from collections import defaultdict
from pathlib import Path

from config import (
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_BM25_DIR,
    RUNS_DIR
)

# =========================
# CONFIG
# =========================
K = 60


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
# GET RANK MAP
# =========================
def get_rank_map(run):
    rank_map = {}

    for qid in run:
        rank_map[qid] = {}
        for i, doc_id in enumerate(run[qid], start=1):
            rank_map[qid][doc_id] = i

    return rank_map


# =========================
# RRF
# =========================
def rrf_fusion(run1, run2):
    fused = defaultdict(list)

    rm1 = get_rank_map(run1)
    rm2 = get_rank_map(run2)

    all_qids = set(rm1.keys()) | set(rm2.keys())

    for qid in all_qids:
        scores = defaultdict(float)

        for doc_id, rank in rm1.get(qid, {}).items():
            scores[doc_id] += 1.0 / (K + rank)

        for doc_id, rank in rm2.get(qid, {}).items():
            scores[doc_id] += 1.0 / (K + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused[qid] = [doc for doc, _ in ranked]

    return fused


# =========================
# AUTO PICK BEST FILE
# =========================
def pick_best_file(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    if not files:
        raise ValueError(f"❌ No run files in {folder}")

    # 👉 đơn giản: lấy file cuối (thường là best bạn chạy)
    # hoặc bạn có thể hard-code nếu muốn chắc chắn
    files.sort()
    return folder / files[-1]


# =========================
# SAVE
# =========================
def save_run(fused_run, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for qid, docs in fused_run.items():
            for rank, doc_id in enumerate(docs[:100], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {1.0/(rank):.4f} RRF\n")

    print(f"✅ Saved → {output_file}")


# =========================
# MAIN
# =========================
def main():
    print("📥 Finding best run files...")

    tfidf_file = pick_best_file(RUNS_SEARCH_TFIDF_DIR)
    bm25_file  = pick_best_file(RUNS_SEARCH_BM25_DIR)

    print(f"👉 TF-IDF: {tfidf_file}")
    print(f"👉 BM25:   {bm25_file}")

    print("📥 Loading runs...")
    run_tfidf = load_run(tfidf_file)
    run_bm25  = load_run(bm25_file)

    print("🔀 Running RRF fusion...")
    fused = rrf_fusion(run_tfidf, run_bm25)

    output_file = RUNS_DIR / "runs_rrf" / "rrf_tfidf_bm25.txt"

    save_run(fused, output_file)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()