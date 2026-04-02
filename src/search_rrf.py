import os
from collections import defaultdict

from config import (
    RUNS_SEARCH_TFIDF_DIR,
    RUNS_SEARCH_BM25_DIR,
    RUNS_DIR
)

# =========================
# CONFIG
# =========================
K = 60  # constant RRF


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
# RRF FUSION
# =========================
def rrf_fusion(runs):
    fused = defaultdict(dict)

    rank_maps = [get_rank_map(r) for r in runs]

    for qid in rank_maps[0]:
        scores = defaultdict(float)

        for rm in rank_maps:
            for doc_id, rank in rm[qid].items():
                scores[doc_id] += 1.0 / (K + rank)

        # sort
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused[qid] = [doc for doc, _ in ranked]

    return fused


# =========================
# SAVE RUN
# =========================
def save_run(fused_run, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for qid, docs in fused_run.items():
            for rank, doc_id in enumerate(docs[:10], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {1.0/(rank):.4f} RRF\n")

    print(f"✅ Saved → {output_file}")


# =========================
# MAIN
# =========================
def main():
   
    tfidf_file = RUNS_SEARCH_TFIDF_DIR / "tfidf_1_2_sub.txt"
    bm25_file = RUNS_SEARCH_BM25_DIR / "bm25_2.0_0.75.txt"
    sentiment_file = RUNS_SEARCH_BM25_DIR / "bm25_sentiment_2.0_0.75_a0.2.txt"

    print("📥 Loading runs...")

    run_tfidf = load_run(tfidf_file)
    run_bm25 = load_run(bm25_file)
    run_sent = load_run(sentiment_file)

    print("🔀 Running RRF fusion...")

    fused = rrf_fusion([run_tfidf, run_bm25, run_sent])

    output_file = RUNS_DIR / "runs_rrf" / "rrf_all.txt"

    save_run(fused, output_file)


if __name__ == "__main__":
    main()