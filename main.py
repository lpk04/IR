from collections import defaultdict
from pathlib import Path

RESULT_FILE = Path(r"D:\IR\demo\run\runs_search_bm25\bm25_2.0_0.75.txt")
OUTPUT_FILE = Path(r"D:\IR\demo\results\top10_results.txt")


def load_results():
    results = defaultdict(list)

    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 6:
                continue

            qid, _, doc_id, rank, score, _ = parts

            results[qid].append({
                "doc_id": doc_id,
                "rank": int(rank),
                "score": float(score)
            })

    return results


def save_top_k(results, k=10):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qid in sorted(results.keys()):
            f.write(f"===== {qid} =====\n")

            docs = sorted(results[qid], key=lambda x: x["score"], reverse=True)

            for i, d in enumerate(docs[:k], 1):
                f.write(f"{i}. DocID: {d['doc_id']} | Score: {d['score']:.4f}\n")

            f.write("\n")


def main():
    results = load_results()
    save_top_k(results, k=10)


if __name__ == "__main__":
    main()