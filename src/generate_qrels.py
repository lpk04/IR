import os
from config import RUNS_DIR, RESULTS_DIR, RUN_FILE, QRELS_FILE

def generate_qrels(top_k=5):
    qrels = {}

    with open(RUN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            qid = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])

            if rank <= top_k:
                if qid not in qrels:
                    qrels[qid] = []

                qrels[qid].append(doc_id)

    return qrels


def save_qrels(qrels):
    with open(QRELS_FILE, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for doc_id in docs:
                f.write(f"{qid}\t{doc_id}\n")

    print(f"✅ Qrels saved → {QRELS_FILE}")


if __name__ == "__main__":
    qrels = generate_qrels(top_k=5)
    save_qrels(qrels)