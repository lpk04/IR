import os
from collections import defaultdict
from pathlib import Path

QRELS = Path("results/qrels_keyword.txt")
RUNS_DIR = Path("run/runs_search_transformer")


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels[qid].add(docid)
    return qrels


def load_run(path):
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[2]
            run[qid].append(docid)
    return run


def main():
    if not QRELS.exists():
        print("qrels file not found:", QRELS)
        return

    qrels = load_qrels(QRELS)

    for file in sorted(os.listdir(RUNS_DIR)):
        if not file.endswith('.txt'):
            continue
        path = RUNS_DIR / file
        run = load_run(path)

        total_q = 0
        q_with_match = 0
        total_matches = 0

        for qid, docs in run.items():
            total_q += 1
            relevant = qrels.get(qid, set())
            inter = relevant & set(docs)
            if inter:
                q_with_match += 1
                total_matches += len(inter)

        print(f"{file}: queries={total_q}, queries_with_match={q_with_match}, total_matches={total_matches}")


if __name__ == '__main__':
    main()
