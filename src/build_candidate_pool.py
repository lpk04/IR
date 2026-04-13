"""Build a candidate pool for manual relevance labeling.

For each query in `queries.txt` this script collects the top-K candidates
from available run files (BM25, TF-IDF, Transformer), merges them,
and writes `data/candidate_pool.csv` for manual labeling.

Run with:
  python src/build_candidate_pool.py --top-k 100
"""
import json
import csv
import os
from collections import defaultdict
from pathlib import Path

TOP_K = 100

ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = ROOT / "queries.txt"
PROCESSED_TRANSFORMER = ROOT / "data" / "processed" / "yelp_reviews_100000_processed.jsonl"
RUNS_BM25_DIR = ROOT / "run" / "runs_search_bm25"
RUNS_TFIDF_DIR = ROOT / "run" / "runs_search_tfidf"
RUNS_TRANSFORMER_DIR = ROOT / "run" / "runs_search_transformer"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "candidate_pool.csv"


def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if len(parts) >= 2:
                qid, q = parts[0], parts[1]
            else:
                qid, q = f"Q{len(queries)+1}", parts[0]
            queries.append((qid, q))
    return queries


def build_line_to_doc_map(processed_path):
    """Return mappings: line_no -> doc_id, doc_id -> metadata"""
    line_to_doc = {}
    doc_meta = {}

    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # file contains `review_line_no` and `review_id`
            line_no = d.get("review_line_no") or d.get("review_line") or d.get("line_no")
            doc_id = d.get("review_id") or d.get("id") or d.get("doc_id")
            text = d.get("text")
            rating = d.get("rating")
            sentiment = d.get("sentiment")

            if line_no is not None and doc_id is not None:
                line_to_doc[str(line_no)] = doc_id

            if doc_id is not None:
                doc_meta[str(doc_id)] = {
                    "text": text,
                    "rating": rating,
                    "sentiment": sentiment,
                }

    return line_to_doc, doc_meta


def parse_run_file(path, top_k=TOP_K):
    """Parse a run file and return dict qid -> list of docids (as strings)."""
    results = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[2]
            if len(results[qid]) < top_k:
                results[qid].append(docid)
    return results


def collect_candidates(queries, line_to_doc_map):
    candidates = {qid: {} for qid, _ in queries}

    # helper to merge from a run directory, tag source
    def merge_runs_from_dir(dir_path, source_label):
        if not dir_path.exists():
            return
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith('.txt'):
                continue
            path = dir_path / fname
            run = parse_run_file(path)
            for qid, docs in run.items():
                if qid not in candidates:
                    continue
                for rank, doc in enumerate(docs, 1):
                    # convert numeric line ids to doc_id if possible
                    doc_id = line_to_doc_map.get(doc, doc)
                    ent = candidates[qid].setdefault(doc_id, {"sources": set(), "ranks": {}})
                    ent["sources"].add(source_label)
                    if source_label not in ent["ranks"]:
                        ent["ranks"][source_label] = rank

    merge_runs_from_dir(RUNS_BM25_DIR, "BM25")
    merge_runs_from_dir(RUNS_TFIDF_DIR, "TFIDF")
    merge_runs_from_dir(RUNS_TRANSFORMER_DIR, "TRANSFORMER")

    return candidates


def auto_label_candidate(info):
    """Heuristically assign a graded relevance label from 0-3.

    Scale:
    0 = not relevant
    1 = little relevant
    2 = relevant
    3 = very relevant
    """
    source_count = len(info.get("sources", set()))
    ranks = info.get("ranks", {})
    best_rank = min(ranks.values()) if ranks else TOP_K + 1

    if source_count >= 3 or (source_count >= 2 and best_rank <= 10):
        return 3
    if source_count >= 2 or best_rank <= 20:
        return 2
    if source_count >= 1:
        return 1
    return 0


def write_candidate_csv(queries, candidates, doc_meta, out_csv_path):
    fields = ["qid", "query", "doc_id", "rating", "sentiment", "sources", "ranks", "label", "text"]
    with open(out_csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for qid, query in queries:
            pool = candidates.get(qid, {})
            for doc_id, info in pool.items():
                meta = doc_meta.get(str(doc_id), {})
                text = (meta.get("text") or "")
                rating = meta.get("rating")
                sentiment = meta.get("sentiment")

                writer.writerow({
                    "qid": qid,
                    "query": query,
                    "doc_id": doc_id,
                    "rating": rating,
                    "sentiment": sentiment,
                    "sources": ";".join(sorted(info["sources"])),
                    "ranks": json.dumps(info["ranks"]),
                    "label": auto_label_candidate(info),
                    "text": text.replace('\n', ' '),
                })


def main():
    print("🔄 Loading queries...")
    queries = load_queries(QUERIES_FILE)

    print("🔍 Building doc id map (this reads the transformer-processed file)...")
    line_to_doc_map, doc_meta = build_line_to_doc_map(PROCESSED_TRANSFORMER)

    print("🔀 Collecting candidates from runs...")
    candidates = collect_candidates(queries, line_to_doc_map)

    print(f"💾 Writing candidate CSV → {OUT_CSV}")
    write_candidate_csv(queries, candidates, doc_meta, OUT_CSV)

    print("✅ Done. data/candidate_pool.csv now includes an automatically assigned 'label' column on a 0-3 scale. Review and adjust labels as needed, then run labels_to_qrels.py to generate data/qrels.txt")


if __name__ == '__main__':
    main()
