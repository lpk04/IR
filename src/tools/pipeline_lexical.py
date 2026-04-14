import subprocess
import sys


def run_script(script, args):
    cmd = [sys.executable, script] + args
    print(f"\n[RUNNING] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with args {args}")
        sys.exit(result.returncode)

    print(f"[DONE] {script} {args}")


def main():
    # =========================
    # 1. Data Preparation
    # =========================
    #Run if you want to prepare data for TF-IDF/BM25 indexing (takes time) - only need to run once
    run_script("src/prepare_data.py", [])
    #Run if you want to prepare data for BM25 with sentiment keywords
    run_script("src/prepare_sentiment.py", [])
    run_script("src/sentiment/preprocess_sentiment.py", [])
    # =========================
    # 2. Indexing - TF-IDF
    # =========================
    tfidf_configs = [
        ("11", "false"),
        ("11", "true"),
        ("12", "false"),
        ("12", "true"),
    ]

    for ngram, sublinear in tfidf_configs:
        run_script("src/index_tfidf.py", [
            "--ngram", ngram,
            "--sublinear", sublinear
        ])

    # =========================
    # 3. Indexing - BM25
    # =========================
    bm25_configs = [
        ("1.2", "0.75"),
        ("1.5", "0.75"),
        ("2.0", "0.75"),
        ("5.0", "0.75"),
    ]

    for k1, b in bm25_configs:
        run_script("src/index_bm25.py", [
            "--k1", k1,
            "--b", b
        ])

    # =========================
    # 4. Search - TF-IDF
    # =========================
    for ngram, sublinear in tfidf_configs:
        run_script("src/search_tfidf.py", [
            "--ngram", ngram,
            "--sublinear", sublinear
        ])

    # =========================
    # 5. Search - BM25
    # =========================
    for k1, b in bm25_configs:
        run_script("src/search_bm25.py", [
            "--k1", k1,
            "--b", b
        ])

    # =========================
    # 6. Reranking - BM25 + Sentiment
    # =========================
    alpha_values = ["0.0", "0.2", "0.5", "1"]

    for alpha in alpha_values:
        run_script("src/rerank_bm25_sentiment.py", [
            "--k1", "1.2",
            "--b", "0.75",
            "--alpha", alpha
        ])

    print("\n=== FULL LEXICAL PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()