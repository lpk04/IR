import subprocess
import sys


def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)

    print(f"\n[RUNNING] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed")
        sys.exit(result.returncode)

    print(f"[DONE] {script}")


def main():
    # =========================
    # 1. Data Preparation (Transformer)
    # =========================
    # Run if you want to prepare data for transformer-based retrieval (takes time) - only need to run once
    run_script("src/prepare_data_transformer.py")

    # =========================
    # 2. Indexing (Dense Embeddings)
    # =========================
    # Run if you want to build a new transformer index (takes time) - only need to run once
    run_script("src/index_transformer.py")

    # =========================
    # 3. Semantic Search Variants
    # =========================

    # 3.1 Dense retrieval (no cross-encoder)
    run_script("src/search_transformer_no_cross.py")

    # 3.2 Dense + Cross-Encoder reranking
    run_script("src/search_transfomer_with_cross.py")

    # 3.3 Hybrid (Sparse + Dense + Cross-Encoder)
    run_script("src/search_transfomer_sparse_cross.py")

    print("\n=== SEMANTIC PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()