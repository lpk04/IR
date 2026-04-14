import subprocess
import sys
from pathlib import Path


def run_script(script, args=None):
    """Execute a script with optional arguments and fail fast on error."""
    if not Path(script).exists():
        print(f"[ERROR] File not found: {script}")
        sys.exit(1)

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
    # 1. Lexical Evaluation
    # =========================
    run_script("src/evaluate_results.py")

    # =========================
    # 2. Transformer Evaluation
    # =========================
    run_script("src/evaluate_results_transfomer.py")

    # =========================
    # 3. Sentiment Reranking Evaluation
    # =========================
    run_script("src/evaluate_results_sentiment.py")

    # =========================
    # 4. Alpha Tuning Evaluation
    # =========================
    run_script("src/evaluate_alpha.py")

    # =========================
    # 5. Hybrid Evaluation
    # =========================
    run_script("src/evaluate_results_hybrid.py")

    # =========================
    # 6. Best Model Summary
    # =========================
    run_script("src/evaluate_result_best.py")

    print("\n=== EVALUATION PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()