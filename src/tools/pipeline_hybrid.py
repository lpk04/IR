import subprocess
import sys
from pathlib import Path


def run_script(script, args=None):
    """Run a Python script with optional arguments and fail-fast behavior."""
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
    # 1. Hybrid - Lexical Only (RRF)
    # =========================
    run_script("src/search_hybrid_lexical_only_rrf.py")

    # =========================
    # 2. Hybrid - Lexical + Semantic (RRF)
    # =========================
    run_script("src/search_hybrid_lexical_semantic_rrf.py")

    print("\n=== HYBRID PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()