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
    run_script("src/filter_out_non_en.py", [])
    run_script("src/merge_data.py", [])

    # =========================
    # 3. Indexing - BM25
    # =========================
    bm25_configs = [
        # ("1.2", "0.75"),
        # ("1.5", "0.75"),
        # ("2.0", "0.75"),
        ("5.0", "0.75"),
    ]

    for k1, b in bm25_configs:
        run_script("src/index_bm25_app.py", [
            "--k1", k1,
            "--b", b
        ])

    # =========================
    # 4. Run App
    # =========================
    run_script("app/app.py", [])

    print("\n=== FULL APP PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()