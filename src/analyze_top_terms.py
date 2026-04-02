import json
from collections import Counter
from tqdm import tqdm

from config import REVIEW_PROCESSED_FILE, TOP_TERMS_FILE


def analyze(top_k=20):
    counter = Counter()

    total_docs = 0
    total_length = 0
    max_length = 0
    min_length = float("inf")

    # =========================
    # Duyệt dữ liệu
    # =========================
    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Analyzing"):
            data = json.loads(line)

            tokens = data["text"].split()
            length = len(tokens)

            # cập nhật thống kê
            total_docs += 1
            total_length += length
            max_length = max(max_length, length)
            min_length = min(min_length, length)

            # đếm từ
            counter.update(tokens)

    avg_length = total_length / total_docs if total_docs > 0 else 0

    # =========================
    # Top words
    # =========================
    top_words = counter.most_common(top_k)

    # =========================
    # Ghi file
    # =========================
    TOP_TERMS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TOP_TERMS_FILE, "w", encoding="utf-8") as f:
        f.write("===== DATA ANALYSIS =====\n\n")

        f.write(f"Total documents: {total_docs}\n")
        f.write(f"Average length: {avg_length:.2f}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Min length: {min_length}\n\n")

        f.write(f"Top {top_k} words:\n")
        for word, freq in top_words:
            f.write(f"{word}: {freq}\n")


if __name__ == "__main__":
    analyze()