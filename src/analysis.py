import json
from collections import Counter

from config import REVIEW_PROCESSED_FILE, ANALYSIS_DIR, ANALYSIS_RESULT_FILE


def analyze_data():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    lengths = []
    counter = Counter()

    total_docs = 0

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            words = data["text"].split()

            lengths.append(len(words))
            counter.update(words)

            total_docs += 1

    # =========================
    # STATS
    # =========================
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    top_words = counter.most_common(20)

    # =========================
    # SAVE RESULT
    # =========================
    with open(ANALYSIS_RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("===== DATA ANALYSIS =====\n\n")

        f.write(f"Total documents: {total_docs}\n")
        f.write(f"Average length: {avg_len:.2f}\n")
        f.write(f"Max length: {max_len}\n")
        f.write(f"Min length: {min_len}\n\n")

        f.write("Top 20 words:\n")
        for word, freq in top_words:
            f.write(f"{word}: {freq}\n")

    print(f"\n✅ Analysis saved → {ANALYSIS_RESULT_FILE}")


if __name__ == "__main__":
    analyze_data()