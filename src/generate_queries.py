import json
from collections import Counter
from config import REVIEW_PROCESSED_FILE, QUERY_FILE

def generate_queries(top_k=20, num_queries=10):
    counter = Counter()

    # =========================
    # Đếm từ
    # =========================
    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            counter.update(data["text"].split())

    # =========================
    # lấy top words
    # =========================
    top_words = [w for w, _ in counter.most_common(top_k)]

    # =========================
    # tạo query (ghép 2 từ)
    # =========================
    queries = []

    for i in range(num_queries):
        w1 = top_words[i]
        w2 = top_words[(i + 1) % top_k]

        query = f"{w1} {w2}"
        queries.append((f"Q{i+1}", query))

    return queries


def save_queries(queries):
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(QUERY_FILE, "w", encoding="utf-8") as f:
        for qid, q in queries:
            f.write(f"{qid}\t{q}\n")

    print(f"✅ Saved → {QUERY_FILE}")


if __name__ == "__main__":
    queries = generate_queries()
    save_queries(queries)

    #print("\nGenerated Queries:")
    #for q in queries:
        #print(q)