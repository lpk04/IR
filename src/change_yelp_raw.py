import json
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_FILE = Path("data/raw/yelp_reviews_100000.jsonl")
OUTPUT_FILE = Path("data/raw/yelp_reviews_100000_changed.jsonl")

MAX_DOCS = None  # set to int if you want limit


# =========================
# SENTIMENT
# =========================
def get_sentiment(stars):
    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    return "negative"


# =========================
# MAIN CONVERSION
# =========================
def convert():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Converting Yelp dataset"):
            if MAX_DOCS and count >= MAX_DOCS:
                break

            try:
                data = json.loads(line)
            except:
                continue

            text = data.get("text", "")
            rating = float(data.get("stars", 0))

            doc = {
                "doc_id": str(count),
                "text": text,  # ⚠️ NO preprocessing
                "rating": rating,
                "sentiment": get_sentiment(rating)
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            count += 1

    print(f"✅ Converted {count} documents")
    print(f"📁 Saved to: {OUTPUT_FILE}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    convert()