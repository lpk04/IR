import json
import re


# =========================
# 1. Expand contractions (giữ negation)
# =========================
def expand_contractions(text):
    text = text.lower()

    contractions = {
        "can't": "can not",
        "won't": "will not",
        "n't": " not",

        "'re": " are",
        "'s": " is",       
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",

        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "what's": "what is",
        "who's": "who is",
        "where's": "where is",
        "how's": "how is",

        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",

        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",

        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "they'll": "they will",

        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
    }

    for key, value in contractions.items():
        text = text.replace(key, value)

    return text

# =========================
# 2. Clean text (chuẩn baseline)
# =========================
def clean_text(text):
    text = expand_contractions(text)

    # giữ chữ, số và dấu '
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # normalize kéo dài chữ: sooo → so
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # xóa khoảng trắng dư
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# 3. Convert rating → sentiment
# =========================
def rating_to_sentiment(r, mode="binary"):
    if mode == "binary":
        if r <= 2:
            return "negative"
        elif r >= 4:
            return "positive"
        else:
            return None   # bỏ neutral
    else:
        if r <= 2:
            return "negative"
        elif r == 3:
            return "neutral"
        else:
            return "positive"


# =========================
# 4. Process file
# =========================
def process_file(input_path, output_path, mode="binary"):
    count_in = 0
    count_out = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            count_in += 1
            data = json.loads(line)

            text = data.get("text", "")
            rating = data.get("rating", 0)

            sentiment = rating_to_sentiment(rating, mode)

            # bỏ neutral nếu binary
            if sentiment is None:
                continue

            cleaned_text = clean_text(text)

            new_data = {
                "doc_id": data.get("doc_id"),
                "text": cleaned_text,
                "rating": rating,
                "sentiment": sentiment
            }

            fout.write(json.dumps(new_data) + "\n")
            count_out += 1

    print(f"Input: {count_in}")
    print(f"Output: {count_out}")


# =========================
# 5. MAIN
# =========================
if __name__ == "__main__":
    from pathlib import Path

    # Project root (adjust if script is nested)
    BASE_DIR = Path(__file__).resolve().parents[2]

    INPUT_FILE = BASE_DIR / "data" / "yelp_reviews_100000_changed.jsonl"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "yelp_reviews_100000_sentiment.jsonl"

    # mode = "binary" (khuyên dùng) hoặc "multiclass"
    process_file(INPUT_FILE, OUTPUT_FILE, mode="binary")