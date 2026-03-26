import json
from tqdm import tqdm

from config import (
    YELP_REVIEW_FILE,
    REVIEW_RAW_FILE,
    REVIEW_PROCESSED_FILE,
    TRACE_DIR,
    TRACE_FILE,
    MAX_DOCS
)

import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# =========================
# SENTIMENT
# =========================
def get_sentiment(stars):
    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"


# =========================
# POS mapping
# =========================
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    # remove newline
    text = text.replace("\n", " ")

    # lowercase
    text = text.lower()

    # remove special char
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra space
    text = re.sub(r"\s+", " ", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # POS tagging
    pos_tags = pos_tag(tokens)

    # lemmatize
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    return " ".join(tokens)


# =========================
# STEP 1: EXTRACT
# =========================
def extract_data():
    REVIEW_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(YELP_REVIEW_FILE, "r", encoding="utf-8") as fin, \
         open(REVIEW_RAW_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Extracting"):
            try:
                data = json.loads(line)
            except:
                continue

            doc = {
                "doc_id": str(count),
                "text": data.get("text", ""),
                "rating": data.get("stars", 0)
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            count += 1
            if count >= MAX_DOCS:
                break

    print(f"\n✅ Extracted → {REVIEW_RAW_FILE}")


# =========================
# STEP 2: PREPROCESS
# =========================
def preprocess_data():
    REVIEW_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    with open(REVIEW_RAW_FILE, "r", encoding="utf-8") as fin, \
         open(REVIEW_PROCESSED_FILE, "w", encoding="utf-8") as fout, \
         open(TRACE_FILE, "w", encoding="utf-8") as log_f:

        for i, line in enumerate(tqdm(fin, desc="Preprocessing")):
            data = json.loads(line)

            raw_text = data["text"]
            clean_text = preprocess_text(raw_text)

            # log 20 dòng đầu
            if i < 20:
                log_f.write("=" * 50 + "\n")
                log_f.write(f"DOC {data['doc_id']}\n")
                log_f.write("RAW:\n" + raw_text + "\n\n")
                log_f.write("CLEAN:\n" + clean_text + "\n\n")

            doc = {
                "doc_id": data["doc_id"],
                "text": clean_text,
                "rating": data["rating"],
                "sentiment": get_sentiment(data["rating"])
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n✅ Processed → {REVIEW_PROCESSED_FILE}")
    print(f"📝 Log → {TRACE_FILE}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    extract_data()
    preprocess_data()