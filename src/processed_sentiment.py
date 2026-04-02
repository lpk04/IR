import json
import re
from tqdm import tqdm

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from config import REVIEW_RAW_FILE, REVIEW_SENTIMENT_FILE_PROCESSED, TRACE_DIR

# =========================
# STOPWORDS
# =========================
STOP_WORDS = set(stopwords.words("english"))

KEEP_WORDS = {
    "not", "no", "never", "none", "nor", "cannot",
    "very", "too", "so", "really", "extremely",
    "quite", "pretty", "rather", "fairly",
    "more", "most", "less", "least",
    "always", "often", "sometimes", "rarely",
    "should", "would", "could",
    "again", "still", "just"
}

CUSTOM_STOPWORDS = STOP_WORDS - KEEP_WORDS

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
# PREPROCESS TEXT
# =========================
def preprocess_text(text):
    text = text.replace("\n", " ")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        t for t in tokens
        if (t not in CUSTOM_STOPWORDS and (len(t) > 2 or t in KEEP_WORDS))
    ]

    pos_tags = pos_tag(tokens)

    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    return " ".join(tokens)

# =========================
# PROCESS
# =========================
def process_sentiment():
    REVIEW_SENTIMENT_FILE_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    log_path = TRACE_DIR / "sentiment_preprocess_log.txt"

    with open(REVIEW_RAW_FILE, "r", encoding="utf-8") as fin, \
         open(REVIEW_SENTIMENT_FILE_PROCESSED, "w", encoding="utf-8") as fout, \
         open(log_path, "w", encoding="utf-8") as log_f:

        for i, line in enumerate(tqdm(fin, desc="Processing", leave=False)):
            data = json.loads(line)

            raw_text = data.get("text", "")
            rating = data.get("rating", 0)

            clean_text = preprocess_text(raw_text)
            sentiment = get_sentiment(rating)

            # log 20 dòng đầu
            if i < 20:
                log_f.write("=" * 50 + "\n")
                log_f.write(f"DOC {data.get('doc_id')}\n")
                log_f.write("RAW:\n" + raw_text + "\n\n")
                log_f.write("CLEAN:\n" + clean_text + "\n\n")
                log_f.write("SENTIMENT: " + sentiment + "\n\n")

            doc = {
                "doc_id": data.get("doc_id"),
                "text": clean_text,
                "label": sentiment
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_sentiment()