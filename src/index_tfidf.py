import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from config import REVIEW_PROCESSED_FILE, INDEX_DIR

def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    texts, ids = [], []

    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            ids.append(d["doc_id"])

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    pickle.dump(vectorizer, open(INDEX_DIR / "tfidf_vectorizer.pkl", "wb"))
    pickle.dump(X, open(INDEX_DIR / "tfidf_matrix.pkl", "wb"))
    pickle.dump(ids, open(INDEX_DIR / "tfidf_doc_ids.pkl", "wb"))

    print("✅ TF-IDF index done")

if __name__ == "__main__":
    build()