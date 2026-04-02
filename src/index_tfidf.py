import json
import pickle
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    REVIEW_PROCESSED_FILE,
    TFIDF_INDEX_DIR,
    TFIDF_TRACE_DIR,
    get_tfidf_paths
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ngram", type=str, required=True,
                        help="11, 12, 13 ...")
    parser.add_argument("--sublinear", type=str, required=True,
                        help="true / false")

    return parser.parse_args()


def build_index(ngram, use_sublinear):
    TFIDF_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TFIDF_TRACE_DIR.mkdir(parents=True, exist_ok=True)

    paths = get_tfidf_paths(ngram, use_sublinear)

    texts = []
    doc_ids = []

    # =========================
    # LOAD DATA
    # =========================
    with open(REVIEW_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            doc_ids.append(d["doc_id"])

    # =========================
    # BUILD TF-IDF
    # =========================
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=ngram,
        sublinear_tf=use_sublinear
    )

    X = vectorizer.fit_transform(texts)

    # =========================
    # SAVE PKL
    # =========================
    with open(paths["vectorizer"], "wb") as f:
        pickle.dump(vectorizer, f)

    with open(paths["matrix"], "wb") as f:
        pickle.dump(X, f)

    with open(paths["ids"], "wb") as f:
        pickle.dump(doc_ids, f)

    # =========================
    # TRACE (SIMULATE PKL)
    # =========================
    features = vectorizer.get_feature_names_out()

    with open(paths["trace"], "w", encoding="utf-8") as f:
        f.write(f"===== TF-IDF {ngram} | sublinear={use_sublinear} =====\n\n")

        # VECTOR
        f.write(">> vectorizer (first 5 vocab)\n")
        for i, w in enumerate(features[:5]):
            f.write(f"{i}: {w}\n")

        f.write("\n")

        # MATRIX
        f.write(">> matrix (first 5 docs)\n")

        for doc_idx in range(min(5, X.shape[0])):
            row = X[doc_idx]
            f.write(f"\nDoc {doc_idx}:\n")

            for i, v in zip(row.indices[:5], row.data[:5]):
                f.write(f"  ({i}, {features[i]}): {v:.4f}\n")

        f.write("\n")

        # IDS
        f.write(">> doc_ids (first 5)\n")
        for i in range(min(5, len(doc_ids))):
            f.write(f"{i}: {doc_ids[i]}\n")

    print(f"✅ Done TF-IDF {ngram} | sublinear={use_sublinear}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    # convert ngram "12" → (1,2)
    ngram = (int(args.ngram[0]), int(args.ngram[1]))

    # convert bool
    sublinear = args.sublinear.lower() == "true"

    build_index(ngram, sublinear)