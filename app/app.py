import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import INDEX_DIR
from prepare import preprocess_text


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    # TF-IDF
    vectorizer = pickle.load(open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb"))
    X = pickle.load(open(INDEX_DIR / "tfidf_matrix.pkl", "rb"))
    tfidf_ids = pickle.load(open(INDEX_DIR / "tfidf_doc_ids.pkl", "rb"))

    # BM25 (chọn 1 bộ tham số)
    bm25 = pickle.load(open(INDEX_DIR / "bm25_1.5_0.75.pkl", "rb"))
    bm25_ids = pickle.load(open(INDEX_DIR / "bm25_doc_ids.pkl", "rb"))

    return vectorizer, X, tfidf_ids, bm25, bm25_ids


# =========================
# SEARCH FUNCTIONS
# =========================
def search_tfidf(query, vectorizer, X, ids):
    q = preprocess_text(query)
    q_vec = vectorizer.transform([q])

    scores = cosine_similarity(q_vec, X).flatten()
    idx = np.argsort(scores)[::-1][:10]

    return [(ids[i], scores[i]) for i in idx]


def search_bm25(query, bm25, ids):
    tokens = preprocess_text(query).split()
    scores = bm25.get_scores(tokens)

    idx = np.argsort(scores)[::-1][:10]

    return [(ids[i], scores[i]) for i in idx]


# =========================
# UI
# =========================
st.title("🔍 Review Search System")

query = st.text_input("👉 Nhập câu tìm kiếm:")

model_choice = st.selectbox(
    "Chọn model:",
    ["TF-IDF (Cosine)", "BM25"]
)

if st.button("Search") and query:
    vectorizer, X, tfidf_ids, bm25, bm25_ids = load_models()

    if model_choice == "TF-IDF (Cosine)":
        results = search_tfidf(query, vectorizer, X, tfidf_ids)
    else:
        results = search_bm25(query, bm25, bm25_ids)

    st.subheader("📊 Kết quả:")

    for rank, (doc_id, score) in enumerate(results, 1):
        st.write(f"{rank}. DocID: {doc_id} | Score: {score:.4f}")