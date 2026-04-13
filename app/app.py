# ================= HTML =================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Search System</title>

<style>
body { font-family: Arial; background:#f5f7fb; margin:0; }

.wrapper { width:1100px; margin:auto; padding:30px; }

h1 { text-align:center; }

.label { font-weight:bold; color:#333; }

.pos { color:#16a34a; font-weight:bold; }
.neg { color:#dc2626; font-weight:bold; }
.neu { color:#6b7280; font-weight:bold; }

.container { display:flex; gap:20px; }

.left { width:45%; }
.right { width:55%; }

.card {
    background:white;
    padding:12px;
    margin-bottom:10px;
    border-radius:8px;
    cursor:pointer;
}

.card:hover { background:#eef3ff; }

.details {
    background:white;
    padding:15px;
    border-radius:8px;
    margin-top:5px;
}

.stats {
    margin-top:20px;
    padding:12px;
    background:white;
    border-radius:8px;
}

input {
    width:400px;
    padding:10px;
    border-radius:6px;
    border:1px solid #ccc;
}

button {
    padding:10px 15px;
    background:#4a6cf7;
    color:white;
    border:none;
    border-radius:6px;
}
</style>
</head>

<body>

<div class="wrapper">

<h1>🔍 Search System (all-MiniLM-L6-v2 + Sentiment)</h1>

<form method="post" style="text-align:center;">
    <input name="query" placeholder="Enter query..." required>
    <button type="submit">Search</button>
</form>

{% if query %}

<p><span class="label">Query:</span> {{ query }}</p>

<p>
    <span class="label">Query Sentiment:</span>
    {% if q_sent == "positive" %}
        <span class="pos">Positive</span>
    {% else %}
        <span class="neg">Negative</span>
    {% endif %}
</p>

<div class="container">

<div class="left">
<h3>📌 Top 10</h3>

{% for doc in results %}
<div class="card" onclick="showDetail({{ loop.index0 }})">
    #{{ loop.index }} {{ doc.raw_text[:100] }}...
</div>
{% endfor %}
</div>

<div class="right">
<h3>📄 Chi tiết</h3>

{% for doc in results %}
<div id="detail{{ loop.index0 }}" class="details" style="display:none;">
    <p><span class="label">📄 Full:</span> {{ doc.raw_text }}</p>
    <p><span class="label">📊 Score:</span> {{ doc.score }}</p>
    <p><span class="label">💬 Sentiment:</span>
        {% if doc.sent == "positive" %}
            <span class="pos">Positive</span>
        {% elif doc.sent == "negative" %}
            <span class="neg">Negative</span>
        {% else %}
            <span class="neu">Neutral</span>
        {% endif %}
    </p>
</div>
{% endfor %}
</div>

</div>

<div class="stats">
<h3>📊 Thống kê</h3>
<p>Top 10: Pos {{ pos10 }} | Neg {{ neg10 }} | Neu {{ neu10 }}</p>
<p>Top 100: Pos {{ pos100 }} | Neg {{ neg100 }} | Neu {{ neu100 }}</p>
</div>

<div class="stats">
<h3>{{ title }}</h3>

<div id="opp-list">
{% for doc in opposite %}
<div>
    <div class="card" onclick="toggleOpp({{ loop.index0 }})">
        {{ doc.text[:120] }}...
    </div>

    <div id="opp{{ loop.index0 }}" class="details" style="display:none;">
        <p><span class="label">📄 Full:</span> {{ doc.text }}</p>
        <p><span class="label">💬 Sentiment:</span>
            {% if doc.sent == "positive" %}
                <span class="pos">Positive</span>
            {% elif doc.sent == "negative" %}
                <span class="neg">Negative</span>
            {% else %}
                <span class="neu">Neutral</span>
            {% endif %}
        </p>
    </div>
</div>
{% endfor %}
</div>

<button type="button" onclick="loadMore()">Xem thêm</button>
</div>

<script>
function showDetail(idx){
    document.querySelectorAll('[id^="detail"]').forEach(x => x.style.display="none");
    document.getElementById("detail" + idx).style.display = "block";
}

function toggleOpp(idx){
    let el = document.getElementById("opp" + idx);
    if(!el) return;
    el.style.display = el.style.display === "block" ? "none" : "block";
}

let allData = {{ all_opposite|tojson }};
let currentIndex = {{ opposite|length }};

function loadMore(){
    let container = document.getElementById("opp-list");

    for(let k = 0; k < 5 && currentIndex < allData.length; k++, currentIndex++){
        let d = allData[currentIndex];

        let sentHTML = d.sent === "positive"
            ? '<span class="pos">Positive</span>'
            : (d.sent === "negative"
                ? '<span class="neg">Negative</span>'
                : '<span class="neu">Neutral</span>');

        let block = document.createElement("div");

        block.innerHTML = `
        <div>
            <div class="card" onclick="toggleOpp(${currentIndex})">
                ${d.text.substring(0,120)}...
            </div>

            <div id="opp${currentIndex}" class="details" style="display:none;">
                <p><span class="label">📄 Full:</span> ${d.text}</p>
                <p><span class="label">💬 Sentiment:</span> ${sentHTML}</p>
            </div>
        </div>`;

        container.appendChild(block);
    }
}
</script>

{% endif %}

</div>
</body>
</html>
"""

import os
import sys
import json
import pickle
import random
import joblib
import numpy as np

from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prepare_data import preprocess_text

app = Flask(__name__)

# ================= GLOBALS =================
raw_docs = {}
doc_sentiment = {}
doc_ids = []
doc_texts = []
doc_embeddings = None

sent_model = None
clf_model = None
vectorizer = None

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
RAW_CLEANED_FILE = os.path.join(BASE_DIR, "data", "raw", "yelp_reviews_100000_cleaned.jsonl")

EMBED_FILE = os.path.join(INDEX_DIR, "transformer_all_MiniLM_L6_v2_embeddings.npy")
DOC_IDS_FILE = os.path.join(INDEX_DIR, "transformer_all_MiniLM_L6_v2_doc_ids.pkl")

CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_vectorizer.joblib")


# ================= LOAD DOCUMENTS =================
def load_cleaned_reviews():
    """
    Load returned reviews from yelp_reviews_100000_cleaned.jsonl.
    Expected schema:
    {"doc_id": "...", "text": "...", "rating": ..., "sentiment": "..."}
    """
    global raw_docs, doc_sentiment, doc_texts

    raw_docs = {}
    doc_sentiment = {}
    doc_texts = []

    with open(RAW_CLEANED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_id = str(d.get("doc_id", ""))
            text = str(d.get("text", ""))
            sentiment = d.get("sentiment", "neutral")

            if not doc_id:
                continue

            raw_docs[doc_id] = text
            doc_sentiment[doc_id] = sentiment
            doc_texts.append(text)


def load_resources():
    global doc_ids, doc_embeddings, sent_model, clf_model, vectorizer

    os.makedirs(INDEX_DIR, exist_ok=True)

    if not os.path.exists(RAW_CLEANED_FILE):
        raise FileNotFoundError(f"Missing cleaned review file: {RAW_CLEANED_FILE}")
    if not os.path.exists(EMBED_FILE):
        raise FileNotFoundError(f"Missing embedding file: {EMBED_FILE}")
    if not os.path.exists(DOC_IDS_FILE):
        raise FileNotFoundError(f"Missing doc ids file: {DOC_IDS_FILE}")

    load_cleaned_reviews()

    with open(DOC_IDS_FILE, "rb") as f:
        doc_ids = [str(x) for x in pickle.load(f)]

    doc_embeddings = np.load(EMBED_FILE).astype(np.float32)

    if len(doc_ids) != len(doc_embeddings):
        raise ValueError(
            f"Embedding mismatch: doc_ids={len(doc_ids)}, embeddings={len(doc_embeddings)}"
        )

    # The cleaned file is used for returning/displaying reviews.
    # Align texts to doc_ids when possible.
    if len(doc_texts) < len(doc_ids):
        raise ValueError(
            f"Cleaned review count is smaller than doc_ids: doc_texts={len(doc_texts)}, doc_ids={len(doc_ids)}"
        )

    # Load sentiment classifier for query/doc sentiment prediction
    clf_model = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Load embedding model for query encoding
    sent_model = SentenceTransformer(MODEL_NAME)


# ================= SENTIMENT =================
def predict_query_sentiment(query):
    q_clean = preprocess_text(query)
    vec = vectorizer.transform([q_clean])
    pred = clf_model.predict(vec)[0]
    return "positive" if int(pred) == 1 else "negative"


def predict_doc_sentiment(text):
    text_clean = preprocess_text(text)
    vec = vectorizer.transform([text_clean])
    pred = clf_model.predict(vec)[0]
    return "positive" if int(pred) == 1 else "negative"


# ================= SEARCH =================
def search(query, alpha=0.2):
    q_sent = predict_query_sentiment(query)

    q_emb = sent_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].astype(np.float32)

    scores = np.dot(doc_embeddings, q_emb)

    top_k = min(100, len(scores))
    top_idx = np.argpartition(scores, -top_k)[-top_k:]

    results = []
    for i in top_idx:
        doc_id = str(doc_ids[i])
        base_score = float(scores[i])

        text = raw_docs.get(doc_id, "")
        doc_sent = predict_doc_sentiment(text)

        boost = 1 if doc_sent == q_sent else -1
        final_score = base_score + alpha * boost

        results.append((doc_id, final_score, doc_sent))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    top10 = results[:10]
    top100 = results
    return q_sent, top10, top100


# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]

        q_sent, top10, top100 = search(query)

        def count_stats(docs):
            pos = neg = neu = 0
            for _, _, sent in docs:
                if sent == "positive":
                    pos += 1
                elif sent == "negative":
                    neg += 1
                else:
                    neu += 1
            return pos, neg, neu

        pos10, neg10, neu10 = count_stats(top10)
        pos100, neg100, neu100 = count_stats(top100)

        data = []
        for doc_id, score, sent in top10:
            data.append({
                "raw_text": raw_docs.get(doc_id, ""),
                "score": round(score, 4),
                "sent": sent
            })

        target = "negative" if q_sent == "positive" else "positive"
        title = "🔄 Bình luận trái chiều"

        candidates = [d for d in top100 if d[2] == target]
        random.shuffle(candidates)

        all_opposite = [{
            "text": raw_docs.get(doc_id, ""),
            "sent": sent
        } for doc_id, _, sent in candidates]

        return render_template_string(
            HTML,
            query=query,
            q_sent=q_sent,
            results=data,
            pos10=pos10, neg10=neg10, neu10=neu10,
            pos100=pos100, neg100=neg100, neu100=neu100,
            opposite=all_opposite[:5],
            all_opposite=all_opposite,
            title=title
        )

    return render_template_string(HTML)


# ================= RUN =================
if __name__ == "__main__":
    load_resources()
    app.run(debug=True, use_reloader=False)