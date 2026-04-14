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

<h1>🔍 Search System (BM25 + Sentiment)</h1>

<form method="post" style="text-align:center;">
<input name="query" placeholder="Enter query..." required>
<button>Search</button>
</form>

{% if query %}

<p><span class="label">Query:</span> {{query}}</p>

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
<div class="card" onclick="showDetail({{loop.index0}})">
#{{loop.index}} {{doc.raw_text[:100]}}...
</div>
{% endfor %}
</div>

<div class="right">
<h3>📄 Chi tiết</h3>

{% for doc in results %}
<div id="detail{{loop.index0}}" class="details" style="display:none;">
<p><span class="label">📄 Full:</span> {{doc.raw_text}}</p>
<p><span class="label">📊 Score:</span> {{doc.score}}</p>
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
<p>Top 10: Pos {{pos10}} | Neg {{neg10}} | Neu {{neu10}}</p>
<p>Top 100: Pos {{pos100}} | Neg {{neg100}} | Neu {{neu100}}</p>
</div>

<div class="stats">
<h3>{{title}}</h3>

<div id="opp-list">
{% for doc in opposite %}
<div>
<div class="card" onclick="toggleOpp({{loop.index0}})">
{{doc.text[:120]}}...
</div>

<div id="opp{{loop.index0}}" class="details" style="display:none;">
<p><span class="label">📄 Full:</span> {{doc.text}}</p>
<p><span class="label">💬 Sentiment:</span> 
{% if doc.sent == "positive" %}
<span class="pos">Positive</span>
{% else %}
<span class="neg">Negative</span>
{% endif %}
</p>
</div>
</div>
{% endfor %}
</div>

<button onclick="loadMore()">Xem thêm</button>
</div>

<script>
function showDetail(idx){
    document.querySelectorAll('[id^="detail"]').forEach(x => x.style.display="none");
    document.getElementById("detail"+idx).style.display="block";
}

function toggleOpp(idx){
    let el = document.getElementById("opp"+idx);
    if(!el) return;
    el.style.display = el.style.display === "block" ? "none" : "block";
}

let allData = {{all_opposite|tojson}};
let currentIndex = {{opposite|length}};

function loadMore(){
    let container = document.getElementById("opp-list");

    for(let k=0;k<5 && currentIndex<allData.length;k++,currentIndex++){
        let d = allData[currentIndex];

        let sentHTML = d.sent === "positive"
            ? '<span class="pos">Positive</span>'
            : '<span class="neg">Negative</span>';

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

import sys
import os
import pickle
import joblib
import json
import random
import numpy as np
from flask import Flask, request, render_template_string

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prepare_data import preprocess_text
from src.config import get_bm25_app_paths, REVIEW_MERGERED_PROCESSED_FILE

app = Flask(__name__)

bm25, doc_ids = None, None
doc_sentiment = {}
processed_docs = []          # ← NEW: full processed documents in BM25 order
model = None
vectorizer = None


# ================= LOAD =================
def load_resources():
    global bm25, doc_ids, model, vectorizer, doc_sentiment, processed_docs

    paths = get_bm25_app_paths(5.0, 0.75)

    bm25 = pickle.load(open(paths["model"], "rb"))
    doc_ids = pickle.load(open(paths["ids"], "rb"))

    # PROCESSED documents (in exact same order as BM25 index)
    processed_docs = []
    doc_sentiment = {}
    with open(REVIEW_MERGERED_PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            processed_docs.append(d)                                 # ← retrieve processed doc
            if "sentiment" in d:
                doc_sentiment[str(d["doc_id"])] = d["sentiment"]

    model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.joblib"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression_vectorizer.joblib"))

    print(f"✅ Loaded {len(processed_docs)} processed docs")


# ================= SENTIMENT =================
def predict_query_sentiment(query):
    q_clean = preprocess_text(query)
    vec = vectorizer.transform([q_clean])
    pred = model.predict(vec)[0]
    return "positive" if pred == 1 else "negative"


# ================= SEARCH (CHANGED LOGIC) =================
def search(query, alpha=0.2):
    q_clean = preprocess_text(query)
    tokens = q_clean.split()

    q_sent = predict_query_sentiment(query)

    scores = bm25.get_scores(tokens)

    # Top 100 indices
    top_idx = np.argpartition(scores, -100)[-100:]

    results = []

    for i in top_idx:
 
        proc_doc = processed_docs[i]

        doc_id = str(proc_doc["doc_id"])

        base_score = scores[i]

        doc_sent = doc_sentiment.get(doc_id, "neutral")

        if doc_sent == q_sent:
            boost = 1
        elif doc_sent == "neutral":
            boost = 0
        else:
            boost = -1

        final = base_score + alpha * boost
        results.append((doc_id, final))

    # Sort only 100 items
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
            for doc_id, _ in docs:
                s = doc_sentiment.get(doc_id, "neutral")
                if s == "positive":
                    pos += 1
                elif s == "negative":
                    neg += 1
                else:
                    neu += 1
            return pos, neg, neu

        pos10, neg10, neu10 = count_stats(top10)
        pos100, neg100, neu100 = count_stats(top100)

        data = []
        for doc_id, score in top10:
            data.append({
                "raw_text": processed_docs[doc_ids.index(doc_id)]["raw_text"], 
                "score": round(score, 4),
                "sent": doc_sentiment.get(doc_id, "neutral")
            })

        target = "negative" if q_sent == "positive" else "positive"
        title = "🔄 Bình luận trái chiều"

        candidates = [d for d in top100 if doc_sentiment.get(d[0]) == target]
        random.shuffle(candidates)

        all_opposite = [{
            "text": processed_docs[doc_ids.index(d[0])]["raw_text"],
            "sent": doc_sentiment.get(d[0], "neutral")
        } for d in candidates]

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