# Information Retrieval (IR) Pipeline

This repository implements an end-to-end Information Retrieval system supporting lexical, semantic, and hybrid retrieval approaches.

---

## 1. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Setup virtual enviroment:
```bash
py -m venv .venv
.venv\Scripts\activate
```
or
```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Download NLTK Resources

Before running preprocessing steps, download required NLTK packages:

```bash
py download_nltk.py
```

---

## 3. Data Preprocessing Pipeline

Run the following scripts in order:

### 3.1 Convert Raw Yelp Data

```bash
py src\change_yelp_raw.py
```

### 3.2 Build Candidate Pool

Construct document candidate pool for retrieval tasks:

```bash
py src\build_candidate_pool.py
```

### 3.3 Generate Qrels (TREC format)

Create relevance judgments for evaluation:

```bash
py src\generate_qrels.py
```

---

## 4. Retrieval Pipelines

### 4.1 Lexical Retrieval (BM25 / TF-IDF)

```bash
py src\tools\pipeline_lexical.py
```

### 4.2 Semantic Retrieval (Sentence Transformers)

Uses embedding models such as:

* all-MiniLM-L6-v2
* ms-marco-MiniLM-L-6-v2

Run:

```bash
py src\tools\pipeline_semantic.py
```

### 4.3 Hybrid Retrieval

Combines lexical and semantic ranking methods:

```bash
py src\tools\pipeline_hybrid.py
```

---

## 5. System Overview

The pipeline includes:

* Raw data transformation (Yelp format normalization)
* Candidate pool coznzstruction
* Qrels generation (TREC standard)
* Lexical retrieval (BM25 / TF-IDF)
* Semantic retrieval (Sentence Transformers)
* Hybrid fusion retrieval

---

## 6. Lauch app

```bash
py src\tools\pipepline_app.py
```

---

## 7. Notes

* Ensure preprocessing steps are executed before running retrieval pipelines.
* Semantic and hybrid pipelines require SentenceTransformer models.
* Output formats are designed for IR evaluation workflows.

---

## 8. Environment

* Python 3.10+
* Recommended: CUDA-enabled GPU for embedding generation
