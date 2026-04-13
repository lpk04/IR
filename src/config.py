from pathlib import Path

# =========================
# ROOT
# =========================
ROOT = Path(__file__).resolve().parent.parent

# =========================
# DATA
# =========================
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =========================
# INDEX
# =========================
INDEX_DIR = ROOT / "index"

# =========================
# DATA FILES
# =========================
# YELP_REVIEW_FILE = RAW_DATA_DIR / "yelp_academic_dataset_review.json"
YELP_REVIEW_FILE = RAW_DATA_DIR / "yelp_reviews_100000.jsonl"

# Extract (raw sau khi lấy từ Yelp)
# REVIEW_RAW_FILE = DATA_DIR / "review.jsonl"
REVIEW_RAW_FILE = RAW_DATA_DIR / "yelp_reviews_100000_cleaned.jsonl"
# Preprocessed
REVIEW_PROCESSED_FILE = PROCESSED_DATA_DIR / "processed_review.jsonl"
REVIEW_TRANSFORMER_PROCESSED_FILE = PROCESSED_DATA_DIR / "yelp_reviews_100000_transformer_processed.jsonl"

# Sentiment
REVIEW_SENTIMENT_FILE = PROCESSED_DATA_DIR / "review_labeled.jsonl"
REVIEW_SENTIMENT_FILE_PROCESSED = PROCESSED_DATA_DIR / "processed_review_sentiment.jsonl"

# =========================
# MODEL
# =========================
MODEL_DIR = ROOT / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
VECTORIZER_PATH = MODEL_DIR / "logistic_regression_vectorizer.joblib"

# =========================
# TRACE / DEBUG
# =========================
TRACE_DIR = ROOT / "trace"

TRACE_FILE = TRACE_DIR / "processed_review.txt"
TRANSFORMER_PREP_TRACE_FILE = TRACE_DIR / "yelp_reviews_100000_processed.txt"
TRANSFORMER_NON_ENGLISH_FILE = TRACE_DIR / "yelp_reviews_100000_non_english.jsonl"
CHECK_DATA_FILE = TRACE_DIR / "check_dataset_sample.jsonl"

# =========================
# RUNS & RESULTS
# =========================
RUNS_DIR = ROOT / "run"
RESULTS_DIR = ROOT / "results"

# =========================
# QUERIES
# =========================
QUERY_TEXT_FILE = ROOT / "queries.txt"

# =========================
# QRELS
# =========================
RUN_FILE = RUNS_DIR / "run_tfidf.txt"
QRELS_FILE = RESULTS_DIR / "qrels.txt"

# =========================
# OTHER
# =========================
MAX_DOCS = 100000
SAMPLE_SIZE = 100000

# =========================
# ANALYSIS
# =========================
ANALYSIS_DIR = TRACE_DIR

TOP_TERMS_FILE = ANALYSIS_DIR / "top_terms_analysis.txt"

# =========================
# TF-IDF INDEX DIR
# =========================
TFIDF_INDEX_DIR = DATA_DIR / "index"
TFIDF_TRACE_DIR = TRACE_DIR / "index_tfidf"
TFIDF_SEARCH_TRACE_DIR = TRACE_DIR / "search_tfidf" 
RUNS_SEARCH_TFIDF_DIR = RUNS_DIR / "runs_search_tfidf"


def get_tfidf_paths(ngram, sublinear):
    name = f"tfidf_{ngram[0]}_{ngram[1]}_{'sub' if sublinear else 'nosub'}"

    return {
        "vectorizer": TFIDF_INDEX_DIR / f"{name}_vectorizer.pkl",
        "matrix": TFIDF_INDEX_DIR / f"{name}_matrix.pkl",
        "ids": TFIDF_INDEX_DIR / f"{name}_doc_ids.pkl",
        "trace": TFIDF_TRACE_DIR / f"{name}.txt"
    }

def get_tfidf_run_paths(ngram, sublinear):
    name = f"tfidf_{ngram[0]}_{ngram[1]}_{'sub' if sublinear else 'nosub'}"

    return {
        "run": RUNS_SEARCH_TFIDF_DIR / f"{name}.txt",
        "trace": TFIDF_SEARCH_TRACE_DIR / f"search_{name}.txt"
    }
# =========================
# BM25 INDEX
# =========================
BM25_INDEX_DIR = DATA_DIR / "index"
BM25_TRACE_DIR = TRACE_DIR / "index_bm25"


def get_bm25_paths(k1, b):
    name = f"bm25_{k1}_{b}"

    return {
        "model": BM25_INDEX_DIR / f"{name}.pkl",
        "ids": BM25_INDEX_DIR / f"{name}_doc_ids.pkl",
        "trace": BM25_TRACE_DIR / f"{name}.txt"
    }
# =========================
# BM25 SEARCH TRACE
# =========================
BM25_SEARCH_TRACE_DIR = TRACE_DIR / "search_bm25"
RUNS_SEARCH_BM25_DIR = RUNS_DIR / "runs_search_bm25"

def get_bm25_run_paths(k1, b):
    name = f"bm25_{k1}_{b}"

    return {
        "run": RUNS_SEARCH_BM25_DIR / f"{name}.txt",
        "trace": BM25_SEARCH_TRACE_DIR / f"search_{name}.txt"
    }

# =========================
# TRANSFORMER INDEX
# =========================
# Store transformer index files in the project's data/index folder
# This aligns transformer index storage with other index types (TFIDF/BM25)
TRANSFORMER_INDEX_DIR = DATA_DIR / "index"
TRANSFORMER_TRACE_DIR = TRACE_DIR / "index_transformer"
TRANSFORMER_SEARCH_TRACE_DIR = TRACE_DIR / "search_transformer"
RUNS_SEARCH_TRANSFORMER_DIR = RUNS_DIR / "runs_search_transformer"


def get_transformer_paths(model_name="all-MiniLM-L6-v2"):
    safe_name = model_name.replace("/", "_").replace("-", "_")
    name = f"transformer_{safe_name}"

    return {
        "embeddings": TRANSFORMER_INDEX_DIR / f"{name}_embeddings.npy",
        "ids": TRANSFORMER_INDEX_DIR / f"{name}_doc_ids.pkl",
        "documents": TRANSFORMER_INDEX_DIR / f"{name}_documents.pkl",
        "meta": TRANSFORMER_INDEX_DIR / f"{name}_meta.pkl",
        "trace": TRANSFORMER_TRACE_DIR / f"{name}.txt"
    }


def get_transformer_run_paths(model_name="all-MiniLM-L6-v2"):
    safe_name = model_name.replace("/", "_").replace("-", "_")
    name = f"transformer_{safe_name}"

    return {
        "run": RUNS_SEARCH_TRANSFORMER_DIR / f"{name}.txt",
        "trace": TRANSFORMER_SEARCH_TRACE_DIR / f"search_{name}.txt"
    }

# =========================
# LOG
# =========================
LOG_FILE = RESULTS_DIR / "qrels_log.txt"

# =========================
# QRELS
# =========================
QRELS_KEYWORD = RESULTS_DIR / "qrels_keyword.txt"
QRELS_COUNT   = RESULTS_DIR / "qrels_count.txt"
QRELS_RATIO   = RESULTS_DIR / "qrels_ratio.txt"
LOG_FILE      = RESULTS_DIR / "qrels_log.txt"

# =========================
# RRF RUNS
# =========================
RUNS_RRF_DIR = RUNS_DIR / "runs_rrf"
