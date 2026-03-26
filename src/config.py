from pathlib import Path

# =========================
# ROOT
# =========================
ROOT = Path("D:/IR/demo")

# =========================
# DATA
# =========================
DATA_DIR = ROOT / "data"

RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# =========================
# DATA FILES
# =========================
YELP_REVIEW_FILE = RAW_DATA_DIR / "yelp_academic_dataset_review.json"

# Extract
REVIEW_RAW_FILE = PROCESSED_DATA_DIR / "review.jsonl"

# Preprocessed
REVIEW_PROCESSED_FILE = PROCESSED_DATA_DIR / "processed_review.jsonl"  

# =========================
# TRACE
# =========================
TRACE_DIR = ROOT / "trace"
TRACE_FILE = TRACE_DIR / "processed_review_log.txt"

# =========================
# OTHER
# =========================
MAX_DOCS = 100000

# =========================
# ANALYSIS
# =========================
ANALYSIS_DIR = ROOT / "trace"

ANALYSIS_RESULT_FILE = ANALYSIS_DIR / "data_analysis_result.txt"

# =========================
# RUNS & RESULTS
# =========================
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

# =========================
# QUERIES
# =========================
QUERY_DIR = ROOT / "queries_Result"
QUERY_TEXT_FILE = ROOT / "queries.txt"

# =========================
# QRELS
# =========================
RUN_FILE = RUNS_DIR / "run_tfidf.txt"  
QRELS_FILE = RESULTS_DIR / "qrels.txt"