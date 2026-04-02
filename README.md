-setup file python setup_project.py
-config python config.py
-tạo mt python -m venv venv
-bat mt venv\Scripts\activate
-cài nltk 
bash--> python--> 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
-->exit() 
--cài thư viện pip install -r requirements.txt
-tải dl
-chuẩn bị dl prepare_data.py 

-index_tfidf.py 
python src/index_tfidf.py --ngram 11 --sublinear false
python src/index_tfidf.py --ngram 11 --sublinear true
python src/index_tfidf.py --ngram 12 --sublinear false
python src/index_tfidf.py --ngram 12 --sublinear true


-search_tfidf.py
python src/search_tfidf.py --ngram 11 --sublinear false
python src/search_tfidf.py --ngram 11 --sublinear true
python src/search_tfidf.py --ngram 12 --sublinear false
python src/search_tfidf.py --ngram 12 --sublinear true


-index_bm25
python src/index_bm25.py --k1 1.2 --b 0.75
python src/index_bm25.py --k1 1.5 --b 0.75
python src/index_bm25.py --k1 2.0 --b 0.75

-search_bm25
python src/search_bm25.py --k1 1.2 --b 0.75
python src/search_bm25.py --k1 1.5 --b 0.75
python src/search_bm25.py --k1 2.0 --b 0.75

-generate_queries.py


git add .
git commit -m "update"
git push


# 1. Generate qrels
python src/generate_qrels.py

# 2. Search TF-IDF
python src/search_tfidf.py

# 3. Search BM25 (nhiều config)
python src/search_bm25.py 1.2 0.75
python src/search_bm25.py 1.5 0.75
python src/search_bm25.py 2.0 0.75

# 4. Evaluate
python src/evaluate_results.py