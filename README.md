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

-index_bm25
python src/index_bm25.py --k1 1.2 --b 0.75
python src/index_bm25.py --k1 1.5 --b 0.75
python src/index_bm25.py --k1 2.0 --b 0.75
python src/index_bm25.py --k1 5.0 --b 0.75

-search_tfidf.py
python src/search_tfidf.py --ngram 11 --sublinear false
python src/search_tfidf.py --ngram 11 --sublinear true
python src/search_tfidf.py --ngram 12 --sublinear false
python src/search_tfidf.py --ngram 12 --sublinear true

-search_bm25
python src/search_bm25.py --k1 1.2 --b 0.75
python src/search_bm25.py --k1 1.5 --b 0.75
python src/search_bm25.py --k1 2.0 --b 0.75
python src/search_bm25.py --k1 5.0 --b 0.75

-generate_queries.py
điều kiện
+if common >= max(2, int(0.5 * len(q_tokens))):
    rel = 1
match số từ vd 3 từ phải match 2

+ratio = common / len(q_tokens)

if ratio >= 0.7:
    rel = 1
match theo %


-evaluate_results.py 


-search_bm25_sentiment.py --k1 1.2 --b 0.75
so sánh các mô hình khi thêm sentiment vào


python src/rerank_bm25_sentiment.py --k1 1.2 --b 0.75 --alpha 0.0
python src/rerank_bm25_sentiment.py --k1 1.2 --b 0.75 --alpha 0.2
python src/rerank_bm25_sentiment.py --k1 1.2 --b 0.75 --alpha 0.5
python src/rerank_bm25_sentiment.py --k1 1.2 --b 0.75 --alpha 1


git add .
git commit -m "update"
git push