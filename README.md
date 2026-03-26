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
-chuẩn bị dl prepare.py 
-generate_queries.py

-index_tfidf.py 
-search_tfidf.py

python index_bm25.py 1.2 0.75
python index_bm25.py 1.5 0.75
python index_bm25.py 2.0 0.75

python search_bm25.py 1.2 0.75
python search_bm25.py 1.5 0.75
python search_bm25.py 2.0 0.75

-compare.py



