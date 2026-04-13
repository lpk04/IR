import nltk

def setup_nltk():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger_eng"
    ]
    
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res)

setup_nltk()