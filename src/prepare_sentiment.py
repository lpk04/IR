import json
import re
from tqdm import tqdm
from config import REVIEW_RAW_FILE, REVIEW_SENTIMENT_FILE

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def map_label(rating):
    if rating >= 4:
        return "pos"
    elif rating == 3:
        return "neu"
    else:
        return "neg"

def process():
    with open(REVIEW_RAW_FILE, "r", encoding="utf-8") as f_in, \
         open(REVIEW_SENTIMENT_FILE, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in):
            data = json.loads(line)

            text = clean_text(data["text"])
            label = map_label(data["rating"])

            new_data = {
                "doc_id": data["doc_id"],
                "text": text,
                "label": label
            }

            f_out.write(json.dumps(new_data) + "\n")

if __name__ == "__main__":
    process()