import json
import re


# =========================
# 1. Expand contractions (giữ negation)
# =========================
def expand_contractions(text):
    text = text.lower()
    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")
    text = text.replace("n't", " not")
    return text


# =========================
# 2. Clean text (chuẩn baseline)
# =========================
def clean_text(text):
    text = expand_contractions(text)

    # giữ chữ, số và dấu '
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # normalize kéo dài chữ: sooo → so
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # xóa khoảng trắng dư
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# 3. Convert rating → label
# =========================
def rating_to_label(r, mode="binary"):
    if mode == "binary":
        if r <= 2:
            return "negative"
        elif r >= 4:
            return "positive"
        else:
            return None   # bỏ neutral
    else:
        if r <= 2:
            return "negative"
        elif r == 3:
            return "neutral"
        else:
            return "positive"


# =========================
# 4. Process file
# =========================
def process_file(input_path, output_path, mode="binary"):
    count_in = 0
    count_out = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            count_in += 1
            data = json.loads(line)

            text = data.get("text", "")
            rating = data.get("rating", 0)

            label = rating_to_label(rating, mode)

            # bỏ neutral nếu binary
            if label is None:
                continue

            cleaned_text = clean_text(text)

            new_data = {
                "doc_id": data.get("doc_id"),
                "text": cleaned_text,
                "label": label
            }

            fout.write(json.dumps(new_data) + "\n")
            count_out += 1

    print(f"Input: {count_in}")
    print(f"Output: {count_out}")


# =========================
# 5. MAIN
# =========================
if __name__ == "__main__":
    INPUT_FILE = r"D:\IR\demo\data\review.jsonl"
    OUTPUT_FILE = r"D:\IR\demo\data\processed\review_labeled.jsonl"

    # mode = "binary" (khuyên dùng) hoặc "multiclass"
    process_file(INPUT_FILE, OUTPUT_FILE, mode="binary")