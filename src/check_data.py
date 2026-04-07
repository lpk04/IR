import random
from tqdm import tqdm

from config import YELP_REVIEW_FILE, CHECK_DATA_FILE, SAMPLE_SIZE


def sample_dataset():
    lines = []

    with open(YELP_REVIEW_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            lines.append(line)

    k = min(SAMPLE_SIZE, len(lines))
    sampled = random.sample(lines, k)

    with open(CHECK_DATA_FILE, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line)

if __name__ == "__main__":
    sample_dataset()