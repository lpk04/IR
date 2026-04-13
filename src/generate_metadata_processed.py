from __future__ import annotations

import argparse
from pathlib import Path

try:
    from config import REVIEW_TRANSFORMER_PROCESSED_FILE, DATA_DIR
except ImportError:
    from .config import REVIEW_TRANSFORMER_PROCESSED_FILE, DATA_DIR

try:
    from generate_metadata import generate_metadata
except ImportError:
    from .generate_metadata import generate_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metadata for the transformer-processed Yelp reviews."
    )
    parser.add_argument("--input", type=Path, default=REVIEW_TRANSFORMER_PROCESSED_FILE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "metadata")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--id-fields", type=str, default="review_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_metadata(
        input_file=args.input,
        output_dir=args.output_dir,
        text_field=args.text_field,
        id_fields=args.id_fields,
    )
    print(f"Saved metadata -> {paths['metadata']}")
    print(f"Saved summary -> {paths['summary']}")


if __name__ == "__main__":
    main()
