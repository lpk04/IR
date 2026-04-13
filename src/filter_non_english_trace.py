from __future__ import annotations

import json
from pathlib import Path

# Use the project's detection code to keep logic consistent
try:
    from prepare_data_transformer import detect_text_field, is_english
except ImportError:
    from .prepare_data_transformer import detect_text_field, is_english

try:
    from config import TRANSFORMER_NON_ENGLISH_FILE
except ImportError:
    from .config import TRANSFORMER_NON_ENGLISH_FILE


def main() -> None:
    inp = Path(TRANSFORMER_NON_ENGLISH_FILE)
    if not inp.exists():
        print(f"Input trace not found: {inp}")
        return

    out = inp.with_name(inp.stem + ".filtered.jsonl")
    backup = inp.with_suffix(".bak")

    total = kept = removed = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                fout.write(line + "\n")
                kept += 1
                continue

            text = detect_text_field(rec, preferred_field="text")
            if text and is_english(text):
                removed += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    # Replace original with filtered (keep backup)
    if backup.exists():
        backup.unlink()
    inp.replace(backup)
    out.replace(inp)

    print(f"Total lines: {total}, kept: {kept}, removed (English detected): {removed}")


if __name__ == "__main__":
    main()
