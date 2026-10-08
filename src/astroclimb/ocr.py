from __future__ import annotations

import json
from pathlib import Path

from .integrity import assert_safe_input


def load_ocr_jsonl(path: str | Path) -> dict[str, str]:
    source = assert_safe_input(path)
    records: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                digest = str(row["hash"])
                text = str(row["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid OCR JSONL at line {line_number}") from error
            if digest in records:
                raise ValueError(f"Duplicate OCR hash {digest} at line {line_number}")
            records[digest] = text
    if not records:
        raise ValueError(f"OCR cache is empty: {source}")
    return records
