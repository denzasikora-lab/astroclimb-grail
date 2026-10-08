#!/usr/bin/env python3
"""Create a resumable, label-free OCR cache with the Kaggle Tesseract binary.

This is a cloud-compatible substitute for ``ocr_macos.swift``.  It reads only
the JPEG object cache, never train/test labels, and emits the JSONL schema
consumed by ``astroclimb.ocr.load_ocr_jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--language", default="eng")
    parser.add_argument("--psm", type=int, default=11, help="Tesseract page segmentation mode")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--limit", type=int, default=0, help="Limit newly processed images; 0 is all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def completed_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                digest = str(json.loads(line)["hash"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid OCR JSONL at line {line_number}: {path}") from error
            if digest in completed:
                raise ValueError(f"Duplicate OCR hash {digest} at line {line_number}: {path}")
            completed.add(digest)
    return completed


def recognize(
    image_path: Path,
    *,
    language: str,
    psm: int,
    timeout_seconds: int,
) -> dict[str, object]:
    started = time.monotonic()
    record: dict[str, object] = {"hash": image_path.stem, "text": ""}
    try:
        completed = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", language, "--psm", str(psm)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        record["text"] = " ".join(completed.stdout.split())
        record["returncode"] = completed.returncode
        if completed.returncode:
            record["error"] = " ".join(completed.stderr.split())[:400]
    except subprocess.TimeoutExpired:
        record["error"] = f"timeout after {timeout_seconds}s"
    record["elapsed_milliseconds"] = round((time.monotonic() - started) * 1000)
    return record


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.psm <= 0:
        raise ValueError("--psm must be positive")
    if shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract is not installed in this environment")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {args.image_dir}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = completed_hashes(args.output_jsonl)
    pending = [
        path
        for path in sorted(args.image_dir.glob("*.jpg"))
        if path.stem not in completed
    ]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"OCR is current: {args.output_jsonl}")
        return
    started = time.monotonic()
    failures = 0
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = (
                pool.submit(
                    recognize,
                    path,
                    language=args.language,
                    psm=args.psm,
                    timeout_seconds=args.timeout_seconds,
                )
                for path in pending
            )
            for index, job in enumerate(jobs, start=1):
                record = job.result()
                failures += int(bool(record.get("error")))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if index % 100 == 0 or index == len(pending):
                    print(
                        f"ocr {index:,}/{len(pending):,}; failures={failures}; "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    print(f"Wrote {args.output_jsonl}; new={len(pending)}; failures={failures}")


if __name__ == "__main__":
    main()
