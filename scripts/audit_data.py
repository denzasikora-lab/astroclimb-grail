#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astroclimb.streaming import build_audit_report, scan_pair_file, write_json


def input_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-scan AstroCLIMB without loading multi-GB CSVs into RAM."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "meta_train.parquet"
    test_output = output_dir / "meta_test.parquet"
    report_output = output_dir / "audit.json"
    input_files = {
        "train": input_identity(args.data_dir / "train.csv"),
        "test": input_identity(args.data_dir / "test.csv"),
    }
    if not args.force and all(path.exists() for path in (train_output, test_output, report_output)):
        existing = json.loads(report_output.read_text(encoding="utf-8"))
        if existing.get("input_files") == input_files:
            print(f"Audit is current at {report_output}. Use --force to rebuild.")
            return 0
        print("Input identity changed or is missing; rebuilding the audit.")

    train_meta, train_summary = scan_pair_file(
        args.data_dir / "train.csv",
        labeled=True,
        progress_every=args.progress_every,
    )
    train_temporary = train_output.with_suffix(".parquet.tmp")
    train_meta.to_parquet(train_temporary, index=False)
    train_temporary.replace(train_output)
    test_meta, test_summary = scan_pair_file(
        args.data_dir / "test.csv",
        labeled=False,
        progress_every=args.progress_every,
    )
    test_temporary = test_output.with_suffix(".parquet.tmp")
    test_meta.to_parquet(test_temporary, index=False)
    test_temporary.replace(test_output)
    report = build_audit_report(
        train_meta,
        test_meta,
        train_summary,
        test_summary,
        args.data_dir,
    )
    report["input_files"] = input_files
    write_json(report, report_output)
    print(f"Wrote {report_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
