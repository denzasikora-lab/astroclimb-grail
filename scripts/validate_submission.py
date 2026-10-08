#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from astroclimb.integrity import assert_safe_input, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AstroCLIMB submission offline.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    submission = pd.read_csv(assert_safe_input(args.submission))
    sample = pd.read_csv(assert_safe_input(args.sample))
    validate_submission(submission, sample)
    counts = submission.drop(columns="id").sum().to_dict()
    print(f"OK: {len(submission):,} rows, class counts={counts}")


if __name__ == "__main__":
    main()
