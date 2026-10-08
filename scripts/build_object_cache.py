#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from astroclimb.cache import build_object_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deduplicated image/text object cache.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", type=int, default=384)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument(
        "--allow-image-failures",
        action="store_true",
        help="Create an explicitly incomplete cache instead of failing on decode errors",
    )
    args = parser.parse_args()
    manifest = build_object_cache(
        [
            (args.data_dir / "train.csv", True),
            (args.data_dir / "test.csv", False),
        ],
        args.output_dir,
        side=args.side,
        quality=args.quality,
        minimum_free_gib=args.minimum_free_gib,
        allow_image_failures=args.allow_image_failures,
    )
    print(manifest)


if __name__ == "__main__":
    main()
