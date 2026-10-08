from __future__ import annotations

import base64
import html
import json
import re
import shutil
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from .streaming import is_image_object, iter_pair_rows, object_digest

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(value))
    return WHITESPACE_RE.sub(" ", value).strip()


def assert_disk_budget(path: str | Path, *, minimum_free_gib: float) -> None:
    free_gib = shutil.disk_usage(Path(path)).free / 2**30
    if free_gib < minimum_free_gib:
        raise OSError(
            f"Only {free_gib:.2f} GiB is free at {Path(path).resolve()}; "
            f"at least {minimum_free_gib:.2f} GiB is required by the configured safety margin"
        )


def decode_image_to_jpeg(encoded: str, destination: Path, *, side: int, quality: int) -> None:
    raw = base64.b64decode(encoded)
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        image.thumbnail((side, side), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=quality, optimize=True)


def build_object_cache(
    pair_files: Iterable[tuple[str | Path, bool]],
    output_dir: str | Path,
    *,
    side: int = 384,
    quality: int = 88,
    minimum_free_gib: float = 8.0,
    progress_every: int = 500,
    allow_image_failures: bool = False,
) -> dict[str, object]:
    """Deduplicate objects and materialize a compact cache for embedding extraction."""
    output = Path(output_dir).resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    assert_disk_budget(output, minimum_free_gib=minimum_free_gib)

    seen: set[str] = set()
    texts: dict[str, str] = {}
    images: list[str] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    pair_count = 0

    for pair_file, labeled in pair_files:
        for row in iter_pair_rows(pair_file, labeled=labeled):
            pair_count += 1
            for value in (row.object_1, row.object_2):
                digest = object_digest(value)
                if digest in seen:
                    continue
                seen.add(digest)
                if is_image_object(value):
                    try:
                        decode_image_to_jpeg(
                            value,
                            image_dir / f"{digest}.jpg",
                            side=side,
                            quality=quality,
                        )
                        images.append(digest)
                    except Exception as error:  # keep the scan running and report exact failures
                        failures.append({"hash": digest, "error": repr(error)[:300]})
                else:
                    texts[digest] = clean_text(value)
            if pair_count % progress_every == 0:
                assert_disk_budget(output, minimum_free_gib=minimum_free_gib)
                print(
                    f"cached {pair_count:,} pairs, {len(images):,} images, "
                    f"{len(texts):,} texts in {time.monotonic() - started:.1f}s",
                    flush=True,
                )

    pd.DataFrame({"hash": list(texts), "text": list(texts.values())}).to_parquet(
        output / "texts.parquet", index=False
    )
    pd.DataFrame({"hash": sorted(images)}).to_parquet(output / "image_hashes.parquet", index=False)
    manifest = {
        "status": "complete" if not failures else "incomplete",
        "pairs": pair_count,
        "unique_objects": len(seen),
        "texts": len(texts),
        "images": len(images),
        "image_failures": failures,
        "image_side": side,
        "jpeg_quality": quality,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    if failures and not allow_image_failures:
        raise RuntimeError(
            f"Failed to decode {len(failures)} image objects; see {manifest_path}. "
            "The incomplete cache is not safe for training."
        )
    return manifest
