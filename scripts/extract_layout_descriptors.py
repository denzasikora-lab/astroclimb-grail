#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from astroclimb.visual import image_layout_descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cheap scientific-figure layout features.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="layout")
    args = parser.parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("image_failures"):
        raise RuntimeError("Refusing an object cache with image decode failures")
    hashes = pd.read_parquet(args.cache_dir / "image_hashes.parquet")[["hash"]]
    if hashes["hash"].astype(str).duplicated().any():
        raise ValueError("image_hashes.parquet contains duplicate hashes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptors: list[np.ndarray] = []
    started = time.monotonic()
    for index, digest in enumerate(hashes["hash"].astype(str), start=1):
        with Image.open(args.cache_dir / "images" / f"{digest}.jpg") as image:
            descriptors.append(image_layout_descriptor(image))
        if index % 1000 == 0:
            print(f"layout {index:,}/{len(hashes):,}", flush=True)
    matrix = np.stack(descriptors).astype(np.float16)
    array_path = args.output_dir / f"emb_{args.name}_image.npy"
    array_tmp = args.output_dir / f"emb_{args.name}_image.npy.tmp"
    with array_tmp.open("wb") as handle:
        np.save(handle, matrix)
    array_tmp.replace(array_path)
    mapping_path = args.output_dir / f"hashes_{args.name}_image.parquet"
    mapping_tmp = args.output_dir / f"hashes_{args.name}_image.parquet.tmp"
    hashes.to_parquet(mapping_tmp, index=False)
    mapping_tmp.replace(mapping_path)
    provenance = {
        "name": args.name,
        "kind": "deterministic_layout_descriptor",
        "shape": list(matrix.shape),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    provenance_path = args.output_dir / f"provenance_{args.name}.json"
    provenance_tmp = args.output_dir / f"provenance_{args.name}.json.tmp"
    provenance_tmp.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    provenance_tmp.replace(provenance_path)
    print(provenance)


if __name__ == "__main__":
    main()
