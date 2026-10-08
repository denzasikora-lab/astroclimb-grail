#!/usr/bin/env python3
"""Prepare training objects only, validating rows against saved metadata."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np

from astroclimb.cache import assert_disk_budget, clean_text, decode_image_to_jpeg
from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.integrity import assert_safe_input
from astroclimb.streaming import (
    PairRow,
    canonical_pair_type,
    is_image_object,
    iter_pair_rows,
    object_digest,
    write_json,
)


def validated_rows(
    source: Path, metadata: pd.DataFrame, modalities: tuple[str, ...] | None = None
) -> Iterator[PairRow]:
    """Validate each row's objects, order, and labels before use.

    Parameters
    ----------
    source : pathlib.Path
        Training CSV to read one row at a time.
    metadata : pandas.DataFrame
        Metadata from a previously executed experiment.
    modalities : tuple[str, ...] or None
        Complete pair modalities for a separate comparison. None selects all rows.

    Returns
    -------
    Iterator[PairRow]
        Validated rows without loading the entire CSV into memory.
    """
    count = 0
    fields = (
        "id", "type_1", "type_2", "pair_type", "hash_1", "hash_2",
        "length_1", "length_2", "label",
    )
    bound = int(metadata[["length_1", "length_2"]].max().max()) + 1024
    selected = metadata if modalities is None else metadata[metadata.pair_type.isin(modalities)]
    if selected.empty or selected.id.duplicated().any():
        raise ValueError("The selected scope is empty or contains duplicate IDs")
    expected_by_id = selected.set_index("id")
    known_ids = set(metadata.id.astype(int))
    seen_ids = set()

    def rows() -> Iterator[PairRow]:
        if modalities is None:
            yield from iter_pair_rows(source, labeled=True, max_field_characters=bound)
            return
        csv.field_size_limit(bound)
        with assert_safe_input(source).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "obj_1", "obj_2", *LABELS}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("The CSV is missing required training columns")
            for raw in reader:
                pair_id = int(raw["id"])
                if pair_id not in known_ids:
                    raise ValueError(f"Unknown pair ID: {pair_id}")
                if pair_id not in expected_by_id.index:
                    continue
                if raw.get("obj_1") is None or raw.get("obj_2") is None:
                    raise ValueError(f"Incomplete selected row id={pair_id}")
                values = [raw[name] for name in LABELS]
                if any(value not in {"0", "1"} for value in values) or values.count("1") != 1:
                    raise ValueError(f"Invalid label in selected row id={pair_id}")
                yield PairRow(pair_id, raw["obj_1"], raw["obj_2"], LABELS[values.index("1")])

    for index, row in enumerate(rows()):
        if index >= len(metadata):
            raise ValueError("The CSV contains more rows than the saved dataset")
        if row.pair_id in seen_ids:
            raise ValueError(f"Duplicate selected row id={row.pair_id}")
        seen_ids.add(row.pair_id)
        kinds = ["image" if is_image_object(v) else "text" for v in (row.object_1, row.object_2)]
        actual = (
            row.pair_id, *kinds, canonical_pair_type(*kinds),
            object_digest(row.object_1), object_digest(row.object_2),
            len(row.object_1), len(row.object_2), row.label,
        )
        record = metadata.iloc[index] if modalities is None else expected_by_id.loc[row.pair_id]
        expected = tuple(row.pair_id if name == "id" else record[name] for name in fields)
        if modalities is None and row.pair_id != record["id"]:
            raise ValueError(f"Invalid row order: id={row.pair_id}")
        if actual != expected:
            raise ValueError(f"Row {index} does not match the saved experiment")
        count += 1
        yield row
    if count != len(selected):
        raise ValueError(f"The selected scope is incomplete: read {count} of {len(selected)} rows")


def prepare(
    source: Path,
    metadata_path: Path,
    root: Path,
    minimum_free_gib: float,
    modalities: tuple[str, ...] | None = None,
    fold_ids_path: Path | None = None,
) -> dict:
    """Build a compact cache only after validating the corresponding rows.

    Parameters
    ----------
    source : pathlib.Path
        Training CSV; the source file is neither copied nor modified.
    metadata_path : pathlib.Path
        Saved training metadata.
    root : pathlib.Path
        Project root for the new computations.
    minimum_free_gib : float
        Minimum free disk space checked during processing.
    modalities : tuple[str, ...] or None
        Complete pair modalities; None selects the original 10,000 pairs.
    fold_ids_path : pathlib.Path or None
        Saved full-dataset fold IDs to preserve the selected rows' assignments.

    Returns
    -------
    dict
        Completed-cache manifest confirming that every selected row matches.
    """
    cache = root / "artifacts/object_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("The cache already exists; automatic overwriting is prohibited")
    assert_disk_budget(cache, minimum_free_gib=minimum_free_gib)
    initial_size = source.stat().st_size
    metadata = pd.read_parquet(metadata_path)
    selected = metadata if modalities is None else metadata[metadata.pair_type.isin(modalities)]
    images = cache / "images"
    images.mkdir(exist_ok=True)
    write_json({"status": "building", "test_data_read": False}, manifest_path)
    seen, image_hashes, texts = set(), [], {}
    started, count = time.monotonic(), 0
    try:
        for count, row in enumerate(validated_rows(source, metadata, modalities), start=1):
            for value in (row.object_1, row.object_2):
                digest = object_digest(value)
                if digest in seen:
                    continue
                seen.add(digest)
                if is_image_object(value):
                    decode_image_to_jpeg(value, images / f"{digest}.jpg", side=384, quality=88)
                    image_hashes.append(digest)
                else:
                    texts[digest] = clean_text(value)
            if count % 100 == 0:
                assert_disk_budget(cache, minimum_free_gib=minimum_free_gib)
            if count % 500 == 0:
                print(f"Validated {count} pairs; images {len(image_hashes)}, captions {len(texts)}", flush=True)
        if source.stat().st_size != initial_size:
            raise ValueError("The source file size changed during reading")
        pd.DataFrame({"hash": list(texts), "text": list(texts.values())}).to_parquet(
            cache / "texts.parquet", index=False,
        )
        pd.DataFrame({"hash": sorted(image_hashes)}).to_parquet(
            cache / "image_hashes.parquet", index=False,
        )
        audit = root / "artifacts/audit"
        audit.mkdir(exist_ok=True)
        selected.to_parquet(audit / "meta_train.parquet", index=False)
        np.save(audit / "source_indices.npy", selected.index.to_numpy())
        if fold_ids_path is not None:
            full_ids = np.load(assert_safe_input(fold_ids_path))
            if full_ids.shape != (len(metadata),) or set(full_ids.tolist()) != set(range(5)):
                raise ValueError("Invalid fold assignments for the source dataset")
            np.save(audit / "fold_ids.npy", full_ids[selected.index.to_numpy()])
        manifest = {
            "status": "complete", "pairs": count, "unique_objects": len(seen),
            "texts": len(texts), "images": len(image_hashes), "image_failures": [],
            "image_side": 384, "jpeg_quality": 88, "saved_metadata_exact_match": True,
            "test_data_read": False, "source_bytes": initial_size,
            "elapsed_seconds": time.monotonic() - started,
            "source_pairs": len(metadata),
            "selected_pair_types": sorted(selected.pair_type.unique().tolist()),
            "scope_complete": True,
            "full_dataset_complete": len(selected) == len(metadata),
        }
        write_json(manifest, manifest_path)
        return manifest
    except Exception as error:
        write_json({"status": "incomplete", "pairs": count, "error": str(error)}, manifest_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=6.0)
    parser.add_argument("--modalities", nargs="+", choices=PAIR_TYPES)
    parser.add_argument("--fold-ids", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.train, args.metadata, args.project, args.minimum_free_gib,
            tuple(args.modalities) if args.modalities else None, args.fold_ids,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
