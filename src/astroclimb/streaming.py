from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

import pandas as pd

from .constants import IMAGE_PREFIXES, LABELS
from .integrity import assert_no_test_targets, assert_safe_input


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def is_image_object(value: str) -> bool:
    return value.startswith(IMAGE_PREFIXES)


def object_digest(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def canonical_pair_type(type_1: str, type_2: str) -> str:
    return "+".join(sorted((type_1, type_2)))


@dataclass(frozen=True)
class PairRow:
    pair_id: int
    object_1: str
    object_2: str
    label: str | None


def iter_pair_rows(
    path: str | Path, *, labeled: bool, max_field_characters: int | None = None
) -> Iterator[PairRow]:
    """Stream pair rows with an optional field-size limit.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV containing paired objects.
    labeled : bool
        Whether exactly one active training label is required.
    max_field_characters : int or None
        Maximum field length to bound memory use for a damaged CSV.

    Returns
    -------
    Iterator[PairRow]
        Validated rows containing both object fields.
    """
    source = assert_safe_input(path)
    configure_csv_field_limit()
    if max_field_characters is not None:
        if max_field_characters < 1:
            raise ValueError("The field-size limit must be positive")
        csv.field_size_limit(max_field_characters)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {source}")
        required = {"id", "obj_1", "obj_2"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing columns in {source.name}: {missing}")
        if labeled:
            missing_labels = sorted(set(LABELS).difference(reader.fieldnames))
            if missing_labels:
                raise ValueError(f"Missing labels in {source.name}: {missing_labels}")
        else:
            assert_no_test_targets(reader.fieldnames)

        for raw in reader:
            if raw.get("obj_1") is None or raw.get("obj_2") is None:
                raise ValueError(f"Incomplete row id={raw.get('id')}: missing paired object")
            label: str | None = None
            if labeled:
                active = [name for name in LABELS if int(raw[name]) == 1]
                if len(active) != 1:
                    raise ValueError(
                        f"Row id={raw['id']} has {len(active)} active labels; expected exactly one"
                    )
                label = active[0]
            yield PairRow(
                pair_id=int(raw["id"]),
                object_1=raw["obj_1"],
                object_2=raw["obj_2"],
                label=label,
            )


def scan_pair_file(
    path: str | Path,
    *,
    labeled: bool,
    progress_every: int = 500,
    progress: Callable[[str], None] | None = print,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    records: list[dict[str, object]] = []
    labels: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    started = time.monotonic()

    for index, row in enumerate(iter_pair_rows(path, labeled=labeled), start=1):
        type_1 = "image" if is_image_object(row.object_1) else "text"
        type_2 = "image" if is_image_object(row.object_2) else "text"
        pair_type = canonical_pair_type(type_1, type_2)
        hash_1 = object_digest(row.object_1)
        hash_2 = object_digest(row.object_2)
        records.append(
            {
                "id": row.pair_id,
                "type_1": type_1,
                "type_2": type_2,
                "pair_type": pair_type,
                "hash_1": hash_1,
                "hash_2": hash_2,
                "length_1": len(row.object_1),
                "length_2": len(row.object_2),
                "label": row.label,
            }
        )
        modalities[pair_type] += 1
        if row.label is not None:
            labels[row.label] += 1
        if progress is not None and index % progress_every == 0:
            elapsed = time.monotonic() - started
            progress(f"{Path(path).name}: {index:,} rows in {elapsed:.1f}s")

    frame = pd.DataFrame.from_records(records)
    objects = pd.concat([frame["hash_1"], frame["hash_2"]], ignore_index=True)
    repeats = objects.value_counts().value_counts().sort_index()
    summary: dict[str, object] = {
        "path": str(Path(path).resolve()),
        "rows": int(len(frame)),
        "unique_objects": int(objects.nunique()),
        "label_counts": dict(labels),
        "pair_type_counts": dict(modalities),
        "object_occurrence_histogram": {str(int(k)): int(v) for k, v in repeats.items()},
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return frame, summary


def build_audit_report(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    train_summary: Mapping[str, object],
    test_summary: Mapping[str, object],
    data_dir: str | Path,
) -> dict[str, object]:
    from .splits import object_component_groups

    def pair_duplicates(meta: pd.DataFrame) -> int:
        keys = [
            (left, right) if left <= right else (right, left)
            for left, right in meta.loc[:, ["hash_1", "hash_2"]].itertuples(index=False)
        ]
        return int(pd.Series(keys, dtype=object).duplicated().sum())

    train_objects = set(train_meta["hash_1"]).union(train_meta["hash_2"])
    test_objects = set(test_meta["hash_1"]).union(test_meta["hash_2"])
    train_groups = pd.Series(object_component_groups(train_meta)).value_counts()
    test_groups = pd.Series(object_component_groups(test_meta)).value_counts()
    disk = shutil.disk_usage(Path(data_dir))
    cross = pd.crosstab(train_meta["pair_type"], train_meta["label"])
    return {
        "integrity_policy": {
            "solution_csv_read": False,
            "test_targets_used": False,
            "id_or_row_order_allowed_as_feature": False,
        },
        "train": dict(train_summary),
        "test": dict(test_summary),
        "pair_graph": {
            "train_unordered_duplicate_pairs": pair_duplicates(train_meta),
            "test_unordered_duplicate_pairs": pair_duplicates(test_meta),
            "train_components": int(len(train_groups)),
            "test_components": int(len(test_groups)),
            "train_largest_component_rows": int(train_groups.max()),
            "test_largest_component_rows": int(test_groups.max()),
        },
        "train_test_unique_object_overlap": len(train_objects.intersection(test_objects)),
        "modality_by_label": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in cross.to_dict(orient="index").items()
        },
        "disk": {
            "total_gib": round(disk.total / 2**30, 2),
            "used_gib": round(disk.used / 2**30, 2),
            "free_gib": round(disk.free / 2**30, 2),
        },
    }


def write_json(payload: Mapping[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
