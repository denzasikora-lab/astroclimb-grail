from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class UnionFind:
    parent: dict[Hashable, Hashable]
    rank: dict[Hashable, int]

    def __init__(self) -> None:
        self.parent = {}
        self.rank = {}

    def find(self, item: Hashable) -> Hashable:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: Hashable, right: Hashable) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def object_component_groups(meta: pd.DataFrame) -> np.ndarray:
    required = {"hash_1", "hash_2"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing object hash columns: {sorted(missing)}")
    components = UnionFind()
    for hash_1, hash_2 in meta.loc[:, ["hash_1", "hash_2"]].itertuples(index=False):
        components.union(hash_1, hash_2)
    roots = [components.find(hash_1) for hash_1 in meta["hash_1"]]
    root_to_group = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=np.int64)


def component_stratified_folds(
    meta: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 20260919,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if "label" not in meta.columns:
        raise ValueError("Metadata requires a label column")
    groups = object_component_groups(meta)
    group_count = len(np.unique(groups))
    if group_count < n_splits:
        raise ValueError(f"Only {group_count} object components are available for {n_splits} folds")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    dummy = np.zeros(len(meta), dtype=np.uint8)
    yield from splitter.split(dummy, meta["label"].astype(str).to_numpy(), groups)


def validate_disjoint_folds(
    meta: pd.DataFrame,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for fold, (train_index, valid_index) in enumerate(folds):
        train_rows = meta.iloc[np.asarray(train_index)]
        valid_rows = meta.iloc[np.asarray(valid_index)]
        train_objects = set(train_rows["hash_1"]).union(train_rows["hash_2"])
        valid_objects = set(valid_rows["hash_1"]).union(valid_rows["hash_2"])
        overlap = train_objects.intersection(valid_objects)
        if overlap:
            raise AssertionError(f"Fold {fold} leaks {len(overlap)} objects")
        reports.append(
            {
                "fold": fold,
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "valid_label_counts": dict(Counter(valid_rows["label"])),
                "object_overlap": 0,
            }
        )
    return reports
