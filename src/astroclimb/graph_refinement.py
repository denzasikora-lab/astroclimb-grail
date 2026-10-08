from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .splits import UnionFind


def unordered_pair_keys(meta: pd.DataFrame) -> np.ndarray:
    left = meta["hash_1"].astype(str).to_numpy()
    right = meta["hash_2"].astype(str).to_numpy()
    return np.asarray(
        [f"{a}|{b}" if a <= b else f"{b}|{a}" for a, b in zip(left, right)],
        dtype=object,
    )


def average_duplicate_pairs(meta: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    """Give duplicate unordered pairs identical predictions without using labels."""
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(unordered_pair_keys(meta)):
        groups[str(key)].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        mean = probabilities[indices].mean(axis=0)
        probabilities[indices] = mean
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def conservative_same_paper_closure(
    meta: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    edge_threshold: float = 0.95,
    closure_floor: float = 0.75,
) -> np.ndarray:
    """Apply high-confidence same-paper transitivity to already-present pairs.

    This is intentionally conservative. It creates clusters only from edges whose
    combined same-figure/same-paper probability exceeds ``edge_threshold``. It
    never invents new test rows and never changes a pair outside those clusters.
    Thresholds must be chosen on OOF predictions, never on the leaderboard.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    if probabilities.shape != (len(meta), 4):
        raise ValueError("Expected one four-class probability row per metadata row")
    components = UnionFind()
    paper_probability = probabilities[:, 0] + probabilities[:, 1]
    for index, row in enumerate(meta.loc[:, ["hash_1", "hash_2"]].itertuples(index=False)):
        if paper_probability[index] >= edge_threshold:
            components.union(row.hash_1, row.hash_2)

    for index, row in enumerate(meta.loc[:, ["hash_1", "hash_2"]].itertuples(index=False)):
        if row.hash_1 not in components.parent or row.hash_2 not in components.parent:
            continue
        if components.find(row.hash_1) != components.find(row.hash_2):
            continue
        current = probabilities[index]
        target_paper_mass = max(float(current[0] + current[1]), closure_floor)
        if target_paper_mass <= current[0] + current[1] + 1e-12:
            continue
        same_figure_share = current[0] / max(current[0] + current[1], 1e-12)
        other_mass = 1.0 - target_paper_mass
        other_share = current[2:] / max(current[2:].sum(), 1e-12)
        probabilities[index, 0] = target_paper_mass * same_figure_share
        probabilities[index, 1] = target_paper_mass * (1.0 - same_figure_share)
        probabilities[index, 2:] = other_mass * other_share
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def greedy_same_figure_matching(
    meta: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.80,
) -> np.ndarray:
    """Enforce at most one high-confidence caption match per figure and vice versa."""
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    candidates = np.flatnonzero(
        (meta["pair_type"].to_numpy() == "image+text") & (probabilities[:, 0] >= threshold)
    )
    candidates = candidates[np.argsort(-probabilities[candidates, 0])]
    used_images: set[str] = set()
    used_texts: set[str] = set()
    selected: set[int] = set()
    for index in candidates:
        row = meta.iloc[index]
        image_hash = row["hash_1"] if row["type_1"] == "image" else row["hash_2"]
        text_hash = row["hash_1"] if row["type_1"] == "text" else row["hash_2"]
        if image_hash in used_images or text_hash in used_texts:
            continue
        selected.add(int(index))
        used_images.add(str(image_hash))
        used_texts.add(str(text_hash))

    for index in candidates:
        if int(index) in selected:
            continue
        row = meta.iloc[index]
        image_hash = str(row["hash_1"] if row["type_1"] == "image" else row["hash_2"])
        text_hash = str(row["hash_1"] if row["type_1"] == "text" else row["hash_2"])
        if image_hash not in used_images and text_hash not in used_texts:
            continue
        removed = probabilities[index, 0]
        probabilities[index, 0] = 0.0
        remainder = probabilities[index, 1:]
        probabilities[index, 1:] = remainder + removed * remainder / max(remainder.sum(), 1e-12)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
