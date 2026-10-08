from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def center_and_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return l2_normalize(values - values.mean(axis=0, keepdims=True))


def symmetric_embedding_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"Expected matching 2D embeddings, received {left.shape} and {right.shape}"
        )
    difference = np.abs(left - right)
    product = left * right
    cosine = np.sum(l2_normalize(left) * l2_normalize(right), axis=1, keepdims=True)
    distance = np.linalg.norm(left - right, axis=1, keepdims=True)
    return np.concatenate([difference, product, cosine, distance], axis=1).astype(np.float32)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    return len(left.intersection(right)) / len(union) if union else 0.0


def _character_ngrams(value: str, width: int) -> set[str]:
    normalized = NON_ALNUM_RE.sub("", value.casefold())
    return {
        normalized[index : index + width] for index in range(max(0, len(normalized) - width + 1))
    }


def lexical_pair_features(left: str, right: str) -> np.ndarray:
    left_tokens = {token.casefold() for token in TOKEN_RE.findall(left)}
    right_tokens = {token.casefold() for token in TOKEN_RE.findall(right)}
    left_numbers = set(NUMBER_RE.findall(left))
    right_numbers = set(NUMBER_RE.findall(right))
    left_acronyms = set(ACRONYM_RE.findall(left))
    right_acronyms = set(ACRONYM_RE.findall(right))
    minimum = max(1, min(len(left), len(right)))
    maximum = max(len(left), len(right))
    return np.asarray(
        [
            math.log1p(minimum),
            math.log1p(maximum),
            maximum / minimum,
            _jaccard(left_tokens, right_tokens),
            _jaccard(left_numbers, right_numbers),
            _jaccard(left_acronyms, right_acronyms),
            len(left_numbers.intersection(right_numbers)),
            len(left_acronyms.intersection(right_acronyms)),
        ],
        dtype=np.float32,
    )


def lexical_feature_matrix(
    meta: pd.DataFrame,
    text_lookup: Mapping[str, str],
) -> np.ndarray:
    """Compute symmetric lexical overlap for text-text rows, zero elsewhere."""
    output = np.zeros((len(meta), 8), dtype=np.float32)
    both_text = (meta["type_1"].to_numpy() == "text") & (meta["type_2"].to_numpy() == "text")
    for position in np.flatnonzero(both_text):
        left_hash = str(meta.iloc[position]["hash_1"])
        right_hash = str(meta.iloc[position]["hash_2"])
        try:
            left = text_lookup[left_hash]
            right = text_lookup[right_hash]
        except KeyError as error:
            raise KeyError(f"Missing cached text for hash {error.args[0]}") from error
        output[position] = lexical_pair_features(left, right)
    return output


def multimodal_ocr_feature_matrix(
    meta: pd.DataFrame,
    text_lookup: Mapping[str, str],
    image_text_lookup: Mapping[str, str],
) -> np.ndarray:
    """Symmetric lexical features using captions for text and OCR for images."""
    output = np.zeros((len(meta), 12), dtype=np.float32)
    for position, row in enumerate(meta.itertuples(index=False)):
        left = (
            image_text_lookup.get(str(row.hash_1), "")
            if row.type_1 == "image"
            else text_lookup.get(str(row.hash_1), "")
        )
        right = (
            image_text_lookup.get(str(row.hash_2), "")
            if row.type_2 == "image"
            else text_lookup.get(str(row.hash_2), "")
        )
        base = lexical_pair_features(left, right)
        output[position, :8] = base
        output[position, 8] = _jaccard(_character_ngrams(left, 3), _character_ngrams(right, 3))
        output[position, 9] = _jaccard(_character_ngrams(left, 4), _character_ngrams(right, 4))
        output[position, 10] = float(bool(left) + bool(right))
        output[position, 11] = float(bool(left) and bool(right))
    return output


@dataclass(frozen=True)
class FactorizedProbabilities:
    same_figure: np.ndarray
    same_paper_given_not_figure: np.ndarray
    related_given_different_paper: np.ndarray

    def leaf_probabilities(self, *, pair_type: np.ndarray | None = None) -> np.ndarray:
        q_figure = np.asarray(self.same_figure, dtype=np.float64).copy()
        q_paper = np.asarray(self.same_paper_given_not_figure, dtype=np.float64)
        q_related = np.asarray(self.related_given_different_paper, dtype=np.float64)
        if pair_type is not None:
            q_figure[np.asarray(pair_type) != "image+text"] = 0.0
        q_figure = np.clip(q_figure, 0.0, 1.0)
        q_paper = np.clip(q_paper, 0.0, 1.0)
        q_related = np.clip(q_related, 0.0, 1.0)
        not_figure = 1.0 - q_figure
        different_paper = not_figure * (1.0 - q_paper)
        probabilities = np.column_stack(
            [
                q_figure,
                not_figure * q_paper,
                different_paper * q_related,
                different_paper * (1.0 - q_related),
            ]
        )
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def gather_embeddings(
    hashes: np.ndarray,
    lookup: Mapping[str, int],
    matrix: np.ndarray,
) -> np.ndarray:
    missing = [value for value in hashes if value not in lookup]
    if missing:
        raise KeyError(f"Missing {len(missing)} embeddings; first hash={missing[0]}")
    return np.asarray(matrix[[lookup[value] for value in hashes]], dtype=np.float32)
