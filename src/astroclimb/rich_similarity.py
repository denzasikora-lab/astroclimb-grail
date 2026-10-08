"""Small symmetric similarity features with strictly fold-local corpus statistics.

No labels or files are read here. ``fit`` accepts pair metadata and learns IDF and
embedding centering from unique objects occurring in those training pairs only.
Frozen pretrained embedding matrices may include other objects; those rows are
never used to fit corpus statistics. Call ``fit`` afresh for each outer fold.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .embedding_store import EncoderSpace
from .features import ACRONYM_RE, NUMBER_RE, l2_normalize

WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:[-+.][a-zA-Z0-9]+)*")
IDENTIFIER_RE = re.compile(
    r"\b(?=[A-Za-z0-9.+-]*[A-Za-z])(?=[A-Za-z0-9.+-]*\d)[A-Za-z0-9]+(?:[-+.][A-Za-z0-9]+)*\b"
)
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class _Document:
    text: str
    normalized: str
    tokens: frozenset[str]
    informative: frozenset[str]
    numbers: frozenset[str]
    acronyms: frozenset[str]
    identifiers: frozenset[str]


def _overlap(left: frozenset[str], right: frozenset[str]) -> list[float]:
    shared = len(left & right)
    small, large = sorted((len(left), len(right)))
    return [
        shared / max(len(left | right), 1),
        shared / max(small, 1),
        shared / max(large, 1),
        float(np.log1p(shared)),
    ]


class RichSimilarityFeatures:
    """Fold-fitted scalar features for nonlinear pair classifiers.

    ``fit(meta_train)`` returns self; ``transform(meta)`` returns finite float32
    values in the order listed by ``feature_names``. Only hash_1/hash_2 and
    type_1/type_2 are required. Missing OCR is empty text; missing caption or
    encoder entries fail loudly. Every feature is invariant to swapping sides.

    Sparse text transforms and embedding gathering are bounded by batch_size.
    max_text_chars bounds OCR noise and unusually long caption memory usage.
    """

    _EMBEDDING_NAMES = (
        "available_sides",
        "cosine",
        "unit_l1_mean",
        "unit_l2",
        "unit_linf",
        "unit_abs_q25",
        "unit_abs_q50",
        "unit_abs_q75",
        "unit_abs_q90",
        "product_std",
        "product_min",
        "product_max",
        "sign_agreement",
        "centered_cosine",
        "whitened_cosine",
        "norm_log_min",
        "norm_log_max",
    )
    _SPARSE_NAMES = (
        "cosine",
        "weighted_jaccard",
        "containment_min",
        "containment_max",
        "nnz_log_min",
        "nnz_log_max",
        "nonempty_sides",
    )

    def __init__(
        self,
        text_lookup: Mapping[str, str],
        ocr_lookup: Mapping[str, str],
        spaces: Iterable[EncoderSpace],
        *,
        max_features: int = 40_000,
        max_text_chars: int = 4_000,
        batch_size: int = 512,
    ) -> None:
        if min(max_features, max_text_chars, batch_size) <= 0:
            raise ValueError("Feature, text and batch limits must be positive")
        self.text_lookup = text_lookup
        self.ocr_lookup = ocr_lookup
        self.spaces = list(spaces)
        if len({space.name for space in self.spaces}) != len(self.spaces):
            raise ValueError("Encoder space names must be unique")
        self.max_features = max_features
        self.max_text_chars = max_text_chars
        self.batch_size = batch_size
        self._documents: dict[tuple[str, str], _Document] = {}
        self.feature_names = [
            "pair.image_image",
            "pair.image_text",
            "pair.text_text",
            "text.nonempty_sides",
            "text.length_log_min",
            "text.length_log_max",
            "text.length_ratio",
            "text.token_log_min",
            "text.token_log_max",
            "text.exact",
            "text.normalized_exact",
            "text.substring",
        ]
        for name in ("token", "informative", "number", "acronym", "identifier"):
            self.feature_names.extend(
                f"text.{name}_{stat}"
                for stat in ("jaccard", "containment_min", "containment_max", "shared_log")
            )
        self.feature_names.extend(
            f"text.idf_{stat}"
            for stat in (
                "jaccard",
                "containment_min",
                "containment_max",
                "shared_max",
                "shared_mean",
            )
        )
        for name in ("word", "char"):
            self.feature_names.extend(f"tfidf.{name}_{stat}" for stat in self._SPARSE_NAMES)
        self.feature_names.extend(
            f"raw.{stat}"
            for stat in (
                "length_log_min",
                "length_log_max",
                "length_ratio",
                "image_length_log_mean",
                "text_length_log_mean",
            )
        )
        for space in self.spaces:
            self.feature_names.extend(f"emb.{space.name}_{stat}" for stat in self._EMBEDDING_NAMES)
            if space.name == "layout":
                self.feature_names.extend(
                    f"layout.{stat}_{index}" for stat in ("absdiff", "mean") for index in range(33)
                )
        self._fitted = False

    @staticmethod
    def _keys(meta: pd.DataFrame, side: int) -> list[tuple[str, str]]:
        keys = list(zip(meta[f"type_{side}"].astype(str), meta[f"hash_{side}"].astype(str)))
        if any(kind not in {"image", "text"} for kind, _ in keys):
            raise ValueError("Object types must be image or text")
        return keys

    def _document(self, key: tuple[str, str]) -> _Document:
        if key not in self._documents:
            kind, digest = key
            value = self.text_lookup[digest] if kind == "text" else self.ocr_lookup.get(digest, "")
            value = str(value)[: self.max_text_chars].strip()
            tokens = frozenset(word.casefold() for word in WORD_RE.findall(value))
            self._documents[key] = _Document(
                value,
                NORMALIZE_RE.sub("", value.casefold()),
                tokens,
                frozenset(
                    word for word in tokens if len(word) >= 3 and word not in ENGLISH_STOP_WORDS
                ),
                frozenset(number.casefold() for number in NUMBER_RE.findall(value)),
                frozenset(ACRONYM_RE.findall(value)),
                frozenset(word.casefold() for word in IDENTIFIER_RE.findall(value)),
            )
        return self._documents[key]

    def fit(self, meta_train: pd.DataFrame) -> "RichSimilarityFeatures":
        self._fitted = False
        self.fit_object_keys_ = frozenset(self._keys(meta_train, 1) + self._keys(meta_train, 2))
        if not self.fit_object_keys_:
            raise ValueError("Cannot fit on empty pair metadata")
        documents = [self._document(key) for key in sorted(self.fit_object_keys_)]
        nonempty = [document.text for document in documents if document.text]
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(document.informative)
        self.token_idf_ = {
            word: float(np.log((1 + len(documents)) / (1 + count)) + 1)
            for word, count in frequencies.items()
        }
        self.unseen_idf_ = float(np.log(1 + len(documents)) + 1)
        self.vectorizers_: dict[str, TfidfVectorizer | None] = {}
        for name, options in (
            ("word", {"analyzer": "word", "ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b"}),
            ("char", {"analyzer": "char_wb", "ngram_range": (3, 4)}),
        ):
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                sublinear_tf=True,
                dtype=np.float32,
                **options,
            )
            try:
                vectorizer.fit(nonempty)
            except ValueError as error:
                if not nonempty or "empty vocabulary" in str(error):
                    self.vectorizers_[name] = None
                    continue
                raise
            self.vectorizers_[name] = vectorizer

        self.embedding_statistics_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for space in self.spaces:
            arrays = []
            for kind in ("image", "text"):
                matrix = space.image if kind == "image" else space.text
                lookup = space.image_lookup if kind == "image" else space.text_lookup
                hashes = sorted(
                    digest for object_kind, digest in self.fit_object_keys_ if object_kind == kind
                )
                if matrix is not None and hashes:
                    arrays.append(l2_normalize(matrix[[lookup[digest] for digest in hashes]]))
            if arrays:
                values = np.concatenate(arrays, axis=0)
                self.embedding_statistics_[space.name] = (
                    values.mean(axis=0),
                    np.maximum(values.std(axis=0), 0.01),
                )
        self._fitted = True
        return self

    def _lexical(self, left: _Document, right: _Document) -> list[float]:
        small, large = sorted((len(left.text), len(right.text)))
        token_small, token_large = sorted((len(left.tokens), len(right.tokens)))
        has_both = bool(left.normalized and right.normalized)
        values = [
            float(bool(left.text)) + float(bool(right.text)),
            float(np.log1p(small)),
            float(np.log1p(large)),
            small / max(large, 1),
            float(np.log1p(token_small)),
            float(np.log1p(token_large)),
            float(has_both and left.text == right.text),
            float(has_both and left.normalized == right.normalized),
            float(
                has_both
                and min(len(left.normalized), len(right.normalized)) >= 12
                and (left.normalized in right.normalized or right.normalized in left.normalized)
            ),
        ]
        for field in ("tokens", "informative", "numbers", "acronyms", "identifiers"):
            values.extend(_overlap(getattr(left, field), getattr(right, field)))
        left_idf = sum(self.token_idf_.get(token, self.unseen_idf_) for token in left.informative)
        right_idf = sum(self.token_idf_.get(token, self.unseen_idf_) for token in right.informative)
        shared = [
            self.token_idf_.get(token, self.unseen_idf_)
            for token in left.informative & right.informative
        ]
        total = sum(shared)
        values.extend(
            [
                total / max(left_idf + right_idf - total, 1),
                total / max(min(left_idf, right_idf), 1),
                total / max(left_idf, right_idf, 1),
                max(shared, default=0),
                total / max(len(shared), 1),
            ]
        )
        return values

    @staticmethod
    def _sparse_features(matrix, left_ids: np.ndarray, right_ids: np.ndarray) -> np.ndarray:
        left, right = matrix[left_ids], matrix[right_ids]
        dot = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        shared = np.asarray(left.minimum(right).sum(axis=1)).ravel()
        a = np.asarray(left.sum(axis=1)).ravel()
        b = np.asarray(right.sum(axis=1)).ravel()
        na, nb = np.diff(left.indptr), np.diff(right.indptr)
        return np.column_stack(
            [
                dot,
                shared / np.maximum(a + b - shared, 1e-8),
                shared / np.maximum(np.minimum(a, b), 1e-8),
                shared / np.maximum(np.maximum(a, b), 1e-8),
                np.log1p(np.minimum(na, nb)),
                np.log1p(np.maximum(na, nb)),
                (na > 0).astype(np.float32) + (nb > 0).astype(np.float32),
            ]
        )

    def _embedding_features(
        self,
        space: EncoderSpace,
        left_keys: list[tuple[str, str]],
        right_keys: list[tuple[str, str]],
    ) -> np.ndarray:
        values = np.zeros((len(left_keys), len(self._EMBEDDING_NAMES)), dtype=np.float32)
        matrix = space.image if space.image is not None else space.text
        if matrix is None:
            return values
        dimension = matrix.shape[1]
        sides = []
        available = []
        for keys in (left_keys, right_keys):
            side = np.zeros((len(keys), dimension), dtype=np.float32)
            present = np.zeros(len(keys), dtype=bool)
            for kind in ("image", "text"):
                embeddings = space.image if kind == "image" else space.text
                lookup = space.image_lookup if kind == "image" else space.text_lookup
                positions = [index for index, key in enumerate(keys) if key[0] == kind]
                if embeddings is not None and positions:
                    side[positions] = embeddings[[lookup[keys[index][1]] for index in positions]]
                    present[positions] = True
            sides.append(side)
            available.append(present)
        values[:, 0] = available[0].astype(np.float32) + available[1].astype(np.float32)
        both = available[0] & available[1]
        if not both.any():
            return values
        raw_left, raw_right = sides[0][both], sides[1][both]
        left, right = l2_normalize(raw_left), l2_normalize(raw_right)
        diff, product = np.abs(left - right), left * right
        mean, std = self.embedding_statistics_.get(
            space.name,
            (np.zeros(dimension, dtype=np.float32), np.ones(dimension, dtype=np.float32)),
        )
        centered_left, centered_right = left - mean, right - mean
        norm_left = np.linalg.norm(raw_left, axis=1)
        norm_right = np.linalg.norm(raw_right, axis=1)
        values[both, 1:] = np.column_stack(
            [
                product.sum(axis=1),
                diff.mean(axis=1),
                np.linalg.norm(diff, axis=1),
                diff.max(axis=1),
                np.quantile(diff, [0.25, 0.5, 0.75, 0.9], axis=1).T,
                product.std(axis=1),
                product.min(axis=1),
                product.max(axis=1),
                ((left >= 0) == (right >= 0)).mean(axis=1),
                (l2_normalize(centered_left) * l2_normalize(centered_right)).sum(axis=1),
                (l2_normalize(centered_left / std) * l2_normalize(centered_right / std)).sum(
                    axis=1
                ),
                np.log1p(np.minimum(norm_left, norm_right)),
                np.log1p(np.maximum(norm_left, norm_right)),
            ]
        )
        return values

    @staticmethod
    def _raw_lengths(meta: pd.DataFrame) -> np.ndarray:
        values = np.zeros((len(meta), 5), dtype=np.float32)
        if not {"length_1", "length_2"}.issubset(meta.columns):
            return values
        left = np.maximum(meta["length_1"].to_numpy(dtype=np.float32), 0)
        right = np.maximum(meta["length_2"].to_numpy(dtype=np.float32), 0)
        small, large = np.minimum(left, right), np.maximum(left, right)
        values[:, 0] = np.log1p(small)
        values[:, 1] = np.log1p(large)
        values[:, 2] = large / np.maximum(small, 1)
        for column, kind in ((3, "image"), (4, "text")):
            is_left = (meta["type_1"].to_numpy() == kind).astype(np.float32)
            is_right = (meta["type_2"].to_numpy() == kind).astype(np.float32)
            count = is_left + is_right
            values[:, column] = np.log1p((left * is_left + right * is_right) / np.maximum(count, 1))
        return values

    @staticmethod
    def _layout_coordinates(
        space: EncoderSpace,
        left_keys: list[tuple[str, str]],
        right_keys: list[tuple[str, str]],
    ) -> np.ndarray:
        """Preserve interpretable 33-D layout coordinates; missing sides are zero."""
        sides = []
        for keys in (left_keys, right_keys):
            values = np.zeros((len(keys), 33), dtype=np.float32)
            for kind in ("image", "text"):
                matrix = space.image if kind == "image" else space.text
                lookup = space.image_lookup if kind == "image" else space.text_lookup
                positions = [index for index, key in enumerate(keys) if key[0] == kind]
                if matrix is not None and positions:
                    if matrix.shape[1] != 33:
                        raise ValueError("The layout encoder must have exactly 33 coordinates")
                    values[positions] = matrix[[lookup[keys[index][1]] for index in positions]]
            sides.append(values)
        return np.concatenate([np.abs(sides[0] - sides[1]), (sides[0] + sides[1]) * 0.5], axis=1)

    def transform(self, meta: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(meta_train) before transform")
        output = np.empty((len(meta), len(self.feature_names)), dtype=np.float32)
        for start in range(0, len(meta), self.batch_size):
            batch = meta.iloc[start : start + self.batch_size]
            left_keys, right_keys = self._keys(batch, 1), self._keys(batch, 2)
            keys = sorted(set(left_keys + right_keys))
            indices = {key: index for index, key in enumerate(keys)}
            documents = [self._document(key) for key in keys]
            left_ids = np.asarray([indices[key] for key in left_keys])
            right_ids = np.asarray([indices[key] for key in right_keys])
            pairs = [
                tuple(sorted((left[0], right[0]))) for left, right in zip(left_keys, right_keys)
            ]
            modality = np.asarray(
                [
                    [
                        pair == ("image", "image"),
                        pair == ("image", "text"),
                        pair == ("text", "text"),
                    ]
                    for pair in pairs
                ],
                dtype=np.float32,
            )
            lexical = np.asarray(
                [
                    self._lexical(documents[left], documents[right])
                    for left, right in zip(left_ids, right_ids)
                ],
                dtype=np.float32,
            )
            blocks = [modality, lexical]
            for vectorizer in self.vectorizers_.values():
                if vectorizer is None:
                    blocks.append(np.zeros((len(batch), len(self._SPARSE_NAMES)), dtype=np.float32))
                else:
                    matrix = vectorizer.transform([document.text for document in documents])
                    blocks.append(self._sparse_features(matrix, left_ids, right_ids))
            blocks.append(self._raw_lengths(batch))
            for space in self.spaces:
                blocks.append(self._embedding_features(space, left_keys, right_keys))
                if space.name == "layout":
                    blocks.append(self._layout_coordinates(space, left_keys, right_keys))
            output[start : start + len(batch)] = np.concatenate(blocks, axis=1)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite similarity features; inspect source embeddings")
        return output

    def fit_transform(self, meta_train: pd.DataFrame) -> np.ndarray:
        return self.fit(meta_train).transform(meta_train)
