from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import LABELS
from .features import FactorizedProbabilities


def default_binary_estimator() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=20260919,
        ),
    )


def catboost_binary_estimator() -> object:
    """A nonlinear CPU/GPU-friendly head, imported lazily for the optional ML extra."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Install the optional ML dependencies with `pip install -e '.[ml]'`"
        ) from error
    return CatBoostClassifier(
        iterations=650,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=20260919,
        verbose=False,
        allow_writing_files=False,
    )


def estimator_factory(name: str) -> Callable[[], object]:
    factories: dict[str, Callable[[], object]] = {
        "logistic": default_binary_estimator,
        "catboost": catboost_binary_estimator,
    }
    try:
        return factories[name]
    except KeyError as error:
        raise ValueError(f"Unknown estimator {name!r}; choose from {sorted(factories)}") from error


def _positive_probability(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


@dataclass
class FactorizedRelationModel:
    figure_model: object | None
    paper_model: object
    related_model: object

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        pair_type: str,
        estimator_factory: Callable[[], object] = default_binary_estimator,
    ) -> "FactorizedRelationModel":
        labels = np.asarray(labels).astype(str)
        figure_model: object | None = None
        if pair_type == "image+text":
            figure_model = estimator_factory()
            figure_model.fit(features, (labels == "same_figure").astype(np.uint8))

        not_figure = labels != "same_figure"
        paper_model = estimator_factory()
        paper_model.fit(
            features[not_figure],
            (labels[not_figure] == "same_paper").astype(np.uint8),
        )

        different_paper = np.isin(labels, ("related_papers", "unrelated_papers"))
        related_model = estimator_factory()
        related_model.fit(
            features[different_paper],
            (labels[different_paper] == "related_papers").astype(np.uint8),
        )
        return cls(figure_model, paper_model, related_model)

    def predict_proba(self, features: np.ndarray, pair_type: str) -> np.ndarray:
        if self.figure_model is None:
            q_figure = np.zeros(len(features), dtype=np.float64)
        else:
            q_figure = _positive_probability(self.figure_model, features)
        q_paper = _positive_probability(self.paper_model, features)
        q_related = _positive_probability(self.related_model, features)
        return FactorizedProbabilities(q_figure, q_paper, q_related).leaf_probabilities(
            pair_type=np.full(len(features), pair_type)
        )


def apply_logit_offsets(probabilities: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped)
    shifted = logits + np.asarray(offsets, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def optimize_macro_f1_offsets(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    rounds: int = 4,
    grid: Iterable[float] = np.linspace(-0.8, 0.8, 33),
) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    best_score = -np.inf
    for _ in range(rounds):
        changed = False
        for class_index in range(len(LABELS)):
            local_best = offsets[class_index]
            for candidate in grid:
                trial = offsets.copy()
                trial[class_index] = candidate
                prediction = np.asarray(LABELS)[apply_logit_offsets(probabilities, trial).argmax(1)]
                score = f1_score(labels, prediction, labels=list(LABELS), average="macro")
                if score > best_score + 1e-12:
                    best_score = score
                    local_best = candidate
                    changed = True
            offsets[class_index] = local_best
        if not changed:
            break
    return offsets


def crossfit_logit_offsets(
    probabilities: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Tune offsets away from each validation fold before scoring that fold."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    fold_ids = np.asarray(fold_ids)
    if len(probabilities) != len(labels) or len(labels) != len(fold_ids):
        raise ValueError("probabilities, labels and fold_ids must contain the same number of rows")
    calibrated = np.zeros_like(probabilities, dtype=np.float64)
    offsets_by_fold: dict[int, np.ndarray] = {}
    for fold in sorted(int(value) for value in np.unique(fold_ids)):
        valid = fold_ids == fold
        train = ~valid
        if not valid.any() or not train.any():
            raise ValueError(f"Fold {fold} does not have both tuning and validation rows")
        offsets = optimize_macro_f1_offsets(probabilities[train], labels[train])
        calibrated[valid] = apply_logit_offsets(probabilities[valid], offsets)
        offsets_by_fold[fold] = offsets
    return calibrated, offsets_by_fold
