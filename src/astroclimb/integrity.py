from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import LABELS, SUBMISSION_COLUMNS


class LeakageGuardError(RuntimeError):
    """Raised when a forbidden test-label source reaches the pipeline."""


def assert_safe_input(path: str | Path) -> Path:
    """Reject the accidentally exposed test-label file before it can be opened."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name.casefold() == "solution.csv":
        raise LeakageGuardError(
            "solution.csv contains test labels and is quarantined. "
            "It must not be read for training, priors, tuning, evaluation, or submission."
        )
    return resolved


def assert_no_test_targets(columns: Iterable[str]) -> None:
    leaked = sorted(set(columns).intersection(LABELS))
    if leaked:
        raise LeakageGuardError(f"Test features contain forbidden target columns: {leaked}")


def validate_one_hot(frame: pd.DataFrame, *, allow_probabilities: bool = False) -> None:
    missing = [column for column in LABELS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    values = frame.loc[:, LABELS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target matrix contains NaN or infinite values")
    if allow_probabilities:
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Probabilities must be in [0, 1]")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to 1")
        return
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Submission labels must be binary")
    if not np.equal(values.sum(axis=1), 1).all():
        raise ValueError("Each row must contain exactly one positive class")


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected = list(SUBMISSION_COLUMNS)
    if list(frame.columns) != expected:
        raise ValueError(f"Expected columns {expected}, received {list(frame.columns)}")
    if len(frame) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, received {len(frame)}")
    if frame["id"].duplicated().any():
        raise ValueError("Submission ids must be unique")
    if frame["id"].tolist() != sample["id"].tolist():
        raise ValueError("Submission id order must exactly match sample_submission.csv")
    validate_one_hot(frame)
