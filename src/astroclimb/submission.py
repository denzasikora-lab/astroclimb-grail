from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS, SUBMISSION_COLUMNS
from .integrity import assert_safe_input, validate_submission


def hard_submission(
    ids: np.ndarray,
    probabilities: np.ndarray,
    sample_path: str | Path,
) -> pd.DataFrame:
    sample_source = assert_safe_input(sample_path)
    sample = pd.read_csv(sample_source)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(sample), len(LABELS)):
        raise ValueError(
            f"Expected probabilities shape {(len(sample), len(LABELS))}, got {probabilities.shape}"
        )
    if list(ids) != sample["id"].tolist():
        raise ValueError("Prediction ids do not match sample_submission.csv")
    chosen = probabilities.argmax(axis=1)
    matrix = np.zeros_like(probabilities, dtype=np.uint8)
    matrix[np.arange(len(matrix)), chosen] = 1
    submission = pd.DataFrame(matrix, columns=LABELS)
    submission.insert(0, "id", ids)
    submission = submission.loc[:, SUBMISSION_COLUMNS]
    validate_submission(submission, sample)
    return submission


def write_submission(frame: pd.DataFrame, sample_path: str | Path, output_path: str | Path) -> None:
    sample = pd.read_csv(assert_safe_input(sample_path))
    validate_submission(frame, sample)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
