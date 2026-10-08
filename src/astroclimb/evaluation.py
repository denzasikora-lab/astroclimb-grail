from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from sklearn.metrics import f1_score


def group_bootstrap_macro_f1_ci(
    truth: Sequence[object] | np.ndarray,
    predicted: Sequence[object] | np.ndarray,
    group_ids: Sequence[Hashable] | np.ndarray,
    labels: Sequence[object],
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return macro-F1 and a percentile CI from a group bootstrap.

    Each bootstrap replicate draws the same number of groups as observed, with
    replacement. Every row belonging to a selected group is included together;
    a group drawn multiple times contributes all of its rows multiple times.

    Returns ``(point_estimate, lower_2_5, upper_97_5)``.
    """
    truth_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    group_array = np.asarray(group_ids)
    label_list = list(labels)

    if truth_array.ndim != 1 or predicted_array.ndim != 1 or group_array.ndim != 1:
        raise ValueError("truth, predicted, and group_ids must be one-dimensional")
    if not (len(truth_array) == len(predicted_array) == len(group_array)):
        raise ValueError("truth, predicted, and group_ids must have equal lengths")
    if len(truth_array) == 0:
        raise ValueError("At least one observation is required")
    if not label_list:
        raise ValueError("At least one label is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    group_rows: dict[Hashable, list[int]] = {}
    for row_index, group_id in enumerate(group_array.tolist()):
        try:
            group_rows.setdefault(group_id, []).append(row_index)
        except TypeError as error:
            raise ValueError("group_ids must contain hashable values") from error
    rows_by_group = [np.asarray(rows, dtype=np.int64) for rows in group_rows.values()]

    point_estimate = f1_score(
        truth_array,
        predicted_array,
        labels=label_list,
        average="macro",
        zero_division=0,
    )
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.empty(n_resamples, dtype=np.float64)
    group_count = len(rows_by_group)
    for replicate in range(n_resamples):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        sampled_rows = np.concatenate([rows_by_group[index] for index in sampled_groups])
        bootstrap_scores[replicate] = f1_score(
            truth_array[sampled_rows],
            predicted_array[sampled_rows],
            labels=label_list,
            average="macro",
            zero_division=0,
        )

    lower, upper = np.percentile(bootstrap_scores, [2.5, 97.5])
    return float(point_estimate), float(lower), float(upper)
