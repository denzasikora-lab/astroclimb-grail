#!/usr/bin/env python3
"""Paired component bootstrap of saved, aligned OOF predictions, never test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.integrity import assert_safe_input
from astroclimb.splits import component_stratified_folds, object_component_groups


def macro_from_confusion(matrix):
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    ).mean(axis=-1)


def compare(meta, baseline, candidate, resamples=2000):
    truth = meta["label"].astype(str).to_numpy()
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    numeric_truth = np.array([label_to_index[label] for label in truth])
    group = object_component_groups(meta)
    grouped_confusion = np.zeros((2, group.max() + 1, 4, 4), dtype=np.int64)
    predictions = []
    for model_id, probability in enumerate((baseline, candidate)):
        if probability.shape != (len(meta), 4) or not np.isfinite(probability).all():
            raise ValueError("Invalid OOF probabilities")
        prediction = probability.argmax(axis=1)
        predictions.append(prediction)
        np.add.at(grouped_confusion[model_id], (group, numeric_truth, prediction), 1)
    group_count = grouped_confusion.shape[1]
    rng = np.random.default_rng(20260919)
    sampled = np.empty((resamples, 2))
    for step in range(resamples):
        weights = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        confusion = np.einsum("g,mgij->mij", weights, grouped_confusion)
        sampled[step] = macro_from_confusion(confusion)
    point = macro_from_confusion(grouped_confusion.sum(axis=1))
    delta = sampled[:, 1] - sampled[:, 0]
    report = {
        "baseline_macro_f1": float(point[0]),
        "candidate_macro_f1": float(point[1]),
        "absolute_delta": float(point[1] - point[0]),
        "relative_delta_percent": float(100 * (point[1] / point[0] - 1)),
        "paired_component_bootstrap_delta_95ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        "candidate_component_bootstrap_95ci": np.quantile(sampled[:, 1], [0.025, 0.975]).tolist(),
        "resamples": resamples,
        "ci_scope": "sampling of fixed predictions; excludes search and training uncertainty",
        "classification_report": classification_report(
            truth, np.asarray(LABELS)[predictions[1]], labels=list(LABELS), output_dict=True
        ),
        "by_modality": {},
        "by_fold": [],
    }
    for pair_type in PAIR_TYPES:
        mask = meta["pair_type"].to_numpy() == pair_type
        labels = LABELS if pair_type == "image+text" else LABELS[1:]
        report["by_modality"][pair_type] = {
            name: float(
                f1_score(
                    truth[mask],
                    np.asarray(LABELS)[prediction[mask]],
                    labels=list(labels),
                    average="macro",
                )
            )
            for name, prediction in zip(("baseline", "candidate"), predictions)
        }
    for fold, (_, validation) in enumerate(component_stratified_folds(meta)):
        report["by_fold"].append(
            {
                "fold": fold,
                **{
                    name: float(
                        f1_score(
                            truth[validation],
                            np.asarray(LABELS)[prediction[validation]],
                            labels=list(LABELS),
                            average="macro",
                        )
                    )
                    for name, prediction in zip(("baseline", "candidate"), predictions)
                },
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        pd.read_parquet(assert_safe_input(args.meta)),
        np.load(assert_safe_input(args.baseline)),
        np.load(assert_safe_input(args.candidate)),
    )
    result["paths"] = {
        "meta": str(args.meta),
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("classification_report", "paths")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
