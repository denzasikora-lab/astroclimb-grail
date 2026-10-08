#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.splits import component_stratified_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze training OOF probabilities without reading test labels."
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Directory containing meta_train.parquet",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing oof_probabilities.npy",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    return parser.parse_args()


def validate_probabilities(probabilities: np.ndarray, row_count: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (row_count, len(LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected OOF probabilities with shape {expected_shape}, received {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain non-finite values")
    if (probabilities < -1e-7).any():
        raise ValueError("OOF probabilities contain negative values")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any() or not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("Every OOF probability row must sum to one")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def subset_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    *,
    score_labels: tuple[str, ...] = LABELS,
) -> dict[str, Any]:
    subset_truth = truth[mask]
    subset_predictions = predictions[mask]
    if len(subset_truth) == 0:
        return {"rows": 0, "accuracy": None, "macro_f1": None, "label_counts": {}}
    return {
        "rows": int(len(subset_truth)),
        "accuracy": float(accuracy_score(subset_truth, subset_predictions)),
        "macro_f1": float(
            f1_score(
                subset_truth,
                subset_predictions,
                labels=list(score_labels),
                average="macro",
                zero_division=0,
            )
        ),
        "label_counts": {label: int(count) for label, count in Counter(subset_truth).items()},
    }


def calibration_report(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    truth_indices = np.asarray([label_to_index[label] for label in truth], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == truth_indices

    bin_ids = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    reliability_bins: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if count == 0:
            reliability_bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(accuracy - mean_confidence)
        ece += count / len(truth) * gap
        reliability_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    one_hot = np.eye(len(LABELS), dtype=np.float64)[truth_indices]
    per_class_brier = np.mean((probabilities - one_hot) ** 2, axis=0)
    return {
        "ece_top_label": float(ece),
        "ece_bins": n_bins,
        "reliability_bins": reliability_bins,
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "brier_per_class": {
            label: float(per_class_brier[index]) for index, label in enumerate(LABELS)
        },
    }


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "rows": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def entropy_report(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    correct = predictions == truth
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy_nats = -np.sum(clipped * np.log(clipped), axis=1)
    normalized_entropy = entropy_nats / np.log(len(LABELS))
    return {
        "units": "nats",
        "correct": {
            "entropy_nats": distribution_summary(entropy_nats[correct]),
            "entropy_normalized": distribution_summary(normalized_entropy[correct]),
        },
        "incorrect": {
            "entropy_nats": distribution_summary(entropy_nats[~correct]),
            "entropy_normalized": distribution_summary(normalized_entropy[~correct]),
        },
    }


def text_length_report(
    train: pd.DataFrame,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    type_1 = train["type_1"].astype(str).to_numpy()
    type_2 = train["type_2"].astype(str).to_numpy()
    length_1 = train["length_1"].to_numpy(dtype=np.float64)
    length_2 = train["length_2"].to_numpy(dtype=np.float64)
    text_count = (type_1 == "text").astype(np.int8) + (type_2 == "text").astype(np.int8)
    text_length_sum = np.where(type_1 == "text", length_1, 0.0) + np.where(
        type_2 == "text", length_2, 0.0
    )
    text_length = np.divide(
        text_length_sum,
        text_count,
        out=np.full(len(train), np.nan, dtype=np.float64),
        where=text_count > 0,
    )
    eligible = np.isfinite(text_length)
    eligible_lengths = text_length[eligible]
    if len(eligible_lengths) == 0:
        return {"definition": "mean character length of text objects in the pair", "bins": []}

    edges = np.quantile(eligible_lengths, [0.0, 0.25, 0.5, 0.75, 1.0])
    assignments = np.searchsorted(edges[1:-1], text_length, side="right")
    bins: list[dict[str, Any]] = []
    for quantile in range(4):
        mask = eligible & (assignments == quantile)
        metrics = subset_metrics(truth, predictions, mask)
        metrics.update(
            {
                "quantile": quantile + 1,
                "lower": float(edges[quantile]),
                "upper": float(edges[quantile + 1]),
            }
        )
        bins.append(metrics)
    return {
        "definition": "mean character length of text objects in the pair",
        "eligible_rows": int(eligible.sum()),
        "excluded_image_image_rows": int((~eligible).sum()),
        "quantile_edges": [float(value) for value in edges],
        "bins": bins,
    }


def repeated_object_report(
    train: pd.DataFrame,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    all_hashes = pd.concat([train["hash_1"], train["hash_2"]], ignore_index=True)
    occurrence_counts = all_hashes.value_counts()
    repeated_1 = train["hash_1"].map(occurrence_counts).to_numpy() > 1
    repeated_2 = train["hash_2"].map(occurrence_counts).to_numpy() > 1
    involvement = repeated_1.astype(np.int8) + repeated_2.astype(np.int8)
    names = {0: "no_repeated_endpoint", 1: "one_repeated_endpoint", 2: "two_repeated_endpoints"}
    return {
        "definition": "an endpoint is repeated when its object hash occurs in more than one pair",
        "categories": {
            names[count]: subset_metrics(truth, predictions, involvement == count)
            for count in range(3)
        },
    }


def reported_fold_count(run_dir: Path) -> int | None:
    report_path = run_dir / "cv_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("folds", "disjoint_folds"):
        folds = report.get(key)
        if isinstance(folds, list) and len(folds) >= 2:
            return len(folds)
    for key in ("macro_f1_by_fold", "fold_score_reports"):
        folds = report.get(key)
        if isinstance(folds, dict) and len(folds) >= 2:
            return len(folds)
    return None


def fold_report(
    train: pd.DataFrame,
    truth: np.ndarray,
    predictions: np.ndarray,
    run_dir: Path,
) -> dict[str, Any]:
    if "fold" in train.columns:
        fold_ids = train["fold"].to_numpy()
        source = "meta_train.fold"
    else:
        try:
            n_splits = reported_fold_count(run_dir)
        except (OSError, json.JSONDecodeError):
            n_splits = None
        if n_splits is None:
            return {"available": False, "reason": "No fold assignments or fold count found"}
        fold_ids = np.full(len(train), -1, dtype=np.int16)
        for fold, (_, valid_indices) in enumerate(
            component_stratified_folds(train, n_splits=n_splits)
        ):
            fold_ids[valid_indices] = fold
        if (fold_ids < 0).any():
            raise AssertionError("Reconstructed folds did not cover every training row")
        source = f"reconstructed object-component split with {n_splits} folds"

    folds = {
        str(fold): subset_metrics(truth, predictions, fold_ids == fold)
        for fold in sorted(np.unique(fold_ids).tolist())
    }
    return {"available": True, "source": source, "folds": folds}


def analyze(audit_dir: Path, run_dir: Path) -> dict[str, Any]:
    train_path = audit_dir / "meta_train.parquet"
    oof_path = run_dir / "oof_probabilities.npy"
    train = pd.read_parquet(train_path)
    required_columns = {
        "label",
        "pair_type",
        "type_1",
        "type_2",
        "hash_1",
        "hash_2",
        "length_1",
        "length_2",
    }
    missing_columns = required_columns.difference(train.columns)
    if missing_columns:
        raise ValueError(f"meta_train.parquet lacks columns: {sorted(missing_columns)}")

    truth = train["label"].astype(str).to_numpy()
    unknown_labels = sorted(set(truth).difference(LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown training labels: {unknown_labels}")
    probabilities = validate_probabilities(
        np.load(oof_path, allow_pickle=False), row_count=len(train)
    )
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    class_report = classification_report(
        truth,
        predictions,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )

    modalities: dict[str, Any] = {}
    modality_confusions: dict[str, Any] = {}
    pair_types = [pair_type for pair_type in PAIR_TYPES if pair_type in set(train["pair_type"])]
    for pair_type in pair_types:
        mask = train["pair_type"].to_numpy() == pair_type
        feasible_labels = LABELS if pair_type == "image+text" else LABELS[1:]
        modality_metrics = subset_metrics(truth, predictions, mask, score_labels=feasible_labels)
        modality_metrics["score_labels"] = list(feasible_labels)
        modality_metrics["macro_f1_all_labels"] = float(
            f1_score(
                truth[mask],
                predictions[mask],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        )
        modalities[pair_type] = modality_metrics
        modality_confusions[pair_type] = {
            "labels": list(LABELS),
            "matrix": confusion_matrix(
                truth[mask], predictions[mask], labels=list(LABELS)
            ).tolist(),
        }

    return {
        "provenance": {
            "train_metadata": str(train_path.resolve()),
            "oof_probabilities": str(oof_path.resolve()),
            "test_labels_read": False,
            "solution_csv_read": False,
        },
        "labels": list(LABELS),
        "overall": {
            "rows": int(len(train)),
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(
                f1_score(
                    truth,
                    predictions,
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(class_report["weighted avg"]["f1-score"]),
            "confusion_matrix": {
                "labels": list(LABELS),
                "matrix": confusion_matrix(truth, predictions, labels=list(LABELS)).tolist(),
            },
        },
        "per_class": {label: class_report[label] for label in LABELS},
        "by_modality": modalities,
        "confusion_matrices_by_modality": modality_confusions,
        "by_fold": fold_report(train, truth, predictions, run_dir),
        "calibration": calibration_report(truth, probabilities),
        "prediction_entropy": entropy_report(truth, probabilities),
        "by_text_length_quantile": text_length_report(train, truth, predictions),
        "by_repeated_object_involvement": repeated_object_report(train, truth, predictions),
    }


def main() -> None:
    args = parse_args()
    report = analyze(args.audit_dir.resolve(), args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
