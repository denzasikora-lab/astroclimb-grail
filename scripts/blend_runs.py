#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from astroclimb.constants import LABELS
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.integrity import assert_safe_input
from astroclimb.splits import component_stratified_folds, validate_disjoint_folds
from astroclimb.submission import hard_submission, write_submission


DEFAULT_SEED = 20260919


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe convex blending of two or more AstroCLIMB runs. "
            "Weights are selected with object-component cross-fitting, and class offsets "
            "are never retuned."
        )
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Run directories in a common audit row order. Each must contain "
            "oof_probabilities.npy and test_probabilities.npy."
        ),
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of object-component folds used by the source OOF runs (default: 5)",
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=DEFAULT_SEED,
        help="StratifiedGroupKFold seed used by the source OOF runs",
    )
    parser.add_argument(
        "--grid-denominator",
        type=int,
        default=20,
        help=("Simplex grid denominator for two to four runs; 20 gives weight increments of 0.05"),
    )
    parser.add_argument(
        "--dirichlet-candidates",
        type=int,
        default=8192,
        help=(
            "Total deterministic candidate budget for more than four runs, including "
            "uniform and one-hot anchors"
        ),
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=256,
        help="Candidate weight vectors evaluated per batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for deterministic Dirichlet candidates",
    )
    return parser.parse_args()


def _integer_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in _integer_compositions(total - first, parts - 1):
            yield (first, *remainder)


def simplex_grid(model_count: int, denominator: int) -> np.ndarray:
    """Return every deterministic simplex point on an integer grid."""
    if not 2 <= model_count <= 4:
        raise ValueError("The exhaustive simplex grid supports two to four models")
    if denominator < 1:
        raise ValueError("grid-denominator must be at least 1")
    integer_weights = np.asarray(
        list(_integer_compositions(denominator, model_count)), dtype=np.float64
    )
    return integer_weights / float(denominator)


def dirichlet_candidates(model_count: int, candidate_count: int, seed: int) -> np.ndarray:
    """Return a reproducible set of simplex candidates with deterministic anchors."""
    minimum = model_count + 1
    if candidate_count < minimum:
        raise ValueError(
            f"dirichlet-candidates must be at least {minimum} for {model_count} models"
        )
    anchors = np.vstack(
        [
            np.full((1, model_count), 1.0 / model_count, dtype=np.float64),
            np.eye(model_count, dtype=np.float64),
        ]
    )
    remaining = candidate_count - len(anchors)
    if remaining == 0:
        return anchors
    generator = np.random.default_rng(seed)
    sampled = generator.dirichlet(np.ones(model_count, dtype=np.float64), size=remaining)
    return np.vstack([anchors, sampled])


def candidate_weights(
    model_count: int,
    *,
    grid_denominator: int,
    dirichlet_candidate_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if model_count <= 4:
        candidates = simplex_grid(model_count, grid_denominator)
        strategy = {
            "name": "deterministic_simplex_grid",
            "grid_denominator": grid_denominator,
            "weight_increment": 1.0 / grid_denominator,
        }
    else:
        candidates = dirichlet_candidates(model_count, dirichlet_candidate_count, seed)
        strategy = {
            "name": "deterministic_dirichlet_candidates",
            "seed": seed,
            "requested_candidates": dirichlet_candidate_count,
            "anchors": "uniform_plus_one_hot",
        }
    strategy["evaluated_candidates"] = int(len(candidates))
    return candidates, strategy


def _validate_probability_array(
    values: np.ndarray,
    *,
    expected_rows: int,
    source: Path,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    expected_shape = (expected_rows, len(LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(f"Expected {source} shape {expected_shape}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source} contains NaN or infinite values")
    if (probabilities < -1e-8).any() or (probabilities > 1.0 + 1e-8).any():
        raise ValueError(f"{source} contains values outside [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Rows in {source} do not sum to one")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def load_run_probabilities(
    run_dir: Path,
    *,
    train_rows: int,
    test_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    directory = run_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {directory}")
    oof_path = assert_safe_input(directory / "oof_probabilities.npy")
    test_path = assert_safe_input(directory / "test_probabilities.npy")
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing {oof_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}")
    oof = _validate_probability_array(
        np.load(oof_path, allow_pickle=False),
        expected_rows=train_rows,
        source=oof_path,
    )
    test = _validate_probability_array(
        np.load(test_path, allow_pickle=False),
        expected_rows=test_rows,
        source=test_path,
    )
    return oof, test


def _macro_f1_from_indices(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        f1_score(
            truth,
            prediction,
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )


def _candidate_macro_f1(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Vectorized four-class macro-F1 for a batch of candidate predictions."""
    batch_size, row_count = predictions.shape
    if truth.shape != (row_count,):
        raise ValueError("Truth and candidate prediction row counts do not match")
    class_count = len(LABELS)
    candidate_offsets = np.arange(batch_size, dtype=np.int64)[:, None] * class_count**2
    encoded = candidate_offsets + truth[None, :] * class_count + predictions
    confusion = np.bincount(encoded.ravel(), minlength=batch_size * class_count**2).reshape(
        batch_size, class_count, class_count
    )
    true_positive = np.diagonal(confusion, axis1=1, axis2=2).astype(np.float64)
    actual = confusion.sum(axis=2, dtype=np.float64)
    predicted = confusion.sum(axis=1, dtype=np.float64)
    denominator = actual + predicted
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return class_f1.mean(axis=1)


def select_weights(
    model_probabilities: np.ndarray,
    truth: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Select the earliest highest-scoring deterministic candidate."""
    if batch_size < 1:
        raise ValueError("candidate-batch-size must be at least 1")
    if model_probabilities.ndim != 3:
        raise ValueError("Model probabilities must have shape (models, rows, classes)")
    if candidates.shape[1] != model_probabilities.shape[0]:
        raise ValueError("Candidate weights do not match the number of models")
    best_index = -1
    best_score = -np.inf
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        weight_batch = candidates[start:stop]
        blended = np.einsum("bm,mrc->brc", weight_batch, model_probabilities, optimize=True)
        predictions = blended.argmax(axis=2)
        scores = _candidate_macro_f1(truth, predictions)
        local_index = int(np.argmax(scores))
        local_score = float(scores[local_index])
        if local_score > best_score:
            best_index = start + local_index
            best_score = local_score
    if best_index < 0:
        raise AssertionError("No blend candidate was evaluated")
    return candidates[best_index].copy(), best_score


def blend_probabilities(weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    blended = np.tensordot(weights, probabilities, axes=(0, 0))
    return blended / blended.sum(axis=1, keepdims=True)


def _weight_records(run_names: Sequence[str], weights: np.ndarray) -> list[dict[str, object]]:
    return [
        {"run": name, "weight": float(weight)}
        for name, weight in zip(run_names, weights, strict=True)
    ]


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) < 2:
        raise ValueError("At least two --run-dirs are required")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")

    audit_dir = args.audit_dir.expanduser().resolve()
    train_path = assert_safe_input(audit_dir / "meta_train.parquet")
    test_path = assert_safe_input(audit_dir / "meta_test.parquet")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if "label" not in train.columns:
        raise ValueError("meta_train.parquet must contain a label column")
    if "label" in test.columns and test["label"].notna().any():
        raise ValueError("meta_test.parquet must not contain non-null target labels")

    label_to_index = {label: index for index, label in enumerate(LABELS)}
    unknown_labels = sorted(set(train["label"].astype(str)).difference(label_to_index))
    if unknown_labels:
        raise ValueError(f"Unknown training labels: {unknown_labels}")
    truth = train["label"].astype(str).map(label_to_index).to_numpy(dtype=np.int64)

    run_paths = [path.expanduser().resolve() for path in args.run_dirs]
    run_names = [f"{index}:{path.name}" for index, path in enumerate(run_paths)]
    loaded = [
        load_run_probabilities(path, train_rows=len(train), test_rows=len(test))
        for path in run_paths
    ]
    oof_probabilities = np.stack([item[0] for item in loaded], axis=0)
    test_probabilities = np.stack([item[1] for item in loaded], axis=0)

    candidates, strategy = candidate_weights(
        len(run_paths),
        grid_denominator=args.grid_denominator,
        dirichlet_candidate_count=args.dirichlet_candidates,
        seed=args.seed,
    )

    folds = list(
        component_stratified_folds(
            train,
            n_splits=args.folds,
            random_state=args.fold_seed,
        )
    )
    fold_audit = validate_disjoint_folds(train, folds)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid_index) in enumerate(folds):
        if (fold_id[np.asarray(valid_index)] >= 0).any():
            raise AssertionError("A training row was assigned to multiple validation folds")
        fold_id[np.asarray(valid_index)] = fold
    if (fold_id < 0).any():
        raise AssertionError("Every training row must receive one validation fold")

    component_scores = []
    for run_name, run_path, probabilities in zip(
        run_names, run_paths, oof_probabilities, strict=True
    ):
        prediction = probabilities.argmax(axis=1)
        component_scores.append(
            {
                "run": run_name,
                "path": str(run_path),
                "macro_f1_loaded_oof": _macro_f1_from_indices(truth, prediction),
            }
        )

    crossfit_oof = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    fold_reports: list[dict[str, object]] = []
    for fold in range(args.folds):
        selection_mask = fold_id != fold
        heldout_mask = fold_id == fold
        weights, selection_score = select_weights(
            oof_probabilities[:, selection_mask, :],
            truth[selection_mask],
            candidates,
            batch_size=args.candidate_batch_size,
        )
        heldout_probabilities = blend_probabilities(weights, oof_probabilities[:, heldout_mask, :])
        crossfit_oof[heldout_mask] = heldout_probabilities
        heldout_score = _macro_f1_from_indices(
            truth[heldout_mask], heldout_probabilities.argmax(axis=1)
        )
        fold_reports.append(
            {
                "fold": fold,
                "selection_rows": int(selection_mask.sum()),
                "heldout_rows": int(heldout_mask.sum()),
                "weights": _weight_records(run_names, weights),
                "selection_macro_f1": selection_score,
                "heldout_macro_f1": heldout_score,
            }
        )

    crossfit_score = _macro_f1_from_indices(truth, crossfit_oof.argmax(axis=1))
    final_weights, final_selection_score = select_weights(
        oof_probabilities,
        truth,
        candidates,
        batch_size=args.candidate_batch_size,
    )
    blended_test = blend_probabilities(final_weights, test_probabilities)
    blended_test = average_duplicate_pairs(test, blended_test)

    submission = hard_submission(test["id"].to_numpy(), blended_test, args.sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    write_submission(submission, args.sample, submission_path)

    report = {
        "validation": "object-component cross-fit blend selection",
        "class_offsets_retuned": False,
        "duplicate_test_pairs_averaged": True,
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
        "candidate_strategy": strategy,
        "component_scores": component_scores,
        "fold_audit": fold_audit,
        "fold_weights_and_scores": fold_reports,
        "macro_f1_crossfit_blend": crossfit_score,
        "final_weights": _weight_records(run_names, final_weights),
        "macro_f1_final_weight_selection_oof": final_selection_score,
        "submission_class_counts": {label: int(submission[label].sum()) for label in LABELS},
        "submission_path": str(submission_path.resolve()),
    }
    report_path = args.output_dir / "blend_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cross-fit blend OOF macro-F1: {crossfit_score:.5f}")
    print(f"All-OOF final-weight selection macro-F1: {final_selection_score:.5f}")
    print(f"Wrote {submission_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
