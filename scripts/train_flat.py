#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace, pair_feature_matrix
from astroclimb.features import lexical_feature_matrix, multimodal_ocr_feature_matrix
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.integrity import assert_safe_input
from astroclimb.modeling import (
    apply_logit_offsets,
    crossfit_logit_offsets,
    optimize_macro_f1_offsets,
)
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.splits import component_stratified_folds, validate_disjoint_folds
from astroclimb.submission import hard_submission, write_submission


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe per-modality flat-head AstroCLIMB ablation."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument(
        "--object-cache-dir",
        type=Path,
        default=None,
        help="Optional cache containing texts.parquet for leakage-safe lexical features",
    )
    parser.add_argument(
        "--ocr-jsonl",
        type=Path,
        default=None,
        help="Optional label-free image OCR cache produced by scripts/ocr_macos.swift",
    )
    parser.add_argument("--spaces", nargs="+", default=["clip"])
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--estimator",
        choices=("logistic", "catboost"),
        default="logistic",
        help="Flat multiclass head fitted independently for each pair modality",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=650,
        help="Boosting iterations for --estimator catboost",
    )
    return parser.parse_args()


def flat_estimator_factory(name: str, *, catboost_iterations: int) -> Callable[[], object]:
    if name == "logistic":

        def logistic() -> object:
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=20260919,
                ),
            )

        return logistic

    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional ML dependencies with `pip install -e '.[ml]'`"
            ) from error

        def catboost() -> object:
            return CatBoostClassifier(
                iterations=catboost_iterations,
                depth=7,
                learning_rate=0.04,
                loss_function="MultiClass",
                eval_metric="MultiClass",
                auto_class_weights="Balanced",
                random_seed=20260919,
                verbose=False,
                allow_writing_files=False,
            )

        return catboost

    raise ValueError(f"Unknown estimator {name!r}")


def feasible_class_indices(pair_type: str) -> tuple[int, ...]:
    if pair_type == "image+text":
        return tuple(range(len(LABELS)))
    return tuple(range(1, len(LABELS)))


def predict_four_class(model: object, features: np.ndarray, pair_type: str) -> np.ndarray:
    local = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    expected = set(feasible_class_indices(pair_type))
    observed = set(int(value) for value in classes)
    if observed != expected:
        raise ValueError(
            f"{pair_type} head exposes classes {sorted(observed)}, expected {sorted(expected)}"
        )
    if local.shape != (len(features), len(classes)):
        raise ValueError(
            f"Unexpected probability shape {local.shape} for {len(features)} rows and "
            f"{len(classes)} classes"
        )
    output = np.zeros((len(features), len(LABELS)), dtype=np.float64)
    output[:, classes] = local
    return output


def enforce_feasible_probabilities(meta: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    if probabilities.shape != (len(meta), len(LABELS)):
        raise ValueError(
            f"Expected probability shape {(len(meta), len(LABELS))}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    not_mixed = meta["pair_type"].astype(str).to_numpy() != "image+text"
    probabilities[not_mixed, LABEL_TO_INDEX["same_figure"]] = 0.0
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("At least one probability row has no feasible class mass")
    return probabilities / totals


def load_text_lookup(cache_dir: Path | None) -> dict[str, str] | None:
    if cache_dir is None:
        return None
    text_frame = pd.read_parquet(cache_dir / "texts.parquet")
    hashes = text_frame["hash"].astype(str)
    if hashes.duplicated().any():
        raise ValueError("texts.parquet contains duplicate hashes")
    return dict(zip(hashes, text_frame["text"].astype(str)))


def count_labels(values: np.ndarray) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def score_reports(
    train: pd.DataFrame,
    truth: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    modality_reports: dict[str, object] = {}
    for pair_type in PAIR_TYPES:
        mask = train["pair_type"].astype(str).to_numpy() == pair_type
        feasible = list(LABELS) if pair_type == "image+text" else list(LABELS[1:])
        modality_reports[pair_type] = {
            "rows": int(mask.sum()),
            "labels": feasible,
            "label_counts": count_labels(truth[mask]),
            "macro_f1_raw_oof": float(
                f1_score(
                    truth[mask],
                    raw_predictions[mask],
                    labels=feasible,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1_crossfit_calibrated": float(
                f1_score(
                    truth[mask],
                    calibrated_predictions[mask],
                    labels=feasible,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    fold_reports: dict[str, object] = {}
    for fold in sorted(int(value) for value in np.unique(fold_id)):
        mask = fold_id == fold
        fold_reports[str(fold)] = {
            "rows": int(mask.sum()),
            "label_counts": count_labels(truth[mask]),
            "macro_f1_raw_oof": float(
                f1_score(
                    truth[mask],
                    raw_predictions[mask],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1_crossfit_calibrated": float(
                f1_score(
                    truth[mask],
                    calibrated_predictions[mask],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    return modality_reports, fold_reports


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.catboost_iterations < 1:
        raise ValueError("--catboost-iterations must be positive")

    sample_path = assert_safe_input(args.sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.audit_dir / "meta_train.parquet")
    test = pd.read_parquet(args.audit_dir / "meta_test.parquet")
    spaces = [EncoderSpace.load(args.embedding_dir, name) for name in args.spaces]
    text_lookup = load_text_lookup(args.object_cache_dir)
    ocr_lookup = load_ocr_jsonl(args.ocr_jsonl) if args.ocr_jsonl is not None else None
    if ocr_lookup is not None and text_lookup is None:
        raise ValueError("--ocr-jsonl requires --object-cache-dir for caption lookup")

    fold_pairs = list(component_stratified_folds(train, n_splits=args.folds))
    disjoint_fold_reports = validate_disjoint_folds(train, fold_pairs)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid_index) in enumerate(fold_pairs):
        fold_id[valid_index] = fold
    if (fold_id < 0).any():
        raise AssertionError("Every training row must receive exactly one validation fold")

    truth = train["label"].astype(str).to_numpy()
    unknown_labels = sorted(set(truth).difference(LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown training labels: {unknown_labels}")
    target = np.asarray([LABEL_TO_INDEX[label] for label in truth], dtype=np.int64)
    oof = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    test_probabilities = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    estimator_factory = flat_estimator_factory(
        args.estimator, catboost_iterations=args.catboost_iterations
    )
    modality_fold_training: dict[str, list[dict[str, object]]] = {}

    train_pair_types = train["pair_type"].astype(str).to_numpy()
    test_pair_types = test["pair_type"].astype(str).to_numpy()
    for pair_type in PAIR_TYPES:
        train_indices = np.flatnonzero(train_pair_types == pair_type)
        test_indices = np.flatnonzero(test_pair_types == pair_type)
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(
                f"Pair type {pair_type!r} needs train and test rows, got "
                f"{len(train_indices)} and {len(test_indices)}"
            )

        train_subset = train.iloc[train_indices].reset_index(drop=True)
        test_subset = test.iloc[test_indices].reset_index(drop=True)
        train_features = pair_feature_matrix(train_subset, spaces)
        test_features = pair_feature_matrix(test_subset, spaces)
        if ocr_lookup is not None and text_lookup is not None:
            train_features = np.column_stack(
                [
                    train_features,
                    multimodal_ocr_feature_matrix(train_subset, text_lookup, ocr_lookup),
                ]
            )
            test_features = np.column_stack(
                [
                    test_features,
                    multimodal_ocr_feature_matrix(test_subset, text_lookup, ocr_lookup),
                ]
            )
        elif text_lookup is not None and pair_type == "text+text":
            train_features = np.column_stack(
                [train_features, lexical_feature_matrix(train_subset, text_lookup)]
            )
            test_features = np.column_stack(
                [test_features, lexical_feature_matrix(test_subset, text_lookup)]
            )

        local_target = target[train_indices]
        modality_fold_training[pair_type] = []
        for fold in range(args.folds):
            local_valid = np.flatnonzero(fold_id[train_indices] == fold)
            local_train = np.flatnonzero(fold_id[train_indices] != fold)
            if len(local_train) == 0 or len(local_valid) == 0:
                raise ValueError(
                    f"{pair_type} fold {fold} has train={len(local_train)}, "
                    f"validation={len(local_valid)}"
                )
            model = estimator_factory()
            model.fit(train_features[local_train], local_target[local_train])
            oof[train_indices[local_valid]] = predict_four_class(
                model, train_features[local_valid], pair_type
            )
            test_probabilities[test_indices] += (
                predict_four_class(model, test_features, pair_type) / args.folds
            )
            modality_fold_training[pair_type].append(
                {
                    "fold": fold,
                    "train_rows": int(len(local_train)),
                    "valid_rows": int(len(local_valid)),
                    "feature_count": int(train_features.shape[1]),
                    "train_label_counts": count_labels(truth[train_indices[local_train]]),
                    "valid_label_counts": count_labels(truth[train_indices[local_valid]]),
                }
            )

    raw_oof = enforce_feasible_probabilities(
        train, average_duplicate_pairs(train, enforce_feasible_probabilities(train, oof))
    )
    raw_test = enforce_feasible_probabilities(
        test,
        average_duplicate_pairs(test, enforce_feasible_probabilities(test, test_probabilities)),
    )
    raw_predictions = np.asarray(LABELS)[raw_oof.argmax(axis=1)]
    raw_score = f1_score(
        truth,
        raw_predictions,
        labels=list(LABELS),
        average="macro",
        zero_division=0,
    )

    calibrated_oof, crossfit_offsets = crossfit_logit_offsets(raw_oof, truth, fold_id)
    calibrated_oof = enforce_feasible_probabilities(
        train, average_duplicate_pairs(train, calibrated_oof)
    )
    final_offsets = optimize_macro_f1_offsets(raw_oof, truth)
    calibrated_test = enforce_feasible_probabilities(
        test,
        average_duplicate_pairs(test, apply_logit_offsets(raw_test, final_offsets)),
    )
    calibrated_predictions = np.asarray(LABELS)[calibrated_oof.argmax(axis=1)]
    calibrated_score = f1_score(
        truth,
        calibrated_predictions,
        labels=list(LABELS),
        average="macro",
        zero_division=0,
    )
    modality_reports, fold_score_reports = score_reports(
        train,
        truth,
        raw_predictions,
        calibrated_predictions,
        fold_id,
    )

    report = {
        "name": "per-modality flat multiclass ablation",
        "validation": "object-component StratifiedGroupKFold",
        "head": "flat_multiclass",
        "spaces": args.spaces,
        "estimator": args.estimator,
        "catboost_iterations": args.catboost_iterations if args.estimator == "catboost" else None,
        "lexical_features": text_lookup is not None,
        "ocr_features": ocr_lookup is not None,
        "same_figure_masked_outside_image_text": True,
        "macro_f1_raw_oof": float(raw_score),
        "macro_f1_crossfit_calibrated": float(calibrated_score),
        "modality_reports": modality_reports,
        "fold_score_reports": fold_score_reports,
        "disjoint_folds": disjoint_fold_reports,
        "modality_fold_training": modality_fold_training,
        "final_test_logit_offsets": dict(zip(LABELS, final_offsets.tolist())),
        "crossfit_logit_offsets": {
            str(fold): dict(zip(LABELS, offsets.tolist()))
            for fold, offsets in crossfit_offsets.items()
        },
        "classification_report_raw_oof": classification_report(
            truth,
            raw_predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_crossfit_calibrated": classification_report(
            truth,
            calibrated_predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_raw_oof": confusion_matrix(
            truth, raw_predictions, labels=list(LABELS)
        ).tolist(),
        "confusion_matrix_crossfit_calibrated": confusion_matrix(
            truth, calibrated_predictions, labels=list(LABELS)
        ).tolist(),
    }
    (args.output_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.output_dir / "oof_probabilities_raw.npy", raw_oof.astype(np.float32))
    np.save(args.output_dir / "oof_probabilities.npy", calibrated_oof.astype(np.float32))
    np.save(args.output_dir / "test_probabilities_raw.npy", raw_test.astype(np.float32))
    np.save(args.output_dir / "test_probabilities.npy", calibrated_test.astype(np.float32))
    submission = hard_submission(test["id"].to_numpy(), calibrated_test, sample_path)
    write_submission(submission, sample_path, args.output_dir / "submission.csv")
    print(f"Object-disjoint raw OOF macro-F1: {raw_score:.5f}")
    print(f"Object-disjoint cross-fit calibrated macro-F1: {calibrated_score:.5f}")
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
