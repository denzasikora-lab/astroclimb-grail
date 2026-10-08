#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.evaluation import group_bootstrap_macro_f1_ci
from astroclimb.modeling import (
    FactorizedRelationModel,
    apply_logit_offsets,
    crossfit_logit_offsets,
    optimize_macro_f1_offsets,
)
from astroclimb.splits import (
    component_stratified_folds,
    object_component_groups,
    validate_disjoint_folds,
)
from astroclimb.submission import hard_submission, write_submission


def scalar_features(meta: pd.DataFrame) -> np.ndarray:
    length_1 = meta["length_1"].to_numpy(dtype=np.float64)
    length_2 = meta["length_2"].to_numpy(dtype=np.float64)
    minimum = np.minimum(length_1, length_2)
    maximum = np.maximum(length_1, length_2)
    image_length = np.where(meta["type_1"].to_numpy() == "image", length_1, length_2)
    text_length = np.where(meta["type_1"].to_numpy() == "text", length_1, length_2)
    both_images = meta["pair_type"].to_numpy() == "image+image"
    both_texts = meta["pair_type"].to_numpy() == "text+text"
    image_length = np.where(both_texts, 0.0, image_length)
    text_length = np.where(both_images, 0.0, text_length)
    return np.column_stack(
        [
            np.log1p(minimum),
            np.log1p(maximum),
            np.log1p(image_length),
            np.log1p(text_length),
            maximum / np.maximum(minimum, 1.0),
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast leakage-safe scalar baseline.")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.audit_dir / "meta_train.parquet")
    test = pd.read_parquet(args.audit_dir / "meta_test.parquet")
    folds = list(component_stratified_folds(train, n_splits=args.folds))
    fold_reports = validate_disjoint_folds(train, folds)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid_index) in enumerate(folds):
        fold_id[valid_index] = fold

    oof = np.zeros((len(train), 4), dtype=np.float64)
    test_probabilities = np.zeros((len(test), 4), dtype=np.float64)
    for pair_type in PAIR_TYPES:
        train_indices = np.flatnonzero(train["pair_type"].to_numpy() == pair_type)
        test_indices = np.flatnonzero(test["pair_type"].to_numpy() == pair_type)
        train_features = scalar_features(train.iloc[train_indices])
        test_features = scalar_features(test.iloc[test_indices])
        labels = train.iloc[train_indices]["label"].astype(str).to_numpy()
        for fold in range(args.folds):
            local_valid = np.flatnonzero(fold_id[train_indices] == fold)
            local_train = np.flatnonzero(fold_id[train_indices] != fold)
            model = FactorizedRelationModel.fit(
                train_features[local_train], labels[local_train], pair_type
            )
            oof[train_indices[local_valid]] = model.predict_proba(
                train_features[local_valid], pair_type
            )
            test_probabilities[test_indices] += (
                model.predict_proba(test_features, pair_type) / args.folds
            )

    truth = train["label"].astype(str).to_numpy()
    raw_oof = average_duplicate_pairs(train, oof)
    raw_prediction = np.asarray(LABELS)[raw_oof.argmax(axis=1)]
    raw_score = f1_score(truth, raw_prediction, labels=list(LABELS), average="macro")
    oof, crossfit_offsets = crossfit_logit_offsets(oof, truth, fold_id)
    final_offsets = optimize_macro_f1_offsets(raw_oof, truth)
    oof = average_duplicate_pairs(train, oof)
    test_probabilities = average_duplicate_pairs(
        test, apply_logit_offsets(test_probabilities, final_offsets)
    )
    prediction = np.asarray(LABELS)[oof.argmax(axis=1)]
    score = f1_score(truth, prediction, labels=list(LABELS), average="macro")
    modality_scores = {}
    for pair_type in PAIR_TYPES:
        mask = train["pair_type"].to_numpy() == pair_type
        feasible_labels = list(LABELS) if pair_type == "image+text" else list(LABELS[1:])
        modality_scores[pair_type] = f1_score(
            truth[mask], prediction[mask], labels=feasible_labels, average="macro"
        )
    fold_scores = {
        str(fold): f1_score(
            truth[fold_id == fold],
            prediction[fold_id == fold],
            labels=list(LABELS),
            average="macro",
        )
        for fold in sorted(int(value) for value in np.unique(fold_id))
    }
    bootstrap = group_bootstrap_macro_f1_ci(
        truth,
        prediction,
        object_component_groups(train),
        LABELS,
        args.bootstrap_resamples,
        20260919,
    )
    report = {
        "name": "metadata-free scalar emergency baseline",
        "features": [
            "log minimum object length",
            "log maximum object length",
            "log image length",
            "log text length",
            "length ratio",
        ],
        "validation": "object-component StratifiedGroupKFold",
        "macro_f1_raw_oof": raw_score,
        "macro_f1_crossfit_calibrated": score,
        "macro_f1_by_modality_feasible_labels": modality_scores,
        "macro_f1_by_fold": fold_scores,
        "macro_f1_group_bootstrap_95ci": {
            "point": bootstrap[0],
            "lower": bootstrap[1],
            "upper": bootstrap[2],
            "resamples": args.bootstrap_resamples,
        },
        "folds": fold_reports,
        "final_test_logit_offsets": dict(zip(LABELS, final_offsets.tolist())),
        "crossfit_logit_offsets": {
            str(fold): dict(zip(LABELS, offsets.tolist()))
            for fold, offsets in crossfit_offsets.items()
        },
        "classification_report": classification_report(
            truth,
            prediction,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=list(LABELS)).tolist(),
    }
    (args.output_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.output_dir / "oof_probabilities_raw.npy", raw_oof.astype(np.float32))
    np.save(args.output_dir / "oof_probabilities.npy", oof.astype(np.float32))
    np.save(args.output_dir / "test_probabilities.npy", test_probabilities.astype(np.float32))
    submission = hard_submission(test["id"].to_numpy(), test_probabilities, args.sample)
    write_submission(submission, args.sample, args.output_dir / "submission.csv")
    print(f"Object-disjoint raw OOF macro-F1: {raw_score:.5f}")
    print(f"Object-disjoint cross-fit calibrated macro-F1: {score:.5f}")
    print(submission.loc[:, LABELS].sum().to_dict())
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
