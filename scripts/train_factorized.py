#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace, pair_feature_matrix
from astroclimb.evaluation import group_bootstrap_macro_f1_ci
from astroclimb.features import lexical_feature_matrix, multimodal_ocr_feature_matrix
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.modeling import (
    FactorizedRelationModel,
    apply_logit_offsets,
    crossfit_logit_offsets,
    estimator_factory,
    optimize_macro_f1_offsets,
)
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.splits import (
    component_stratified_folds,
    object_component_groups,
    validate_disjoint_folds,
)
from astroclimb.submission import hard_submission, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe factorized AstroCLIMB baseline.")
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
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--estimator",
        choices=("logistic", "catboost"),
        default="logistic",
        help="Binary head used at each ontology decision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.audit_dir / "meta_train.parquet")
    test = pd.read_parquet(args.audit_dir / "meta_test.parquet")
    spaces = [EncoderSpace.load(args.embedding_dir, name) for name in args.spaces]
    text_lookup: dict[str, str] | None = None
    if args.object_cache_dir is not None:
        text_frame = pd.read_parquet(args.object_cache_dir / "texts.parquet")
        if text_frame["hash"].astype(str).duplicated().any():
            raise ValueError("texts.parquet contains duplicate hashes")
        text_lookup = dict(zip(text_frame["hash"].astype(str), text_frame["text"].astype(str)))
    ocr_lookup = load_ocr_jsonl(args.ocr_jsonl) if args.ocr_jsonl is not None else None
    if ocr_lookup is not None and text_lookup is None:
        raise ValueError("--ocr-jsonl requires --object-cache-dir for caption lookup")
    fold_pairs = list(component_stratified_folds(train, n_splits=args.folds))
    fold_reports = validate_disjoint_folds(train, fold_pairs)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid_index) in enumerate(fold_pairs):
        fold_id[valid_index] = fold
    if (fold_id < 0).any():
        raise AssertionError("Every training row must receive exactly one validation fold")

    oof = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    test_probabilities = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    factory = estimator_factory(args.estimator)
    for pair_type in PAIR_TYPES:
        train_indices = np.flatnonzero(train["pair_type"].to_numpy() == pair_type)
        test_indices = np.flatnonzero(test["pair_type"].to_numpy() == pair_type)
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
        for fold in range(args.folds):
            local_valid = np.flatnonzero(fold_id[train_indices] == fold)
            local_train = np.flatnonzero(fold_id[train_indices] != fold)
            model = FactorizedRelationModel.fit(
                train_features[local_train],
                train_subset.iloc[local_train]["label"].to_numpy(),
                pair_type,
                estimator_factory=factory,
            )
            oof[train_indices[local_valid]] = model.predict_proba(
                train_features[local_valid], pair_type
            )
            test_probabilities[test_indices] += (
                model.predict_proba(test_features, pair_type) / args.folds
            )

    truth = train["label"].astype(str).to_numpy()
    raw_oof = average_duplicate_pairs(train, oof)
    raw_predictions = np.asarray(LABELS)[raw_oof.argmax(1)]
    raw_score = f1_score(truth, raw_predictions, labels=list(LABELS), average="macro")
    calibrated_oof, crossfit_offsets = crossfit_logit_offsets(oof, truth, fold_id)
    final_offsets = optimize_macro_f1_offsets(oof, truth)
    calibrated_test = apply_logit_offsets(test_probabilities, final_offsets)
    calibrated_oof = average_duplicate_pairs(train, calibrated_oof)
    calibrated_test = average_duplicate_pairs(test, calibrated_test)
    predictions = np.asarray(LABELS)[calibrated_oof.argmax(1)]
    score = f1_score(truth, predictions, labels=list(LABELS), average="macro")
    modality_scores = {}
    for pair_type in PAIR_TYPES:
        mask = train["pair_type"].to_numpy() == pair_type
        feasible_labels = list(LABELS) if pair_type == "image+text" else list(LABELS[1:])
        modality_scores[pair_type] = f1_score(
            truth[mask], predictions[mask], labels=feasible_labels, average="macro"
        )
    fold_scores = {
        str(fold): f1_score(
            truth[fold_id == fold],
            predictions[fold_id == fold],
            labels=list(LABELS),
            average="macro",
        )
        for fold in sorted(int(value) for value in np.unique(fold_id))
    }
    bootstrap = group_bootstrap_macro_f1_ci(
        truth,
        predictions,
        object_component_groups(train),
        LABELS,
        args.bootstrap_resamples,
        20260919,
    )
    report = {
        "validation": "object-component StratifiedGroupKFold",
        "folds": fold_reports,
        "spaces": args.spaces,
        "estimator": args.estimator,
        "lexical_features": text_lookup is not None,
        "ocr_features": ocr_lookup is not None,
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
        "final_test_logit_offsets": dict(zip(LABELS, final_offsets.tolist())),
        "crossfit_logit_offsets": {
            str(fold): dict(zip(LABELS, offsets.tolist()))
            for fold, offsets in crossfit_offsets.items()
        },
        "classification_report": classification_report(
            train["label"].astype(str),
            predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            train["label"].astype(str), predictions, labels=list(LABELS)
        ).tolist(),
    }
    (args.output_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.output_dir / "oof_probabilities_raw.npy", raw_oof.astype(np.float32))
    np.save(args.output_dir / "oof_probabilities.npy", calibrated_oof.astype(np.float32))
    np.save(args.output_dir / "test_probabilities.npy", calibrated_test.astype(np.float32))
    submission = hard_submission(test["id"].to_numpy(), calibrated_test, args.sample)
    write_submission(submission, args.sample, args.output_dir / "submission.csv")
    print(f"Object-disjoint raw OOF macro-F1: {raw_score:.5f}")
    print(f"Object-disjoint cross-fit calibrated macro-F1: {score:.5f}")
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
