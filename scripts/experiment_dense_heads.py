#!/usr/bin/env python3
"""Bounded developmental comparison on original object-component folds.

Only train labels are consumed. Test objects are used for final predictions.
The fixed variants are intentionally not a hyperparameter search.
"""
# ruff: noqa: E402 - thread caps must be set before importing numeric libraries.

from __future__ import annotations

import os

for thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_env] = "2"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace, pair_feature_matrix
from astroclimb.features import multimodal_ocr_feature_matrix
from astroclimb.modeling import FactorizedRelationModel
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.splits import component_stratified_folds, validate_disjoint_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, nargs="*", default=[0.003, 0.015])
    parser.add_argument(
        "--rbf", action="store_true", help="Also run one fixed flat RBF SVC variant"
    )
    args = parser.parse_args()
    if len(args.c) + int(args.rbf) > 3:
        raise ValueError("At most three fixed variants are allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train = pd.read_parquet(args.artifacts / "audit/meta_train.parquet")
    test = pd.read_parquet(args.artifacts / "audit/meta_test.parquet")
    if "label" in test:
        assert test["label"].isna().all(), "Test labels must never be consumed"
        test = test.drop(columns="label")
    truth = train["label"].astype(str).to_numpy()
    folds = list(component_stratified_folds(train, n_splits=5))
    fold_reports = validate_disjoint_folds(train, folds)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid) in enumerate(folds):
        fold_id[valid] = fold
    assert (fold_id >= 0).all()
    np.save(args.output / "fold_ids.npy", fold_id)
    spaces = [
        EncoderSpace.load(args.artifacts / "embeddings", name)
        for name in ("clip_b16", "clip_b32", "scincl", "layout")
    ]
    text = pd.read_parquet(args.artifacts / "object_cache/texts.parquet")
    texts = dict(zip(text["hash"].astype(str), text["text"].astype(str)))
    ocr = load_ocr_jsonl(args.artifacts / "ocr_fast.jsonl")
    features = {}
    for pair_type in PAIR_TYPES:
        train_indices = np.flatnonzero(train["pair_type"].to_numpy() == pair_type)
        test_indices = np.flatnonzero(test["pair_type"].to_numpy() == pair_type)
        train_subset = train.iloc[train_indices].reset_index(drop=True)
        test_subset = test.iloc[test_indices].reset_index(drop=True)
        x_train = np.column_stack(
            (
                pair_feature_matrix(train_subset, spaces),
                multimodal_ocr_feature_matrix(train_subset, texts, ocr),
            )
        ).astype(np.float32)
        x_test = np.column_stack(
            (
                pair_feature_matrix(test_subset, spaces),
                multimodal_ocr_feature_matrix(test_subset, texts, ocr),
            )
        ).astype(np.float32)
        assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
        features[pair_type] = train_indices, test_indices, x_train, x_test
        print(
            json.dumps(
                {
                    "prepared": pair_type,
                    "train_shape": x_train.shape,
                    "elapsed_s": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )

    variants = [("factorized", c) for c in args.c]
    if args.rbf:
        variants.append(("flat_rbf", 3.0))
    for kind, c in variants:
        variant_started = time.monotonic()
        output = args.output / f"{kind}_c{str(c).replace('.', 'p')}"
        output.mkdir(exist_ok=True)
        oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
        test_proba = np.zeros((len(test), len(LABELS)), dtype=np.float32)

        def factory():
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=20260919,
                ),
            )

        for pair_type, (train_indices, test_indices, x_train, x_test) in features.items():
            for fold in range(5):
                valid = np.flatnonzero(fold_id[train_indices] == fold)
                fit = np.flatnonzero(fold_id[train_indices] != fold)
                with threadpool_limits(limits=2):
                    if kind == "factorized":
                        model = FactorizedRelationModel.fit(
                            x_train[fit],
                            truth[train_indices[fit]],
                            pair_type,
                            estimator_factory=factory,
                        )
                        oof[train_indices[valid]] = model.predict_proba(x_train[valid], pair_type)
                        test_proba[test_indices] += model.predict_proba(x_test, pair_type) / 5
                    else:
                        model = make_pipeline(
                            StandardScaler(),
                            SVC(
                                C=c,
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                                random_state=20260919,
                                cache_size=512,
                            ),
                        )
                        model.fit(x_train[fit], truth[train_indices[fit]])
                        valid_probability = model.predict_proba(x_train[valid])
                        test_probability = model.predict_proba(x_test)
                        for column, label in enumerate(model.classes_):
                            target = list(LABELS).index(label)
                            oof[train_indices[valid], target] = valid_probability[:, column]
                            test_proba[test_indices, target] += test_probability[:, column] / 5
                print(
                    json.dumps(
                        {
                            "kind": kind,
                            "C": c,
                            "pair_type": pair_type,
                            "fold": fold,
                            "elapsed_s": round(time.monotonic() - variant_started, 1),
                        }
                    ),
                    flush=True,
                )
        assert np.isfinite(oof).all() and np.isfinite(test_proba).all()
        assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(test_proba.sum(axis=1), 1.0, atol=1e-5)
        assert (oof[train["pair_type"].to_numpy() != "image+text", 0] == 0).all()
        prediction = np.asarray(LABELS)[oof.argmax(axis=1)]
        score = float(f1_score(truth, prediction, labels=list(LABELS), average="macro"))
        report = {
            "interpretation": "Developmental CV after prior experimentation; not an untouched holdout estimate or guarantee.",
            "validation": "five original object-component-disjoint folds",
            "spaces": [space.name for space in spaces],
            "params": {
                "C": c,
                "head": kind,
                "scaling": "StandardScaler fit separately inside each training fold (and head for factorized variant)",
                "class_weight": "balanced",
                "solver": "liblinear" if kind == "factorized" else "RBF SVC",
                "gamma": "scale" if kind == "flat_rbf" else None,
                "seed": 20260919,
                "max_iter": 2000 if kind == "factorized" else -1,
                "probability": True,
            },
            "fold_reports": fold_reports,
            "macro_f1_raw_oof": score,
            "fold_macro_f1_raw": {
                str(fold): float(
                    f1_score(
                        truth[fold_id == fold],
                        prediction[fold_id == fold],
                        labels=list(LABELS),
                        average="macro",
                    )
                )
                for fold in range(5)
            },
            "modality_macro_f1_feasible_labels": {
                pair_type: float(
                    f1_score(
                        truth[train["pair_type"].to_numpy() == pair_type],
                        prediction[train["pair_type"].to_numpy() == pair_type],
                        labels=list(LABELS if pair_type == "image+text" else LABELS[1:]),
                        average="macro",
                    )
                )
                for pair_type in PAIR_TYPES
            },
            "classification_report": classification_report(
                truth, prediction, labels=list(LABELS), output_dict=True, zero_division=0
            ),
            "runtime_s": time.monotonic() - variant_started,
            "calibration": "no external logit offsets; raw ontology-factorized probabilities"
            if kind == "factorized"
            else "libsvm internal probability calibration within outer train only; raw output",
            "feature_dimensions": {
                pair_type: int(values[2].shape[1]) for pair_type, values in features.items()
            },
        }
        np.save(output / "oof_probabilities_raw.npy", oof)
        np.save(output / "test_probabilities_raw.npy", test_proba)
        (output / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completed": str(output),
                    "macro_f1_raw_oof": score,
                    "runtime_s": round(report["runtime_s"], 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
