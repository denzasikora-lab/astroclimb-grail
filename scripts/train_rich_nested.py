#!/usr/bin/env python3
"""Nested object-disjoint model selection and calibration of compact pair features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from threadpoolctl import threadpool_limits

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace
from astroclimb.evaluation import group_bootstrap_macro_f1_ci
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.modeling import (
    FactorizedRelationModel,
    apply_logit_offsets,
    optimize_macro_f1_offsets,
)
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.rich_similarity import RichSimilarityFeatures
from astroclimb.splits import (
    component_stratified_folds,
    object_component_groups,
    validate_disjoint_folds,
)
from astroclimb.submission import hard_submission, write_submission


CONFIGS = {
    "hgb7": {"leaves": 7, "iterations": 180, "l2": 10.0, "factorized": False},
    "hgb15": {"leaves": 15, "iterations": 180, "l2": 15.0, "factorized": False},
    "factorized7": {"leaves": 7, "iterations": 180, "l2": 10.0, "factorized": True},
}


def score(truth: np.ndarray, probability: np.ndarray, labels=LABELS) -> float:
    return float(
        f1_score(
            truth,
            np.asarray(LABELS)[probability.argmax(1)],
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def make_estimator(config: dict):
    return HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=config["iterations"],
        max_leaf_nodes=config["leaves"],
        min_samples_leaf=35,
        l2_regularization=config["l2"],
        early_stopping=False,
        random_state=20260919,
    )


def fit_predict(name, x_train, truth, pair_type, *targets):
    config = CONFIGS[name]
    if config["factorized"]:
        model = FactorizedRelationModel.fit(
            x_train, truth, pair_type, estimator_factory=lambda: make_estimator(config)
        )
        return [model.predict_proba(x, pair_type) for x in targets]
    model = make_estimator(config).fit(x_train, truth)
    result = []
    for x in targets:
        probability = np.zeros((len(x), len(LABELS)), dtype=np.float64)
        local_probability = model.predict_proba(x)
        for column, label in enumerate(model.classes_):
            probability[:, LABELS.index(str(label))] = local_probability[:, column]
        result.append(probability)
    return result


def masked_offsets(probability, offsets, meta):
    result = apply_logit_offsets(probability, offsets)
    result[meta["pair_type"].to_numpy() != "image+text", 0] = 0.0
    return result / result.sum(axis=1, keepdims=True)


def train(args):
    started = time.monotonic()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "cv_report.json").exists():
        raise FileExistsError(f"Completed run already exists: {output}")
    root = args.project
    train_meta = pd.read_parquet(root / "artifacts/audit/meta_train.parquet")
    test_meta = pd.read_parquet(root / "artifacts/audit/meta_test.parquet")
    if test_meta.get("label", pd.Series(dtype=object)).notna().any():
        raise ValueError("Test metadata unexpectedly has labels")
    text_frame = pd.read_parquet(root / "artifacts/object_cache/texts.parquet")
    text_lookup = dict(zip(text_frame["hash"].astype(str), text_frame["text"].astype(str)))
    ocr_lookup = load_ocr_jsonl(root / "artifacts/ocr_fast.jsonl")
    spaces = [
        EncoderSpace.load(root / "artifacts/embeddings", name)
        for name in ("clip_b16", "clip_b32", "scincl", "layout")
    ]
    truth = train_meta["label"].astype(str).to_numpy()
    outer_folds = list(component_stratified_folds(train_meta, n_splits=5))
    validation = validate_disjoint_folds(train_meta, outer_folds)
    oof = np.zeros((len(train_meta), 4), dtype=np.float64)
    test_prob = np.zeros((len(test_meta), 4), dtype=np.float64)
    fixed_oof = {name: np.zeros_like(oof) for name in args.models}
    fixed_test = {name: np.zeros_like(test_prob) for name in args.models}
    fold_ids = np.full(len(train_meta), -1, dtype=np.int16)
    details = []
    protocol = {
        "outer_folds": 5,
        "inner_folds": 3,
        "models": {n: CONFIGS[n] for n in args.models},
        "selection": "inner OOF feasible-class macro-F1 per modality",
        "calibration": "class offsets fitted only to selected inner OOF predictions",
        "test_prediction": "mean of five outer models with their inner-selected offsets",
        "idf_and_embedding_statistics": "fitted exclusively on each fitting fold's objects",
        "solution_csv_read": False,
        "test_labels_read": False,
        "interpretation": "development validation with nested selection; dataset was used in earlier research",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for fold, (fit_index, valid_index) in enumerate(outer_folds):
        fold_started = time.monotonic()
        outer_train = train_meta.iloc[fit_index].reset_index(drop=True)
        outer_valid = train_meta.iloc[valid_index].reset_index(drop=True)
        inner_truth = truth[fit_index]
        inner = list(component_stratified_folds(outer_train, n_splits=3, random_state=20260920))
        inner_checks = validate_disjoint_folds(outer_train, inner)
        inner_prob = {name: np.zeros((len(outer_train), 4)) for name in args.models}
        for inner_id, (inner_fit, inner_valid) in enumerate(inner):
            features = RichSimilarityFeatures(text_lookup, ocr_lookup, spaces)
            features.fit(outer_train.iloc[inner_fit])
            x_fit = features.transform(outer_train.iloc[inner_fit])
            x_valid = features.transform(outer_train.iloc[inner_valid])
            fit_types = outer_train.iloc[inner_fit]["pair_type"].to_numpy()
            valid_types = outer_train.iloc[inner_valid]["pair_type"].to_numpy()
            for pair_type in PAIR_TYPES:
                f = fit_types == pair_type
                v = valid_types == pair_type
                for name in args.models:
                    inner_prob[name][inner_valid[v]] = fit_predict(
                        name, x_fit[f], inner_truth[inner_fit[f]], pair_type, x_valid[v]
                    )[0]
            print(
                f"outer={fold} inner={inner_id} rows={len(inner_fit)} "
                f"features={x_fit.shape[1]} elapsed={time.monotonic() - fold_started:.1f}s",
                flush=True,
            )
        routed_inner = np.zeros((len(outer_train), 4))
        choices, scores = {}, {}
        outer_types = outer_train["pair_type"].to_numpy()
        for pair_type in PAIR_TYPES:
            mask = outer_types == pair_type
            labels = LABELS if pair_type == "image+text" else LABELS[1:]
            scores[pair_type] = {
                name: score(inner_truth[mask], p[mask], labels) for name, p in inner_prob.items()
            }
            choices[pair_type] = max(scores[pair_type], key=scores[pair_type].get)
            routed_inner[mask] = inner_prob[choices[pair_type]][mask]
        offsets = optimize_macro_f1_offsets(routed_inner, inner_truth)
        features = RichSimilarityFeatures(text_lookup, ocr_lookup, spaces).fit(outer_train)
        x_train, x_valid, x_test = [
            features.transform(m) for m in (outer_train, outer_valid, test_meta)
        ]
        candidate_valid = {name: np.zeros((len(outer_valid), 4)) for name in args.models}
        candidate_test = {name: np.zeros_like(test_prob) for name in args.models}
        for pair_type in PAIR_TYPES:
            f = outer_types == pair_type
            v = outer_valid["pair_type"].to_numpy() == pair_type
            t = test_meta["pair_type"].to_numpy() == pair_type
            for name in args.models:
                vp, tp = fit_predict(
                    name, x_train[f], inner_truth[f], pair_type, x_valid[v], x_test[t]
                )
                candidate_valid[name][v] = vp
                candidate_test[name][t] = tp
        routed_valid, routed_test = np.zeros((len(outer_valid), 4)), np.zeros_like(test_prob)
        for pair_type, name in choices.items():
            v = outer_valid["pair_type"].to_numpy() == pair_type
            t = test_meta["pair_type"].to_numpy() == pair_type
            routed_valid[v] = candidate_valid[name][v]
            routed_test[t] = candidate_test[name][t]
        oof[valid_index] = masked_offsets(routed_valid, offsets, outer_valid)
        fold_test = masked_offsets(routed_test, offsets, test_meta)
        test_prob += fold_test / 5
        fold_ids[valid_index] = fold
        for name in args.models:
            fixed_oof[name][valid_index] = candidate_valid[name]
            fixed_test[name] += candidate_test[name] / 5
        detail = {
            "fold": fold,
            "choices": choices,
            "inner_scores": scores,
            "offsets": dict(zip(LABELS, offsets.tolist())),
            "inner_disjoint_checks": inner_checks,
            "macro_f1": score(truth[valid_index], oof[valid_index]),
            "fixed_macro_f1": {n: score(truth[valid_index], p) for n, p in candidate_valid.items()},
            "elapsed_seconds": time.monotonic() - fold_started,
        }
        details.append(detail)
        np.savez_compressed(
            output / f"fold_{fold}.npz",
            valid_indices=valid_index,
            oof=oof[valid_index],
            test=fold_test,
            inner_indices=fit_index,
            inner_selected=routed_inner,
        )
        (output / f"fold_{fold}.json").write_text(json.dumps(detail, indent=2))
        print(json.dumps(detail, separators=(",", ":")), flush=True)
    oof = average_duplicate_pairs(train_meta, oof)
    test_prob = average_duplicate_pairs(test_meta, test_prob)
    for name in args.models:
        np.save(output / f"oof_{name}_raw.npy", fixed_oof[name].astype(np.float32))
        np.save(output / f"test_{name}_raw.npy", fixed_test[name].astype(np.float32))
    np.save(output / "oof_probabilities.npy", oof.astype(np.float32))
    np.save(output / "test_probabilities.npy", test_prob.astype(np.float32))
    np.save(output / "fold_ids.npy", fold_ids)
    ci = group_bootstrap_macro_f1_ci(
        truth,
        np.asarray(LABELS)[oof.argmax(1)],
        object_component_groups(train_meta),
        LABELS,
        1000,
        20260919,
    )
    report = {
        "protocol": protocol,
        "validation_checks": validation,
        "folds": details,
        "macro_f1_nested": score(truth, oof),
        "fixed_raw_macro_f1": {n: score(truth, p) for n, p in fixed_oof.items()},
        "macro_f1_group_bootstrap_95ci": {"point": ci[0], "lower": ci[1], "upper": ci[2]},
        "classification_report": classification_report(
            truth, np.asarray(LABELS)[oof.argmax(1)], labels=list(LABELS), output_dict=True
        ),
        "feature_names": features.feature_names,
        "elapsed_seconds": time.monotonic() - started,
    }
    sample = root / "astroclimb/sample_submission.csv"
    write_submission(
        hard_submission(test_meta["id"].to_numpy(), test_prob, sample),
        sample,
        output / "submission.csv",
    )
    if "factorized7" in fixed_test:
        write_submission(
            hard_submission(
                test_meta["id"].to_numpy(),
                np.load(output / "test_factorized7_raw.npy"),
                sample,
            ),
            sample,
            output / "submission_factorized7_raw.csv",
        )
    report["submission_sha256"] = hashlib.sha256(
        (output / "submission.csv").read_bytes()
    ).hexdigest()
    (output / "cv_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"FINAL nested macro-F1: {report['macro_f1_nested']:.6f}; CI {ci}; "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        train(args)


if __name__ == "__main__":
    main()
