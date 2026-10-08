#!/usr/bin/env python3
"""Controlled factorized7 feature ablations on original object-disjoint folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from threadpoolctl import threadpool_limits

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.rich_similarity import RichSimilarityFeatures
from astroclimb.splits import component_stratified_folds, validate_disjoint_folds
from train_rich_nested import CONFIGS, fit_predict


ABLATIONS = ("no_ocr", "no_tfidf_rarity")


def metrics(meta: pd.DataFrame, probability: np.ndarray) -> dict:
    if probability.shape != (len(meta), len(LABELS)):
        raise ValueError("Wrong probability shape")
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite probability")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    if (probability[meta["pair_type"].to_numpy() != "image+text", 0] != 0).any():
        raise ValueError("Ontology mask violated")
    truth = meta["label"].astype(str).to_numpy()
    prediction = np.asarray(LABELS)[probability.argmax(axis=1)]
    return {
        "macro_f1_raw_oof": float(
            f1_score(truth, prediction, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            truth, prediction, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "modality_macro_f1_feasible_labels": {
            kind: float(
                f1_score(
                    truth[meta["pair_type"].to_numpy() == kind],
                    prediction[meta["pair_type"].to_numpy() == kind],
                    labels=list(LABELS if kind == "image+text" else LABELS[1:]),
                    average="macro",
                    zero_division=0,
                )
            )
            for kind in PAIR_TYPES
        },
    }


def compare(args: argparse.Namespace, meta: pd.DataFrame) -> None:
    if args.reference_oof is None or not args.reference_oof.is_file():
        print("Full-feature reference not yet available; comparison remains pending.", flush=True)
        return
    reference = metrics(meta, np.load(args.reference_oof))
    comparison = {
        "reference_oof": str(args.reference_oof),
        "reference_sha256": hashlib.sha256(args.reference_oof.read_bytes()).hexdigest(),
        "reference_metrics": reference,
        "model": "fixed factorized7, no offsets or routing",
        "interpretation": "Controlled developmental feature ablation. Scores are not an untouched holdout estimate.",
        "ablations": {},
    }
    for name in ABLATIONS:
        path = args.output_dir / name / "oof_probabilities_raw.npy"
        if not path.exists():
            continue
        ablated = metrics(meta, np.load(path))
        comparison["ablations"][name] = {
            "metrics": ablated,
            "full_minus_ablated_macro_f1": reference["macro_f1_raw_oof"]
            - ablated["macro_f1_raw_oof"],
            "full_minus_ablated_by_modality": {
                kind: reference["modality_macro_f1_feasible_labels"][kind]
                - ablated["modality_macro_f1_feasible_labels"][kind]
                for kind in PAIR_TYPES
            },
        }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)


def run(args: argparse.Namespace) -> None:
    artifacts = args.project / "artifacts"
    train = pd.read_parquet(artifacts / "audit/meta_train.parquet")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        compare(args, train)
        return
    truth = train["label"].astype(str).to_numpy()
    folds = list(component_stratified_folds(train, n_splits=5))
    validation = validate_disjoint_folds(train, folds)
    text = pd.read_parquet(artifacts / "object_cache/texts.parquet")
    texts = dict(zip(text["hash"].astype(str), text["text"].astype(str)))
    ocr = load_ocr_jsonl(artifacts / "ocr_fast.jsonl")
    spaces = [
        EncoderSpace.load(artifacts / "embeddings", name)
        for name in ("clip_b16", "clip_b32", "scincl", "layout")
    ]
    for name in ABLATIONS:
        output = args.output_dir / name
        output.mkdir(exist_ok=True)
        if (output / "cv_report.json").exists():
            raise FileExistsError(f"Completed ablation already exists: {output}")
        start = time.monotonic()
        oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
        details = []
        for fold, (fit, valid) in enumerate(folds):
            fold_started = time.monotonic()
            train_meta = train.iloc[fit].reset_index(drop=True)
            valid_meta = train.iloc[valid].reset_index(drop=True)
            features = RichSimilarityFeatures(texts, {} if name == "no_ocr" else ocr, spaces).fit(
                train_meta
            )
            selected = np.asarray(
                [
                    not (feature.startswith("tfidf.") or feature.startswith("text.idf_"))
                    if name == "no_tfidf_rarity"
                    else True
                    for feature in features.feature_names
                ]
            )
            x_train = features.transform(train_meta)[:, selected]
            x_valid = features.transform(valid_meta)[:, selected]
            for kind in PAIR_TYPES:
                local_fit = train_meta["pair_type"].to_numpy() == kind
                local_valid = valid_meta["pair_type"].to_numpy() == kind
                oof[valid[local_valid]] = fit_predict(
                    "factorized7",
                    x_train[local_fit],
                    truth[fit[local_fit]],
                    kind,
                    x_valid[local_valid],
                )[0]
            detail = {
                "fold": fold,
                "seconds": time.monotonic() - fold_started,
                "feature_count": int(selected.sum()),
                **metrics(train.iloc[valid], oof[valid]),
            }
            details.append(detail)
            print(
                json.dumps(
                    {
                        "ablation": name,
                        "fold": fold,
                        "macro_f1": detail["macro_f1_raw_oof"],
                        "seconds": round(detail["seconds"], 1),
                        "features": int(selected.sum()),
                    }
                ),
                flush=True,
            )
        report = {
            "ablation": name,
            "model": "factorized7",
            "config": CONFIGS["factorized7"],
            "fixed_parameters": {
                "learning_rate": 0.055,
                "min_samples_leaf": 35,
                "early_stopping": False,
                "random_state": 20260919,
            },
            "validation": "original five object-component-disjoint outer folds",
            "validation_checks": validation,
            "folds": details,
            "feature_names": [f for f, keep in zip(features.feature_names, selected) if keep],
            "dropped_feature_names": [
                f for f, keep in zip(features.feature_names, selected) if not keep
            ],
            "corpus_statistics": "Fitted only on each outer fitting fold's unique objects",
            "calibration": "none",
            "routing": "none",
            "inner_search": "none",
            "test_data_read": False,
            "test_labels_read": False,
            "interpretation": "Controlled developmental feature ablation after prior dataset experiments; not an untouched holdout.",
            "runtime_seconds": time.monotonic() - start,
            **metrics(train, oof),
        }
        np.save(output / "oof_probabilities_raw.npy", oof)
        (output / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completed": name,
                    "macro_f1_raw_oof": report["macro_f1_raw_oof"],
                    "seconds": round(report["runtime_seconds"], 1),
                }
            ),
            flush=True,
        )
    compare(args, train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
