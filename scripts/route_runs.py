#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.evaluation import group_bootstrap_macro_f1_ci
from astroclimb.graph_refinement import average_duplicate_pairs
from astroclimb.integrity import assert_safe_input
from astroclimb.splits import component_stratified_folds, object_component_groups
from astroclimb.submission import hard_submission, write_submission


def load_probabilities(run_dirs: list[Path], filename: str, rows: int) -> list[np.ndarray]:
    outputs = []
    for directory in run_dirs:
        values = np.load(directory / filename)
        if values.shape != (rows, len(LABELS)):
            raise ValueError(f"Unexpected shape {values.shape} in {directory / filename}")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid probabilities in {directory / filename}")
        outputs.append(values.astype(np.float64))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit selection of a source run independently for each modality."
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise ValueError("At least two run directories are required")
    sample_path = assert_safe_input(args.sample)
    train = pd.read_parquet(args.audit_dir / "meta_train.parquet")
    test = pd.read_parquet(args.audit_dir / "meta_test.parquet")
    if test.get("label", pd.Series(dtype=object)).notna().any():
        raise ValueError("Test metadata unexpectedly contains labels")
    oof_runs = load_probabilities(args.run_dirs, "oof_probabilities.npy", len(train))
    test_runs = load_probabilities(args.run_dirs, "test_probabilities.npy", len(test))
    truth = train["label"].astype(str).to_numpy()
    names = np.asarray(LABELS)
    fold_id = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, valid_index) in enumerate(component_stratified_folds(train, n_splits=args.folds)):
        fold_id[valid_index] = fold

    crossfit = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    fold_choices: dict[str, dict[str, object]] = {}
    pair_types = train["pair_type"].astype(str).to_numpy()
    for fold in range(args.folds):
        fold_choices[str(fold)] = {}
        for pair_type in PAIR_TYPES:
            tuning = (fold_id != fold) & (pair_types == pair_type)
            validation = (fold_id == fold) & (pair_types == pair_type)
            labels = list(LABELS) if pair_type == "image+text" else list(LABELS[1:])
            scores = [
                f1_score(
                    truth[tuning],
                    names[probabilities[tuning].argmax(1)],
                    labels=labels,
                    average="macro",
                )
                for probabilities in oof_runs
            ]
            selected = int(np.argmax(scores))
            crossfit[validation] = oof_runs[selected][validation]
            fold_choices[str(fold)][pair_type] = {
                "selected_run": str(args.run_dirs[selected]),
                "tuning_scores": scores,
                "validation_rows": int(validation.sum()),
            }

    final_test = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    final_choices: dict[str, object] = {}
    test_pair_types = test["pair_type"].astype(str).to_numpy()
    for pair_type in PAIR_TYPES:
        train_mask = pair_types == pair_type
        test_mask = test_pair_types == pair_type
        labels = list(LABELS) if pair_type == "image+text" else list(LABELS[1:])
        scores = [
            f1_score(
                truth[train_mask],
                names[probabilities[train_mask].argmax(1)],
                labels=labels,
                average="macro",
            )
            for probabilities in oof_runs
        ]
        selected = int(np.argmax(scores))
        final_test[test_mask] = test_runs[selected][test_mask]
        final_choices[pair_type] = {
            "selected_run": str(args.run_dirs[selected]),
            "oof_scores": scores,
        }

    final_test = average_duplicate_pairs(test, final_test)
    prediction = names[crossfit.argmax(1)]
    score = f1_score(truth, prediction, labels=list(LABELS), average="macro")
    bootstrap = group_bootstrap_macro_f1_ci(
        truth,
        prediction,
        object_component_groups(train),
        LABELS,
        1000,
        20260919,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "validation": "cross-fit modality routing on object-component folds",
        "source_runs": [str(path) for path in args.run_dirs],
        "macro_f1_crossfit": score,
        "group_bootstrap_95ci": {
            "point": bootstrap[0],
            "lower": bootstrap[1],
            "upper": bootstrap[2],
            "resamples": 1000,
        },
        "fold_choices": fold_choices,
        "final_choices": final_choices,
    }
    (args.output_dir / "route_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "oof_probabilities.npy", crossfit.astype(np.float32))
    np.save(args.output_dir / "test_probabilities.npy", final_test.astype(np.float32))
    submission = hard_submission(test["id"].to_numpy(), final_test, sample_path)
    write_submission(submission, sample_path, args.output_dir / "submission.csv")
    print(f"Cross-fit routed OOF macro-F1: {score:.5f}")
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
