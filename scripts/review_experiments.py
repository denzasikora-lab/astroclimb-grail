#!/usr/bin/env python3
"""Сравнивает cosine-бейзлайны, CLIP-абляции и варианты OCR на общих разбиениях."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from astroclimb.constants import LABELS, PAIR_TYPES
from astroclimb.embedding_store import EncoderSpace
from astroclimb.features import l2_normalize
from astroclimb.integrity import assert_safe_input
from astroclimb.modeling import FactorizedRelationModel
from astroclimb.ocr import load_ocr_jsonl
from astroclimb.rich_similarity import RichSimilarityFeatures
from astroclimb.splits import component_stratified_folds, validate_disjoint_folds
from train_rich_nested import CONFIGS, make_estimator


RUNS = {
    "cosine_b16_logistic": ("logistic", ("clip_b16",), None),
    "cosine_b32_logistic": ("logistic", ("clip_b32",), None),
    "cosine_scincl_logistic": ("logistic", ("scincl",), "text+text"),
    "cosine_b16_scincl_logistic": ("logistic", ("clip_b16", "scincl"), None),
    "cosine_dual_scincl_logistic": ("logistic", ("clip_b16", "clip_b32", "scincl"), None),
    "cosine_dual_scincl_factorized7": ("factorized7", ("clip_b16", "clip_b32", "scincl"), None),
    "rich_full": ("factorized7", (), None),
    "rich_no_b16": ("factorized7", (), None),
    "rich_no_b32": ("factorized7", (), None),
    "rich_full_resolution_ocr": ("factorized7", (), None),
}


def cosine_matrix(meta: pd.DataFrame, spaces: list[EncoderSpace]) -> np.ndarray:
    """Вычисляет косинусные сходства без обучения статистик корпуса.

    Parameters
    ----------
    meta : pandas.DataFrame
        Типы и хеши двух объектов каждой пары.
    spaces : list[EncoderSpace]
        Пространства замороженных представлений в порядке столбцов результата.

    Returns
    -------
    numpy.ndarray
        Сходства формы (число пар, число пространств). Для недоступного
        пространства возвращается ноль; классификаторы разделены по модальности.
    """
    result = np.zeros((len(meta), len(spaces)), dtype=np.float32)
    for column, space in enumerate(spaces):
        available = [
            kind
            for kind, matrix in [("image", space.image), ("text", space.text)]
            if matrix is not None
        ]
        covered = meta["type_1"].isin(available) & meta["type_2"].isin(available)
        rows = np.flatnonzero(covered.to_numpy())
        if not len(rows):
            continue
        subset = meta.iloc[rows]
        left = space.side(subset["hash_1"].astype(str).to_numpy(), subset["type_1"].to_numpy())
        right = space.side(subset["hash_2"].astype(str).to_numpy(), subset["type_2"].to_numpy())
        result[rows, column] = np.einsum("ij,ij->i", l2_normalize(left), l2_normalize(right))
    return result


def feature_mask(names: list[str], experiment: str) -> np.ndarray:
    """Выбирает признаки, сохраняя остальные блоки при CLIP-абляции.

    Parameters
    ----------
    names : list[str]
        Имена признаков полного преобразования.
    experiment : str
        Имя эксперимента rich_full, rich_no_b16 или rich_no_b32.

    Returns
    -------
    numpy.ndarray
        Булева маска столбцов; удаляется только блок указанного CLIP.
    """
    prefixes = {"rich_no_b16": "emb.clip_b16_", "rich_no_b32": "emb.clip_b32_"}
    prefix = prefixes.get(experiment)
    return np.asarray([not (prefix and name.startswith(prefix)) for name in names], dtype=bool)


def estimator(kind: str):
    if kind == "factorized7":
        return make_estimator(CONFIGS["factorized7"])
    if kind != "logistic":
        raise ValueError(f"Неизвестный классификатор: {kind}")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=20260919,
        ),
    )


def evaluate(meta: pd.DataFrame, probability: np.ndarray, scope: str | None) -> dict:
    selected = (
        np.ones(len(meta), dtype=bool) if scope is None else meta["pair_type"].to_numpy() == scope
    )
    actual = probability[selected]
    if actual.shape != (int(selected.sum()), len(LABELS)) or not np.isfinite(actual).all():
        raise ValueError("Неверная форма или неконечные вероятности")
    if (actual < 0).any() or not np.allclose(actual.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Нарушена нормировка вероятностей")
    if (actual[meta.loc[selected, "pair_type"].to_numpy() != "image+text", 0] != 0).any():
        raise ValueError("Предсказан невозможный класс same_figure")
    labels = list(LABELS if scope in (None, "image+text") else LABELS[1:])
    truth = meta.loc[selected, "label"].astype(str).to_numpy()
    prediction = np.asarray(LABELS)[actual.argmax(axis=1)]
    return {
        "scope": scope or "all_modalities",
        "rows": len(truth),
        "metric_labels": labels,
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            truth, prediction, labels=labels, output_dict=True, zero_division=0
        ),
    }


def run(args: argparse.Namespace) -> None:
    root = args.project.resolve()
    meta = pd.read_parquet(root / "artifacts/audit/meta_train.parquet")
    truth = meta["label"].astype(str).to_numpy()
    active_modalities = [kind for kind in PAIR_TYPES if kind in set(meta.pair_type)]
    if not active_modalities or not meta.pair_type.isin(PAIR_TYPES).all():
        raise ValueError("Неверная или пустая область модальностей")
    if args.fold_ids is not None:
        ids = np.load(assert_safe_input(args.fold_ids))
        if ids.shape != (len(meta),) or set(ids.tolist()) != set(range(5)):
            raise ValueError("Сохраненные разбиения не соответствуют пяти внешним частям")
        folds = [(np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)) for fold in range(5)]
    else:
        folds = list(component_stratified_folds(meta, n_splits=5))
    validation = validate_disjoint_folds(meta, folds)
    fold_vector = np.empty(len(meta), dtype=np.int16)
    for fold, (_, valid) in enumerate(folds):
        fold_vector[valid] = fold
    rich_runs = [name for name in args.runs if name.startswith("rich_")]
    needed = (
        ("clip_b16", "clip_b32", "scincl", "layout")
        if rich_runs
        else sorted({space for name in args.runs for space in RUNS[name][1]})
    )
    spaces = {name: EncoderSpace.load(root / "artifacts/embeddings", name) for name in needed}
    embedding_hashes = {}
    for path in sorted((root / "artifacts/embeddings").glob("*.npy")):
        if not any(path.name.startswith(f"emb_{name}_") for name in needed):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        embedding_hashes[path.name] = digest.hexdigest()
    environment = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
    }
    provenance = {}
    for name in needed:
        path = root / "artifacts/embeddings" / f"provenance_{name}.json"
        if path.exists():
            provenance[name] = json.loads(path.read_text(encoding="utf-8"))
    texts, ocr, full_ocr = {}, {}, None
    if rich_runs:
        frame = pd.read_parquet(root / "artifacts/object_cache/texts.parquet")
        texts = dict(zip(frame["hash"].astype(str), frame["text"].astype(str)))
        ocr = load_ocr_jsonl(args.ocr_jsonl or root / "artifacts/ocr_fast.jsonl")
    if "rich_full_resolution_ocr" in args.runs:
        if args.full_resolution_ocr is None:
            raise ValueError("Для сравнения разрешений требуется --full-resolution-ocr")
        full_ocr = load_ocr_jsonl(args.full_resolution_ocr)
        if set(full_ocr) != set(ocr):
            raise ValueError("Два OCR-кеша должны покрывать одинаковые хеши изображений")
    cosine = {
        name: cosine_matrix(meta, [spaces[s] for s in config[1]])
        for name, config in RUNS.items()
        if name in args.runs and config[1]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        if (args.output_dir / name / "cv_report.json").exists():
            raise FileExistsError(f"Завершенный эксперимент уже существует: {name}")
    input_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            meta[["hash_1", "hash_2", "label"]],
            index=True,
        )
        .to_numpy()
        .tobytes()
    ).hexdigest()
    output = {name: np.zeros((len(meta), len(LABELS)), dtype=np.float32) for name in args.runs}
    details = {name: [] for name in args.runs}
    started = time.monotonic()
    for fold, (fit, valid) in enumerate(folds):
        fitting, heldout = meta.iloc[fit], meta.iloc[valid]
        rich = None
        if rich_runs:
            rich = RichSimilarityFeatures(texts, ocr, list(spaces.values())).fit(fitting)
            x_fit, x_valid = rich.transform(fitting), rich.transform(heldout)
            if len(rich.feature_names) != 190:
                raise ValueError("Полная модель должна иметь ровно 190 признаков")
        high = None
        if full_ocr is not None:
            high = RichSimilarityFeatures(texts, full_ocr, list(spaces.values())).fit(fitting)
            h_fit, h_valid = high.transform(fitting), high.transform(heldout)
        for name in args.runs:
            kind, _, scope = RUNS[name]
            if name in cosine:
                local_fit, local_valid = cosine[name][fit], cosine[name][valid]
                feature_names = [f"cosine.{space}" for space in RUNS[name][1]]
            elif name == "rich_full_resolution_ocr":
                local_fit, local_valid = h_fit, h_valid
                feature_names = high.feature_names
            else:
                mask = feature_mask(rich.feature_names, name)
                local_fit, local_valid = x_fit[:, mask], x_valid[:, mask]
                feature_names = [n for n, keep in zip(rich.feature_names, mask) if keep]
            for modality in active_modalities:
                if scope is not None and modality != scope:
                    continue
                train_mask = fitting["pair_type"].to_numpy() == modality
                valid_mask = heldout["pair_type"].to_numpy() == modality
                if not train_mask.any() or not valid_mask.any():
                    raise ValueError(f"В разбиении отсутствует модальность {modality}")
                model = FactorizedRelationModel.fit(
                    local_fit[train_mask],
                    truth[fit[train_mask]],
                    modality,
                    estimator_factory=lambda: estimator(kind),
                )
                output[name][valid[valid_mask]] = model.predict_proba(
                    local_valid[valid_mask], modality
                )
            metrics = evaluate(heldout, output[name][valid], scope)
            details[name].append(
                {"fold": fold, "macro_f1": metrics["macro_f1"], "features": feature_names}
            )
            print(
                json.dumps(
                    {
                        "experiment": name,
                        "fold": fold,
                        "macro_f1": metrics["macro_f1"],
                        "feature_count": len(feature_names),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {}
    for name in args.runs:
        directory = args.output_dir / name
        directory.mkdir(exist_ok=True)
        result = evaluate(meta, output[name], RUNS[name][2])
        report = {
            **result,
            "experiment": name,
            "estimator": RUNS[name][0],
            "folds": details[name],
            "validation_checks": validation,
            "input_fingerprint": input_fingerprint,
            "corpus_statistics": "Только уникальные объекты обучающей части каждого разбиения",
            "decision_offsets": False,
            "model_selection": False,
            "test_data_read": False,
            "interpretation": "Оценка разработки после предшествующих экспериментов на том же наборе",
            "elapsed_total_seconds": time.monotonic() - started,
            "included_pair_types": active_modalities,
            "per_modality": {
                modality: evaluate(meta, output[name], modality)
                for modality in active_modalities
                if RUNS[name][2] in (None, modality)
            },
            "fold_ids_sha256": hashlib.sha256(fold_vector.tobytes()).hexdigest(),
            "embedding_sha256": embedding_hashes,
            "embedding_provenance": provenance,
            "environment": environment,
        }
        np.save(directory / "oof_probabilities.npy", output[name])
        (directory / "cv_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary[name] = {key: result[key] for key in ("scope", "rows", "metric_labels", "macro_f1")}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-ids", type=Path)
    parser.add_argument("--ocr-jsonl", type=Path)
    parser.add_argument("--full-resolution-ocr", type=Path)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUNS),
        default=[n for n in RUNS if n != "rich_full_resolution_ocr"],
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads должен быть положительным")
    if args.full_resolution_ocr is not None and "rich_full_resolution_ocr" not in args.runs:
        args.runs.append("rich_full_resolution_ocr")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
