"""Регрессионные проверки GRAIL, собранные в одном модуле."""

from __future__ import annotations


# Проверки из embedding_store

import numpy as np
import pandas as pd

from astroclimb.embedding_store import EncoderSpace, pair_feature_matrix


def test_mixed_pair_uses_joint_and_single_modality_spaces_without_order_leakage() -> None:
    meta = pd.DataFrame(
        {
            "pair_type": ["image+text", "image+text"],
            "type_1": ["image", "text"],
            "type_2": ["text", "image"],
            "hash_1": ["i", "t"],
            "hash_2": ["t", "i"],
            "length_1": [10, 20],
            "length_2": [20, 10],
        }
    )
    joint = EncoderSpace(
        "joint",
        image=np.asarray([[1.0, 0.0]]),
        text=np.asarray([[0.0, 1.0]]),
        image_lookup={"i": 0},
        text_lookup={"t": 0},
    )
    vision = EncoderSpace(
        "vision",
        image=np.asarray([[2.0, 3.0]]),
        text=None,
        image_lookup={"i": 0},
        text_lookup={},
    )
    text = EncoderSpace(
        "text",
        image=None,
        text=np.asarray([[4.0, 5.0, 6.0]]),
        image_lookup={},
        text_lookup={"t": 0},
    )
    features = pair_feature_matrix(meta, [joint, vision, text])
    np.testing.assert_allclose(features[0], features[1])
    assert features.shape[1] == 4 + 2 + 2 + 3 + 3


# Проверки из evaluation

import numpy as np
import pytest
from sklearn.metrics import f1_score

from astroclimb.evaluation import group_bootstrap_macro_f1_ci


def test_group_bootstrap_matches_manual_group_resampling() -> None:
    truth = np.asarray(["a", "b", "a", "a", "b", "b", "a"])
    predicted = np.asarray(["a", "a", "a", "b", "b", "b", "b"])
    groups = np.asarray([10, 10, 20, 20, 20, 30, 30])
    labels = ["a", "b"]
    n_resamples = 31
    seed = 17

    point, lower, upper = group_bootstrap_macro_f1_ci(
        truth, predicted, groups, labels, n_resamples, seed
    )

    expected_point = f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
    rows_by_group = [np.flatnonzero(groups == group) for group in (10, 20, 30)]
    rng = np.random.default_rng(seed)
    expected_scores = []
    for _ in range(n_resamples):
        sampled_groups = rng.integers(0, len(rows_by_group), size=len(rows_by_group))
        sampled_rows = np.concatenate([rows_by_group[index] for index in sampled_groups])
        expected_scores.append(
            f1_score(
                truth[sampled_rows],
                predicted[sampled_rows],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    expected_lower, expected_upper = np.percentile(expected_scores, [2.5, 97.5])

    assert point == pytest.approx(expected_point)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_group_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    arguments = (
        ["a", "a", "b", "b", "a", "b"],
        ["a", "b", "b", "a", "a", "b"],
        ["g1", "g1", "g2", "g2", "g3", "g3"],
        ["a", "b"],
        100,
        20260919,
    )

    assert group_bootstrap_macro_f1_ci(*arguments) == group_bootstrap_macro_f1_ci(*arguments)


def test_group_bootstrap_perfect_predictions_have_degenerate_interval() -> None:
    truth = ["a", "b", "a", "b"]
    result = group_bootstrap_macro_f1_ci(truth, truth, [1, 1, 2, 2], ["a", "b"], 50, 5)

    assert result == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("truth", "predicted", "groups", "labels", "n_resamples", "message"),
    [
        (["a"], [], [1], ["a"], 10, "equal lengths"),
        ([], [], [], ["a"], 10, "At least one observation"),
        (["a"], ["a"], [1], [], 10, "At least one label"),
        (["a"], ["a"], [1], ["a"], 0, "positive"),
    ],
)
def test_group_bootstrap_rejects_invalid_inputs(
    truth: list[str],
    predicted: list[str],
    groups: list[int],
    labels: list[str],
    n_resamples: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_bootstrap_macro_f1_ci(truth, predicted, groups, labels, n_resamples, seed=1)


# Проверки из features

import numpy as np
import pandas as pd

from astroclimb.features import (
    FactorizedProbabilities,
    lexical_feature_matrix,
    lexical_pair_features,
    multimodal_ocr_feature_matrix,
    symmetric_embedding_features,
)


def test_symmetric_features_are_order_invariant() -> None:
    left = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    right = np.asarray([[4.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        symmetric_embedding_features(left, right),
        symmetric_embedding_features(right, left),
    )


def test_factorized_probabilities_sum_to_one_and_mask_same_figure() -> None:
    factorized = FactorizedProbabilities(
        same_figure=np.asarray([0.7, 0.7]),
        same_paper_given_not_figure=np.asarray([0.5, 0.5]),
        related_given_different_paper=np.asarray([0.25, 0.25]),
    )
    probabilities = factorized.leaf_probabilities(pair_type=np.asarray(["image+text", "text+text"]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[1, 0] == 0.0


def test_lexical_pair_features_are_symmetric() -> None:
    left_right = lexical_pair_features("M31 flux 42 km/s", "Flux of M31 is 42 km/s")
    right_left = lexical_pair_features("Flux of M31 is 42 km/s", "M31 flux 42 km/s")
    np.testing.assert_allclose(left_right, right_left)


def test_lexical_feature_matrix_only_uses_text_pairs() -> None:
    meta = pd.DataFrame(
        {
            "type_1": ["text", "image"],
            "type_2": ["text", "text"],
            "hash_1": ["a", "img"],
            "hash_2": ["b", "b"],
        }
    )
    values = lexical_feature_matrix(meta, {"a": "M31 flux 42", "b": "M31 flux 42"})
    assert values.shape == (2, 8)
    assert values[0, 3] == 1.0
    np.testing.assert_array_equal(values[1], np.zeros(8))


def test_multimodal_ocr_features_are_swap_invariant() -> None:
    meta = pd.DataFrame(
        {
            "type_1": ["image", "text"],
            "type_2": ["text", "image"],
            "hash_1": ["i", "t"],
            "hash_2": ["t", "i"],
        }
    )
    values = multimodal_ocr_feature_matrix(
        meta,
        text_lookup={"t": "Temperature 42 K"},
        image_text_lookup={"i": "Temperature 42K"},
    )
    np.testing.assert_allclose(values[0], values[1])
    assert values[0, 8] > 0


# Проверки из graph_refinement

import numpy as np
import pandas as pd

from astroclimb.graph_refinement import average_duplicate_pairs, conservative_same_paper_closure


def test_duplicate_pairs_are_averaged_without_direction() -> None:
    meta = pd.DataFrame(
        {
            "hash_1": ["a", "b", "x"],
            "hash_2": ["b", "a", "y"],
        }
    )
    probabilities = np.asarray([[0.0, 0.8, 0.1, 0.1], [0.0, 0.2, 0.3, 0.5], [0.0, 0.1, 0.2, 0.7]])
    refined = average_duplicate_pairs(meta, probabilities)
    np.testing.assert_allclose(refined[0], refined[1])
    np.testing.assert_allclose(refined[2], probabilities[2])


def test_high_confidence_paper_edges_create_conservative_closure() -> None:
    meta = pd.DataFrame({"hash_1": ["a", "b", "a"], "hash_2": ["b", "c", "c"]})
    probabilities = np.asarray(
        [[0.0, 0.97, 0.02, 0.01], [0.0, 0.98, 0.01, 0.01], [0.0, 0.3, 0.4, 0.3]]
    )
    refined = conservative_same_paper_closure(meta, probabilities)
    assert refined[2, :2].sum() >= 0.75
    np.testing.assert_allclose(refined.sum(axis=1), 1.0)


# Проверки из integrity

import numpy as np
import pandas as pd
import pytest

from astroclimb.constants import LABELS
from astroclimb.integrity import LeakageGuardError, assert_safe_input, validate_one_hot


def test_solution_is_quarantined() -> None:
    with pytest.raises(LeakageGuardError):
        assert_safe_input("/tmp/solution.csv")


def test_one_hot_validation() -> None:
    valid = pd.DataFrame(np.eye(4, dtype=int), columns=LABELS)
    validate_one_hot(valid)
    invalid = valid.copy()
    invalid.loc[0, "same_paper"] = 1
    with pytest.raises(ValueError):
        validate_one_hot(invalid)


# Проверки из modeling

import numpy as np

from astroclimb.modeling import apply_logit_offsets, crossfit_logit_offsets


def test_logit_offsets_preserve_probability_rows() -> None:
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
    shifted = apply_logit_offsets(probabilities, np.asarray([0.1, -0.2, 0.0, 0.3]))
    np.testing.assert_allclose(shifted.sum(axis=1), 1.0)
    assert (shifted > 0).all()


def test_crossfit_offsets_preserve_rows_and_cover_every_fold() -> None:
    probabilities = np.tile(np.asarray([[0.55, 0.15, 0.15, 0.15]]), (8, 1))
    labels = np.asarray(["same_figure", "same_paper", "related_papers", "unrelated_papers"] * 2)
    fold_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    calibrated, offsets = crossfit_logit_offsets(probabilities, labels, fold_ids)
    assert set(offsets) == {0, 1}
    assert calibrated.shape == probabilities.shape
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)


# Проверки из nested_training

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_runner():
    path = Path(__file__).parents[1] / "scripts" / "train_rich_nested.py"
    spec = importlib.util.spec_from_file_location("rich_nested_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_cannot_create_impossible_class():
    runner = load_runner()
    meta = pd.DataFrame({"pair_type": ["image+image", "text+text", "image+text"]})
    p = np.array([[0, 0.3, 0.4, 0.3], [0, 0.5, 0.2, 0.3], [0.1, 0.3, 0.3, 0.3]])
    calibrated = runner.masked_offsets(p, np.array([100.0, 0, 0, 0]), meta)
    np.testing.assert_array_equal(calibrated[:2, 0], [0.0, 0.0])
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    assert calibrated[2, 0] > 0.99


def test_flat_probability_columns_follow_competition_label_order(monkeypatch):
    runner = load_runner()

    class FakeClassifier:
        classes_ = ["related_papers", "same_paper", "unrelated_papers"]

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.tile([0.2, 0.7, 0.1], (len(x), 1))

    monkeypatch.setattr(runner, "make_estimator", lambda _: FakeClassifier())
    x = np.zeros((3, 1))
    result = runner.fit_predict("hgb7", x, np.array(FakeClassifier.classes_), "text+text", x)[0]
    np.testing.assert_allclose(result, np.tile([0, 0.7, 0.2, 0.1], (3, 1)))


# Проверки из rich_similarity

import numpy as np
import pandas as pd
import pytest

from astroclimb.embedding_store import EncoderSpace
from astroclimb.rich_similarity import RichSimilarityFeatures


def _meta(rows):
    return pd.DataFrame(rows, columns=["type_1", "hash_1", "type_2", "hash_2"])


def _spaces():
    return [
        EncoderSpace(
            "joint",
            np.asarray([[1.0, 2, -1], [0.0, 1, 0]]),
            np.asarray([[1.0, 1, -1], [1.0, 0, 0]]),
            {"i": 0, "j": 1},
            {"a": 0, "b": 1},
        ),
        EncoderSpace("text", None, np.asarray([[0.0, 0], [1.0, 2]]), {}, {"a": 0, "b": 1}),
    ]


def test_features_are_symmetric_and_finite_with_partial_spaces():
    meta = _meta(
        [("image", "i", "text", "a"), ("text", "a", "text", "b"), ("image", "i", "image", "j")]
    )
    swapped = meta.rename(
        columns={"type_1": "type_2", "type_2": "type_1", "hash_1": "hash_2", "hash_2": "hash_1"}
    )
    model = RichSimilarityFeatures(
        {"a": "HST flux of NGC-123 is 42", "b": "NGC-123 flux 42"},
        {"i": "HST NGC-123 42"},
        _spaces(),
        batch_size=2,
    ).fit(meta)
    actual = model.transform(meta)
    np.testing.assert_allclose(actual, model.transform(swapped), atol=1e-6)
    assert actual.dtype == np.float32
    assert actual.shape == (3, len(model.feature_names))
    assert np.isfinite(actual).all()
    assert actual[0, model.feature_names.index("emb.text_available_sides")] == 1
    assert actual[0, model.feature_names.index("emb.text_cosine")] == 0
    assert actual[0, model.feature_names.index("text.identifier_jaccard")] == 1


def test_fit_uses_only_unique_training_objects_and_never_labels():
    train = _meta([("text", "a", "text", "b")] * 5)
    heldout = _meta([("text", "secret", "text", "a")])
    lookup = {"a": "galaxy HST", "b": "galaxy velocity", "secret": "ZEBRAHIDDEN"}
    model = RichSimilarityFeatures(lookup, {}, []).fit(train.assign(same_paper=1))
    assert model.fit_object_keys_ == frozenset({("text", "a"), ("text", "b")})
    assert model.token_idf_["galaxy"] == 1
    assert "zebrahidden" not in model.vectorizers_["word"].vocabulary_
    assert ("text", "secret") not in model._documents
    before = dict(model.token_idf_)
    model.transform(heldout)
    assert model.token_idf_ == before
    assert "zebrahidden" not in model.vectorizers_["word"].vocabulary_
    duplicate_free = RichSimilarityFeatures(lookup, {}, []).fit(train.iloc[:1])
    np.testing.assert_allclose(model.transform(train), duplicate_free.transform(train))


def test_embedding_centering_uses_only_unique_training_hashes():
    train = _meta([("text", "a", "text", "a")] * 3)
    space = EncoderSpace("t", None, np.asarray([[1.0, 0], [-1000.0, 1]]), {}, {"a": 0, "b": 1})
    model = RichSimilarityFeatures({"a": "flux", "b": "unseen"}, {}, [space]).fit(train)
    mean, _ = model.embedding_statistics_["t"]
    np.testing.assert_array_equal(mean, [1.0, 0])


def test_all_missing_ocr_and_empty_transform_are_supported():
    meta = _meta([("image", "i", "image", "j")])
    model = RichSimilarityFeatures({}, {}, []).fit(meta)
    assert all(vectorizer is None for vectorizer in model.vectorizers_.values())
    result = model.transform(meta)
    assert np.isfinite(result).all()
    assert result[0, model.feature_names.index("text.exact")] == 0
    assert model.transform(meta.iloc[:0]).shape == (0, len(model.feature_names))


def test_missing_caption_and_invalid_modality_fail_loudly():
    with pytest.raises(KeyError):
        RichSimilarityFeatures({}, {}, []).fit(_meta([("text", "a", "image", "i")]))
    with pytest.raises(ValueError, match="Object types"):
        RichSimilarityFeatures({}, {}, []).fit(_meta([("audio", "a", "image", "i")]))
    with pytest.raises(RuntimeError, match="fit"):
        RichSimilarityFeatures({}, {}, []).transform(_meta([]))


def test_refitting_discards_previous_corpus_statistics():
    model = RichSimilarityFeatures({"a": "unseen zebra", "b": "galaxy HST"}, {}, [])
    model.fit(_meta([("text", "a", "text", "a")]))
    model.fit(_meta([("text", "b", "text", "b")]))
    assert "zebra" not in model.token_idf_
    assert "zebra" not in model.vectorizers_["word"].vocabulary_


def test_raw_lengths_and_layout_coordinates_are_symmetric_and_preserved():
    meta = _meta(
        [("image", "i", "text", "a"), ("image", "i", "image", "j"), ("text", "a", "text", "a")]
    )
    meta["length_1"] = [1000, 1000, 100]
    meta["length_2"] = [100, 2000, 100]
    swapped = meta.rename(
        columns={
            "type_1": "type_2",
            "type_2": "type_1",
            "hash_1": "hash_2",
            "hash_2": "hash_1",
            "length_1": "length_2",
            "length_2": "length_1",
        }
    )
    vectors = np.arange(66, dtype=np.float32).reshape(2, 33)
    layout = EncoderSpace("layout", vectors, None, {"i": 0, "j": 1}, {})
    model = RichSimilarityFeatures({"a": "caption"}, {}, [layout]).fit(meta)
    values = model.transform(meta)
    np.testing.assert_allclose(values, model.transform(swapped), atol=1e-6)
    begin = model.feature_names.index("layout.absdiff_0")
    np.testing.assert_allclose(values[0, begin : begin + 33], vectors[0])
    np.testing.assert_allclose(values[0, begin + 33 : begin + 66], vectors[0] * 0.5)
    np.testing.assert_allclose(values[1, begin : begin + 33], np.abs(vectors[0] - vectors[1]))
    np.testing.assert_array_equal(values[2, begin : begin + 66], np.zeros(66))
    assert values[0, model.feature_names.index("raw.length_ratio")] == 10
    np.testing.assert_allclose(
        values[0, model.feature_names.index("raw.image_length_log_mean")], np.log1p(1000)
    )
    np.testing.assert_allclose(
        values[0, model.feature_names.index("raw.text_length_log_mean")], np.log1p(100)
    )
    missing = model.transform(meta.drop(columns=["length_1", "length_2"]))
    first = model.feature_names.index("raw.length_log_min")
    np.testing.assert_array_equal(missing[:, first : first + 5], np.zeros((3, 5)))


# Проверки из splits

import pandas as pd

from astroclimb.splits import object_component_groups


def test_connected_objects_share_group() -> None:
    meta = pd.DataFrame(
        {
            "hash_1": ["a", "b", "d"],
            "hash_2": ["b", "c", "e"],
            "label": ["same_paper", "related_papers", "unrelated_papers"],
        }
    )
    groups = object_component_groups(meta)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


# Проверки из visual

import numpy as np
from PIL import Image

from astroclimb.visual import image_layout_descriptor


def test_layout_descriptor_is_finite_and_detects_white_background() -> None:
    white = Image.new("RGB", (40, 20), "white")
    descriptor = image_layout_descriptor(white)
    assert descriptor.shape == (33,)
    assert np.isfinite(descriptor).all()
    assert descriptor[5] == 1.0
    assert descriptor[6] == 0.0


def _load_review_runner():
    import sys

    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "review_runner", scripts / "review_experiments.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_review_cosines_preserve_symmetry_and_partial_modality_coverage():
    runner = _load_review_runner()
    meta = _meta(
        [
            ("image", "i", "text", "a"),
            ("text", "a", "image", "i"),
            ("text", "a", "text", "b"),
            ("image", "i", "image", "j"),
        ]
    )
    joint = EncoderSpace(
        "clip_b16",
        np.asarray([[1, 0], [0, 1]]),
        np.asarray([[0.6, 0.8], [1, 0]]),
        {"i": 0, "j": 1},
        {"a": 0, "b": 1},
    )
    text_space = EncoderSpace(
        "scincl", None, np.asarray([[1, 0], [0.6, 0.8]]), {}, {"a": 0, "b": 1}
    )
    actual = runner.cosine_matrix(meta, [joint, text_space])
    np.testing.assert_allclose(actual, [[0.6, 0], [0.6, 0], [0.6, 0.6], [0, 0]], atol=1e-6)
    swapped = meta.rename(
        columns={"type_1": "type_2", "type_2": "type_1", "hash_1": "hash_2", "hash_2": "hash_1"}
    )
    np.testing.assert_allclose(
        actual, runner.cosine_matrix(swapped, [joint, text_space]), atol=1e-6
    )


def test_review_clip_ablation_removes_only_seventeen_encoder_features():
    runner = _load_review_runner()
    spaces = [
        EncoderSpace(name, None, None, {}, {})
        for name in ("clip_b16", "clip_b32", "scincl", "layout")
    ]
    names = RichSimilarityFeatures({}, {}, spaces).feature_names
    assert len(names) == 190
    for experiment, encoder in [("rich_no_b16", "clip_b16"), ("rich_no_b32", "clip_b32")]:
        mask = runner.feature_mask(names, experiment)
        assert mask.sum() == 173
        assert [name for name, keep in zip(names, mask) if not keep] == [
            name for name in names if name.startswith(f"emb.{encoder}_")
        ]
    assert runner.feature_mask(names, "rich_full").all()


def test_review_scincl_metric_uses_only_caption_pairs_and_three_labels():
    runner = _load_review_runner()
    meta = pd.DataFrame(
        {"pair_type": ["image+text", "text+text"], "label": ["same_figure", "same_paper"]}
    )
    probability = np.asarray([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    result = runner.evaluate(meta, probability, "text+text")
    assert result["rows"] == 1
    assert result["metric_labels"] == ["same_paper", "related_papers", "unrelated_papers"]
    assert result["macro_f1"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="нормировка"):
        runner.evaluate(meta, probability, None)


def test_review_runner_evaluates_every_row_once_on_disjoint_saved_folds(tmp_path):
    import argparse
    import json

    runner = _load_review_runner()
    rows, fold_ids, image_hashes, text_hashes = [], [], [], []
    for fold in range(5):
        for kind in ("image+text", "image+image", "text+text"):
            types = kind.split("+")
            for label in LABELS if kind == "image+text" else LABELS[1:]:
                hashes = [f"{fold}_{kind}_{label}_{side}" for side in (1, 2)]
                rows.append(
                    dict(
                        pair_type=kind,
                        type_1=types[0],
                        type_2=types[1],
                        hash_1=hashes[0],
                        hash_2=hashes[1],
                        label=label,
                    )
                )
                fold_ids.append(fold)
                for modality, digest in zip(types, hashes):
                    (image_hashes if modality == "image" else text_hashes).append(digest)
    audit = tmp_path / "artifacts" / "audit"
    embedding_dir = tmp_path / "artifacts" / "embeddings"
    audit.mkdir(parents=True)
    embedding_dir.mkdir()
    pd.DataFrame(rows).to_parquet(audit / "meta_train.parquet", index=False)
    ids = tmp_path / "fold_ids.npy"
    np.save(ids, fold_ids)
    rng = np.random.default_rng(42)
    for modality, hashes in [("image", image_hashes), ("text", text_hashes)]:
        pd.DataFrame({"hash": hashes}).to_parquet(
            embedding_dir / f"hashes_clip_b16_{modality}.parquet", index=False
        )
        np.save(
            embedding_dir / f"emb_clip_b16_{modality}.npy",
            rng.normal(size=(len(hashes), 3)).astype(np.float16),
        )
    output = tmp_path / "review"
    args = argparse.Namespace(
        project=tmp_path,
        fold_ids=ids,
        runs=["cosine_b16_logistic"],
        output_dir=output,
        full_resolution_ocr=None,
        ocr_jsonl=None,
    )
    runner.run(args)
    probability = np.load(output / "cosine_b16_logistic" / "oof_probabilities.npy")
    report = json.loads((output / "cosine_b16_logistic" / "cv_report.json").read_text())
    assert probability.shape == (50, 4)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
    assert len(report["folds"]) == 5
    assert sum(check["valid_rows"] for check in report["validation_checks"]) == 50
    assert all(check["object_overlap"] == 0 for check in report["validation_checks"])
    assert report["test_data_read"] is False
    assert report["decision_offsets"] is False
    with pytest.raises(FileExistsError, match="существует"):
        runner.run(args)


def test_streaming_rejects_incomplete_object_row(tmp_path):
    from astroclimb.streaming import iter_pair_rows

    source = tmp_path / "train.csv"
    source.write_text(
        "id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2\n"
        "0,1,0,0,0,first\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Неполная строка id=0"):
        list(iter_pair_rows(source, labeled=True, max_field_characters=1024))


def test_streaming_bounds_csv_field_memory(tmp_path):
    import csv
    from astroclimb.streaming import iter_pair_rows

    source = tmp_path / "train.csv"
    source.write_text(
        "id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2\n"
        "0,1,0,0,0," + "a" * 200 + ",second\n",
        encoding="utf-8",
    )
    with pytest.raises(csv.Error, match="field larger"):
        list(iter_pair_rows(source, labeled=True, max_field_characters=100))


def _load_prepare_runner():
    path = Path(__file__).parents[1] / "scripts" / "prepare_review_data.py"
    spec = importlib.util.spec_from_file_location("prepare_review_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preparation_metadata():
    from astroclimb.streaming import object_digest

    return pd.DataFrame([
        dict(id=0, type_1="text", type_2="text", pair_type="text+text",
             hash_1=object_digest("first"), hash_2=object_digest("second"),
             length_1=5, length_2=6, label="same_paper"),
        dict(id=1, type_1="image", type_2="image", pair_type="image+image",
             hash_1=object_digest("iVBORfirst"), hash_2=object_digest("iVBORsecond"),
             length_1=10, length_2=11, label="same_paper"),
    ])


def test_preparation_never_accepts_missing_rows_in_full_scope(tmp_path):
    runner = _load_prepare_runner()
    source = tmp_path / "train.csv"
    source.write_text(
        "id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2\n"
        "0,0,1,0,0,first,second\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="область неполная"):
        list(runner.validated_rows(source, _preparation_metadata()))
    rows = list(runner.validated_rows(source, _preparation_metadata(), ("text+text",)))
    assert [row.pair_id for row in rows] == [0]


def test_preparation_scope_skips_only_unselected_modalities(tmp_path):
    runner = _load_prepare_runner()
    source = tmp_path / "train.csv"
    source.write_text(
        "id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2\n"
        "0,0,1,0,0,first,second\n"
        "1,0,1,0,0,iVBORfirst\n", encoding="utf-8",
    )
    assert len(list(runner.validated_rows(source, _preparation_metadata(), ("text+text",)))) == 1
    with pytest.raises(ValueError, match="Неполная строка"):
        list(runner.validated_rows(source, _preparation_metadata()))


def test_preparation_rejects_changed_objects_and_duplicate_selected_ids(tmp_path):
    runner = _load_prepare_runner()
    source = tmp_path / "train.csv"
    header = "id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2\n"
    source.write_text(header + "0,0,1,0,0,wrong,second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не совпадает"):
        list(runner.validated_rows(source, _preparation_metadata(), ("text+text",)))
    source.write_text(header + "0,0,1,0,0,first,second\n" * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Повторная выбранная"):
        list(runner.validated_rows(source, _preparation_metadata(), ("text+text",)))


def test_review_image_caption_metric_preserves_all_four_labels():
    runner = _load_review_runner()
    meta = pd.DataFrame({"pair_type": ["image+text"], "label": ["same_figure"]})
    report = runner.evaluate(meta, np.asarray([[1, 0, 0, 0]], dtype=np.float32), "image+text")
    assert report["metric_labels"] == list(LABELS)
    assert report["macro_f1"] == pytest.approx(0.25)
