# GRAIL for AstroCLIMB

Implementation of *GRAIL: Compact Multimodal Similarities for Astronomy Citation Linking*, accepted in the AstroCLIMB shared-task track at WASP 2026. Paper author: Egor Fedosov, Ural Federal University.

The system classifies figure and caption pairs into four relations: `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`. The primary model uses 190 symmetric features from frozen CLIP ViT-B/16, CLIP ViT-B/32, SciNCL, OCR, lexical overlap, and image layout. Classifiers are fitted separately for each pair modality.

## Verified results

| System | Local macro-F1 | Kaggle |
| --- | ---: | ---: |
| Fixed factorized model | 0.48018 | 0.48335 |
| Nested model selection and decision offsets | 0.47692 | 0.48264 |
| Fixed model without OCR | 0.47212 | Not evaluated |
| Fixed model without TF-IDF and term rarity | 0.46743 | Not evaluated |

Local scores use five outer folds that separate connected components of exact objects. These are development results, not an independent test. Separation by source paper is not guaranteed. Kaggle scores refer to two submissions made on September 19, 2026. Reviewer-requested comparisons were run separately on October 8, 2026 and are reported below; the original submitted-system scores are preserved.

### Executed cosine comparisons and CLIP ablations

| System | Evaluation scope | Macro-F1 |
| --- | --- | ---: |
| CLIP B/16 cosine, logistic heads | 10,000 pairs, 4 classes | 0.40436 |
| CLIP B/32 cosine, logistic heads | 10,000 pairs, 4 classes | 0.39016 |
| SciNCL cosine, logistic heads | 3,000 caption pairs, 3 classes | 0.50052 |
| CLIP B/16 + SciNCL cosine, logistic heads | 10,000 pairs, 4 classes | 0.42432 |
| Dual CLIP + SciNCL cosine, logistic heads | 10,000 pairs, 4 classes | 0.42133 |
| Dual CLIP + SciNCL cosine, fixed boosted heads | 10,000 pairs, 4 classes | 0.41223 |
| Full model, replay with regenerated representations | 10,000 pairs, 4 classes | 0.48040 |
| Full model without the B/16 block | 10,000 pairs, 4 classes | 0.47077 |
| Full model without the B/32 block | 10,000 pairs, 4 classes | 0.47280 |

With the same boosted classifier, the full 190 features improve macro-F1 by 0.06817 over the three cosine features. Removing B/16 decreases the score by 0.00963; removing B/32 decreases it by 0.00760. All other features and settings remain unchanged, and each CLIP ablation uses 173 features.

The source `train.csv` is checked against the saved object hashes, types, lengths, and labels of all 10,000 pairs. The original five outer folds are reused, with no exact-object overlap between fitting and validation partitions. Representations are regenerated from pinned model revisions, and saved OCR is reused by object hash. These comparisons use no scoring data, decision offsets, or model selection.

The replayed full-model score differs from the original saved score by 0.00022. It is the control for the new comparisons and does not replace the submitted-system results. Full-precision scores, fold reports, representation provenance, checksums, and out-of-fold (OOF) predictions are available in [reports/review_experiments](reports/review_experiments). Recorded execution-time code hashes are preserved even when documentation and messages are subsequently translated.

## Installation

Python 3.10 or newer is required. Library versions from the original local experiment are recorded in `reports/environment.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,dev]'
python -m pytest
```

All regression checks are in `tests/test_all_regressions.py`.

## Data

Obtain the data separately from the [AstroCLIMB competition page](https://www.kaggle.com/competitions/astroclimb). Place the permitted `train.csv`, `test.csv`, and `sample_submission.csv` files in `astroclimb/` to reproduce the original submission pipeline. Raw images, captions, model weights, and credentials are not distributed in this repository.

The code rejects reads of `solution.csv`. Pair IDs and row order are not predictors. Vocabularies, IDF weights, and embedding statistics are fitted only on objects in the fitting partition of each fold.

## Reproducing features and the original experiment

Run these commands from the repository root.

```bash
export PYTHONPATH=src
python scripts/audit_data.py --data-dir astroclimb --output-dir artifacts/audit
python scripts/build_object_cache.py --data-dir astroclimb --output-dir artifacts/object_cache --side 384 --quality 88 --minimum-free-gib 8

python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind joint --name clip_b16 --model openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --batch-size 32 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind joint --name clip_b32 --model openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --batch-size 32 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind text --name scincl --model malteos/scincl --revision ebc5348d184ba2fc9beee69b4e394263fce57b2e --batch-size 32 --pooling cls --max-length 512 --device auto
python scripts/extract_layout_descriptors.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --name layout
```

Exact reproduction of the local OCR requires macOS Vision:

```bash
swiftc -module-cache-path artifacts/swift_module_cache scripts/ocr_macos.swift -o artifacts/ocr_macos
artifacts/ocr_macos artifacts/object_cache/images artifacts/ocr_fast.jsonl 0 fast
python scripts/train_rich_nested.py --project "$PWD" --output-dir artifacts/run_rich_nested --threads 2
```

For Linux, use `scripts/ocr_tesseract.py`. Changing the OCR engine may change the scores and should be reported as a separate experiment.

## Reviewer-requested experiments

```bash
PYTHONPATH=src python scripts/review_experiments.py --project "$PWD" --output-dir artifacts/review_experiments --threads 2
```

The script compares simple classifiers over CLIP/SciNCL cosine similarities with a factorized model using the paper's fixed settings. Final-model ablations remove only the B/16 or B/32 block while retaining the other features. All comparisons use five object-component folds, without decision offsets or model selection on validation labels. SciNCL alone is evaluated only on caption pairs, where its representation is available; this three-class score is not directly comparable to overall four-class macro-F1.

For an OCR-resolution comparison, pass `--full-resolution-ocr path/to/ocr.jsonl`. The file must contain OCR of the original images using the same engine and settings. The script compares the two OCR variants with identical classifiers and folds. No such resolution experiment is claimed in the saved results. To reproduce the saved experiment's folds exactly, use `--fold-ids artifacts/run_rich_nested/fold_ids.npy`.

### Repeating comparisons without scoring data

These comparisons require only `train.csv`, saved training metadata, outer folds, and OCR. The full competition archive, `test.csv`, and `sample_submission.csv` are unnecessary. Preparation streams the CSV, checks every object against its saved hash, and creates only a compact cache without copying the raw CSV. The `--minimum-free-gib 6` setting enforces a free-disk-space reserve.

In this example, saved metadata, folds, and OCR are in `saved/`. Use a new directory for each replay: completed caches and reports are protected against overwriting.

```bash
review_root=artifacts/review_reproduction
PYTHONPATH=src python scripts/prepare_review_data.py --train train.csv --metadata saved/meta_train.parquet --fold-ids saved/fold_ids.npy --project "$review_root" --minimum-free-gib 6

python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind joint --name clip_b16 --model openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --batch-size 16 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind joint --name clip_b32 --model openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --batch-size 16 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind text --name scincl --model malteos/scincl --revision ebc5348d184ba2fc9beee69b4e394263fce57b2e --batch-size 16 --pooling cls --max-length 512 --device auto
PYTHONPATH=src python scripts/extract_layout_descriptors.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --name layout

PYTHONPATH=src python scripts/review_experiments.py --project "$review_root" --output-dir "$review_root/artifacts/review_experiments" --fold-ids "$review_root/artifacts/audit/fold_ids.npy" --ocr-jsonl saved/ocr_fast.jsonl --threads 2
```

Run CLIP and SciNCL sequentially. With weights cached locally, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to disable network access. Preparation also supports `--modalities` for a complete saved subset; such results apply only to the selected modalities, not to the full dataset.

## Repository structure

- `src/astroclimb/`: object processing, features, models, folds, and input validation.
- `scripts/`: representation extraction, training, and result analysis.
- `tests/test_all_regressions.py`: the unified regression suite.
- `reports/`: compact reports from executed experiments.
- `configs/`: settings for the original experiment.

## License and sources

The code is distributed under the MIT license; see `LICENSE`. Kaggle data and pretrained models are subject to their own terms of use.

- [AstroCLIMB](https://www.kaggle.com/competitions/astroclimb)
- [CLIP](https://proceedings.mlr.press/v139/radford21a.html)
- [SciNCL](https://aclanthology.org/2022.emnlp-main.802/)
