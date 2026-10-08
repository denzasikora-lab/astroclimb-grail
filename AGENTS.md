# Repository instructions

All repository and manuscript text must be in English. This includes documentation, comments, docstrings, command-line messages, test descriptions, citation metadata, and saved report descriptions. Do not introduce Cyrillic characters. The publication-text regression checks enforce the content restrictions.

Preserve numerical results, OOF predictions, model settings, API names, and execution-time provenance when editing explanatory text. Do not replace measured results with simulated values.

Keep all regression checks in `tests/test_all_regressions.py`; do not create separate test modules. Preserve the existing NumPy-style docstring structure.

Do not commit raw competition data, captions, images, model weights, credentials, or local caches. Preserve the original submitted manuscript when preparing revised versions.
