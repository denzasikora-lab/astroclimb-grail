from __future__ import annotations

LABELS: tuple[str, ...] = (
    "same_figure",
    "same_paper",
    "related_papers",
    "unrelated_papers",
)

PAIR_TYPES: tuple[str, ...] = ("image+text", "text+text", "image+image")
IMAGE_PREFIXES: tuple[str, ...] = ("iVBORw0KGgo", "/9j/")
SUBMISSION_COLUMNS: tuple[str, ...] = ("id", *LABELS)
