from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .features import gather_embeddings, symmetric_embedding_features


@dataclass
class EncoderSpace:
    name: str
    image: np.ndarray | None
    text: np.ndarray | None
    image_lookup: dict[str, int]
    text_lookup: dict[str, int]

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "EncoderSpace":
        root = Path(directory)
        image_path = root / f"emb_{name}_image.npy"
        text_path = root / f"emb_{name}_text.npy"
        image = np.load(image_path, mmap_mode="r") if image_path.exists() else None
        text = np.load(text_path, mmap_mode="r") if text_path.exists() else None
        image_lookup: dict[str, int] = {}
        text_lookup: dict[str, int] = {}
        if image is not None:
            hash_path = root / f"hashes_{name}_image.parquet"
            if not hash_path.exists():
                raise FileNotFoundError(f"Missing mapping for {image_path.name}: {hash_path.name}")
            hashes = pd.read_parquet(hash_path)["hash"].astype(str)
            if len(hashes) != image.shape[0]:
                raise ValueError(
                    f"{image_path.name} has {image.shape[0]} rows but {hash_path.name} has {len(hashes)}"
                )
            if hashes.duplicated().any():
                raise ValueError(f"{hash_path.name} contains duplicate hashes")
            image_lookup = {value: index for index, value in enumerate(hashes)}
        if text is not None:
            hash_path = root / f"hashes_{name}_text.parquet"
            if not hash_path.exists():
                raise FileNotFoundError(f"Missing mapping for {text_path.name}: {hash_path.name}")
            hashes = pd.read_parquet(hash_path)["hash"].astype(str)
            if len(hashes) != text.shape[0]:
                raise ValueError(
                    f"{text_path.name} has {text.shape[0]} rows but {hash_path.name} has {len(hashes)}"
                )
            if hashes.duplicated().any():
                raise ValueError(f"{hash_path.name} contains duplicate hashes")
            text_lookup = {value: index for index, value in enumerate(hashes)}
        if image is None and text is None:
            raise FileNotFoundError(f"No embeddings found for encoder space {name!r} in {root}")
        return cls(name, image, text, image_lookup, text_lookup)

    def side(self, hashes: np.ndarray, types: np.ndarray) -> np.ndarray:
        output: np.ndarray | None = None
        for modality in ("image", "text"):
            mask = types == modality
            if not mask.any():
                continue
            matrix = self.image if modality == "image" else self.text
            lookup = self.image_lookup if modality == "image" else self.text_lookup
            if matrix is None:
                raise ValueError(f"Encoder {self.name!r} has no {modality} embeddings")
            selected = gather_embeddings(hashes[mask], lookup, matrix)
            if output is None:
                output = np.zeros((len(hashes), selected.shape[1]), dtype=np.float32)
            output[mask] = selected
        if output is None:
            raise ValueError("No objects supplied")
        return output


def pair_feature_matrix(
    meta: pd.DataFrame,
    spaces: Iterable[EncoderSpace],
) -> np.ndarray:
    pair_types = meta["pair_type"].astype(str).unique()
    if len(pair_types) != 1:
        raise ValueError("pair_feature_matrix expects exactly one modality subset")
    pair_type = pair_types[0]
    blocks: list[np.ndarray] = []
    hash_1 = meta["hash_1"].astype(str).to_numpy()
    hash_2 = meta["hash_2"].astype(str).to_numpy()
    type_1 = meta["type_1"].astype(str).to_numpy()
    type_2 = meta["type_2"].astype(str).to_numpy()
    for space in spaces:
        if pair_type == "image+text":
            if space.image is not None and space.text is not None:
                left = space.side(hash_1, type_1)
                right = space.side(hash_2, type_2)
                blocks.append(symmetric_embedding_features(left, right))
            elif space.image is not None:
                image_hashes = np.where(type_1 == "image", hash_1, hash_2)
                blocks.append(gather_embeddings(image_hashes, space.image_lookup, space.image))
            elif space.text is not None:
                text_hashes = np.where(type_1 == "text", hash_1, hash_2)
                blocks.append(gather_embeddings(text_hashes, space.text_lookup, space.text))
        elif pair_type == "image+image" and space.image is not None:
            left = gather_embeddings(hash_1, space.image_lookup, space.image)
            right = gather_embeddings(hash_2, space.image_lookup, space.image)
            blocks.append(symmetric_embedding_features(left, right))
        elif pair_type == "text+text" and space.text is not None:
            left = gather_embeddings(hash_1, space.text_lookup, space.text)
            right = gather_embeddings(hash_2, space.text_lookup, space.text)
            blocks.append(symmetric_embedding_features(left, right))
    length_min = np.minimum(meta["length_1"], meta["length_2"]).to_numpy(dtype=np.float32)
    length_max = np.maximum(meta["length_1"], meta["length_2"]).to_numpy(dtype=np.float32)
    scalar = np.column_stack(
        [
            np.log1p(length_min),
            np.log1p(length_max),
            length_max / np.maximum(length_min, 1.0),
        ]
    ).astype(np.float32)
    blocks.append(scalar)
    if len(blocks) == 1:
        raise ValueError("No encoder space covers both modalities in this pair subset")
    return np.concatenate(blocks, axis=1)
