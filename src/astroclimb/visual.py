from __future__ import annotations

import numpy as np
from PIL import Image


def image_layout_descriptor(image: Image.Image) -> np.ndarray:
    """Compact deterministic layout/color descriptor for scientific figures."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    histogram, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / max(gray.size, 1)
    channels = np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))])
    scalars = np.asarray(
        [
            np.log1p(width),
            np.log1p(height),
            np.log(max(width, height) / max(1, min(width, height))),
            gray.mean(),
            gray.std(),
            (gray > 0.96).mean(),
            (gray < 0.04).mean(),
            horizontal.mean() if horizontal.size else 0.0,
            vertical.mean() if vertical.size else 0.0,
            (horizontal > 0.12).mean() if horizontal.size else 0.0,
            (vertical > 0.12).mean() if vertical.size else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalars, channels.astype(np.float32), histogram])
