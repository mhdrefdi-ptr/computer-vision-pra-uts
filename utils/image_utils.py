from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_max(image: np.ndarray, max_dim: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_grayscale_mask(mask_path: str | Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    file_path = Path(mask_path)
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Failed to read mask bytes: {mask_path}")
    mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask
