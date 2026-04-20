from __future__ import annotations

import cv2
import numpy as np

from core.image_ops import to_gray


def minmax_normalization(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image).astype(np.float32)
    min_v = gray.min()
    max_v = gray.max()
    if max_v - min_v == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    norm = (gray - min_v) / (max_v - min_v)
    return (norm * 255).astype(np.uint8)


def contrast_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    return clahe.apply(gray)


def sharpening(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, cv2.CV_8U, kernel)


def manual_minmax_steps(patch: np.ndarray) -> dict[str, object]:
    p = patch.astype(np.float32)
    min_v = float(p.min())
    max_v = float(p.max())
    if max_v - min_v == 0:
        normalized = np.zeros_like(p, dtype=np.uint8)
    else:
        normalized = (((p - min_v) / (max_v - min_v)) * 255).astype(np.uint8)

    steps = [
        f"Min = {min_v:.2f}",
        f"Max = {max_v:.2f}",
        "Rumus: ((x - min) / (max - min)) * 255",
        f"Contoh piksel [0,0]: (({p[0,0]:.2f} - {min_v:.2f}) / ({max_v:.2f} - {min_v:.2f})) * 255",
    ]
    return {"result": normalized, "steps": steps}
