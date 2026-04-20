from __future__ import annotations

import cv2
import numpy as np

from core.image_ops import bgr_to_rgb, to_gray


def threshold_global(image: np.ndarray, value: int = 127) -> np.ndarray:
    gray = to_gray(image)
    _, out = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    return out


def threshold_otsu(image: np.ndarray) -> tuple[np.ndarray, int]:
    gray = to_gray(image)
    threshold, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out, int(threshold)


def harris_corner_overlay(
    image: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, threshold_ratio: float = 0.01
) -> tuple[np.ndarray, int]:
    gray = to_gray(image)
    gray_float = np.float32(gray)
    response = cv2.cornerHarris(gray_float, block_size, ksize, k)
    response = cv2.dilate(response, None)

    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    mask = response > threshold_ratio * response.max()
    vis[mask] = [0, 0, 255]
    return vis, int(mask.sum())


def manual_threshold_steps(patch: np.ndarray, threshold: int) -> dict[str, object]:
    binary = (patch > threshold).astype(np.uint8) * 255
    steps = [
        f"Threshold = {threshold}",
        "Rumus: pixel > threshold => 255, selain itu 0",
        f"Contoh [0,0]: {int(patch[0,0])} {'>' if int(patch[0,0]) > threshold else '<='} {threshold}",
    ]
    return {"result": binary, "steps": steps}


def manual_harris_notes() -> list[str]:
    return [
        "Hitung gradien Ix dan Iy (Sobel).",
        "Hitung matriks struktur M pada window lokal.",
        "Hitung respon: R = det(M) - k * (trace(M))^2.",
        "Corner terdeteksi bila R melebihi threshold lokal.",
    ]
