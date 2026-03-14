from __future__ import annotations

import cv2
import numpy as np

from modules.preprocessing import convert_to_grayscale


def detect_harris_corners(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
) -> np.ndarray:
    gray = convert_to_grayscale(image)
    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, block_size, ksize, k)
    harris_response = cv2.dilate(harris_response, None)

    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[harris_response > threshold_ratio * harris_response.max()] = [0, 0, 255]
    return vis

