from __future__ import annotations

import cv2
import numpy as np

from modules.preprocessing import convert_to_grayscale


def apply_binary_threshold(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

