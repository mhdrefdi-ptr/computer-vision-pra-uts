from __future__ import annotations

import cv2
import numpy as np


def apply_erosion(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def apply_dilation(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def apply_fill_holes(image: np.ndarray) -> np.ndarray:
    binary = (image > 0).astype(np.uint8) * 255
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, flood_inv)
