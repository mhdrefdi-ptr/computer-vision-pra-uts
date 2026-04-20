from __future__ import annotations

import cv2
import numpy as np

from core.image_ops import binary_mask


def opening(binary_image: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((ksize, ksize), np.uint8)
    img = binary_mask(binary_image)
    out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return out, kernel


def closing(binary_image: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((ksize, ksize), np.uint8)
    img = binary_mask(binary_image)
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return out, kernel


def fill_holes(binary_image: np.ndarray) -> np.ndarray:
    img = binary_mask(binary_image)
    h, w = img.shape[:2]
    flood = img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(img, flood_inv)


def manual_morphology_steps(operation: str, kernel: np.ndarray) -> list[str]:
    return [
        f"Operasi: {operation}",
        f"Structuring element ukuran {kernel.shape[0]}x{kernel.shape[1]}",
        "Opening = Erosi lalu Dilasi.",
        "Closing = Dilasi lalu Erosi.",
    ]
