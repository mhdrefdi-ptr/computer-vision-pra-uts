from __future__ import annotations

import cv2
import numpy as np

from core.image_ops import to_gray


PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
PREWITT_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    gray = to_gray(image)
    k = cv2.getGaussianKernel(ksize, sigma)
    kernel = k @ k.T
    out = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    return out, kernel


def sobel_edges(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = to_gray(image)
    gx = cv2.filter2D(gray, cv2.CV_32F, SOBEL_X)
    gy = cv2.filter2D(gray, cv2.CV_32F, SOBEL_Y)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag), {"kx": SOBEL_X, "ky": SOBEL_Y}


def prewitt_edges(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = to_gray(image)
    gx = cv2.filter2D(gray, cv2.CV_32F, PREWITT_X)
    gy = cv2.filter2D(gray, cv2.CV_32F, PREWITT_Y)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag), {"kx": PREWITT_X, "ky": PREWITT_Y}


def manual_convolution_steps(patch: np.ndarray, kernel: np.ndarray) -> dict[str, object]:
    patch = patch.astype(np.float32)
    kernel = kernel.astype(np.float32)
    result = cv2.filter2D(patch, cv2.CV_32F, kernel)
    center = patch[6:9, 6:9]
    conv_example = float((center * kernel).sum())
    steps = [
        "Sliding window 3x3 pada patch 15x15.",
        "Untuk setiap posisi, kalikan elemen window dengan kernel, lalu jumlahkan.",
        f"Contoh window tengah menghasilkan nilai {conv_example:.2f}.",
    ]
    return {"result": result, "steps": steps}
