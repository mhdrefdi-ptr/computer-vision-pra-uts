from __future__ import annotations

import cv2
import numpy as np

from modules.preprocessing import convert_to_grayscale


def apply_gaussian_blur(image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma_x: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def apply_sobel(image: np.ndarray) -> np.ndarray:
    gray = convert_to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def apply_prewitt(image: np.ndarray) -> np.ndarray:
    gray = convert_to_grayscale(image)
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)

