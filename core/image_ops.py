from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Tidak bisa membaca bytes image: {file_path}")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Tidak bisa decode image: {file_path}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max(image: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return image
    new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v == 0:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - min_v) / (max_v - min_v)
    return (out * 255).astype(np.uint8)


def histogram_figure(before: np.ndarray, after: np.ndarray) -> plt.Figure:
    b = np.asarray(to_gray(before), dtype=np.float32).ravel()
    a = np.asarray(to_gray(after), dtype=np.float32).ravel()

    # Defensive sanitization for cloud/runtime differences.
    b = np.nan_to_num(b, nan=0.0, posinf=255.0, neginf=0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    b = np.clip(np.rint(b), 0, 255).astype(np.uint8, copy=False)
    a = np.clip(np.rint(a), 0, 255).astype(np.uint8, copy=False)

    if b.size == 0:
        b = np.array([0], dtype=np.uint8)
    if a.size == 0:
        a = np.array([0], dtype=np.uint8)

    # Use bincount on uint8 to avoid np.histogram edge/runtime issues on cloud.
    b_counts = np.bincount(b, minlength=256)
    a_counts = np.bincount(a, minlength=256)
    b_hist = b_counts.reshape(32, 8).sum(axis=1)
    a_hist = a_counts.reshape(32, 8).sum(axis=1)
    b_centers = np.arange(32) * 8 + 4
    a_centers = np.arange(32) * 8 + 4
    bar_w = (255 / 32) * 0.9

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=120)
    axes[0].bar(b_centers, b_hist, width=bar_w, color="steelblue")
    axes[0].set_title("Histogram Before")
    axes[1].bar(a_centers, a_hist, width=bar_w, color="darkorange")
    axes[1].set_title("Histogram After")
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def binary_mask(image: np.ndarray) -> np.ndarray:
    return ((image > 0).astype(np.uint8) * 255)
