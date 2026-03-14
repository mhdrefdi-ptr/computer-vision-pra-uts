from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_dataset(path: str | Path) -> List[Path]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []
    files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: str(p).lower())


def get_image_list(path: str | Path = "dataset/images") -> List[str]:
    root = Path(path)
    return [str(p.relative_to(root)).replace("\\", "/") for p in load_dataset(path)]


def load_image(filename: str | Path) -> np.ndarray:
    file_path = Path(filename)
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Failed to read file bytes: {filename}")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {filename}")
    return image


def create_thumbnail(image: np.ndarray, max_size: int = 180) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
    thumb = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
