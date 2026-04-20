from __future__ import annotations

import pandas as pd
import numpy as np

from core.image_ops import to_gray


PATCH_SIZE = 15


def extract_patch(image: np.ndarray, start_x: int, start_y: int, patch_size: int = PATCH_SIZE) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape
    end_x = min(start_x + patch_size, w)
    end_y = min(start_y + patch_size, h)
    patch = gray[start_y:end_y, start_x:end_x]

    if patch.shape != (patch_size, patch_size):
        padded = np.zeros((patch_size, patch_size), dtype=np.uint8)
        padded[: patch.shape[0], : patch.shape[1]] = patch
        return padded
    return patch


def patch_dataframe(patch: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(patch)
