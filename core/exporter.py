from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.image_ops import bgr_to_rgb


EXPORT_ROOT = Path("exports")


def create_export_dir(prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXPORT_ROOT / f"{prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def save_metrics(path: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = path / "metrics.csv"
    xlsx_path = path / "metrics.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def save_manual_matrix(path: Path, matrix: np.ndarray, name: str) -> tuple[Path, Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matrix)
    csv_path = path / f"{name}.csv"
    xlsx_path = path / f"{name}.xlsx"
    png_path = path / f"{name}.png"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(matrix, cmap="gray")
    ax.set_title(name)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    return csv_path, xlsx_path, png_path
