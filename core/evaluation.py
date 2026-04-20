from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.dataset_manager import resolve_mask_path
from core.image_ops import binary_mask, to_gray


def load_mask(mask_path: str | Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    raw = np.fromfile(str(mask_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Mask tidak bisa dibaca: {mask_path}")
    mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask tidak bisa di-decode: {mask_path}")
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary_mask(mask)


def find_ground_truth_mask(image_path: Path, dataset_root: Path, mask_root: Path) -> Path | None:
    # Primary lookup: deterministic mapping to dedicated mask root.
    primary = resolve_mask_path(image_path, dataset_root, mask_root)
    if primary.exists():
        return primary

    # Legacy fallback for old folder structure inside dataset root.
    rel = image_path.relative_to(dataset_root / "data")
    legacy_png = dataset_root / "masks" / rel.with_suffix(".png")
    legacy_jpg = dataset_root / "masks" / rel.with_suffix(".jpg")
    if legacy_png.exists():
        return legacy_png
    if legacy_jpg.exists():
        return legacy_jpg
    return None


def segmentation_mask_from_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = to_gray(image)
    _, pred = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return pred


def iou_from_masks(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    p = pred > 0
    g = gt > 0
    intersection = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    iou = float(intersection / union) if union else 0.0
    return {"intersection": intersection, "union": union, "iou": iou}


def overlay_segmentation(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    pred_mask = pred > 0
    gt_mask = gt > 0

    overlay = base.copy()
    overlay[pred_mask] = [0, 255, 0]
    overlay[gt_mask] = [0, 0, 255]
    return cv2.addWeighted(base, 0.6, overlay, 0.4, 0)


def manual_iou_steps(pred_patch: np.ndarray, gt_patch: np.ndarray) -> dict[str, object]:
    p = pred_patch > 0
    g = gt_patch > 0
    intersection = np.logical_and(p, g).astype(np.uint8)
    union = np.logical_or(p, g).astype(np.uint8)
    i = int(intersection.sum())
    u = int(union.sum())
    iou = float(i / u) if u else 0.0
    steps = [
        "Intersection: sel yang bernilai 1 pada pred dan GT.",
        "Union: sel yang bernilai 1 pada pred atau GT.",
        f"IoU = {i}/{u} = {iou:.4f}",
    ]
    return {"intersection": intersection * 255, "union": union * 255, "iou": iou, "steps": steps}
