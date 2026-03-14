from __future__ import annotations

import numpy as np


def calculate_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def calculate_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return float((2 * precision * recall) / denom) if denom else 0.0


def calculate_iou(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    pred = mask_pred > 0
    true = mask_true > 0
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(intersection / union) if union else 0.0


def confusion_from_masks(mask_pred: np.ndarray, mask_true: np.ndarray) -> tuple[int, int, int, int]:
    pred = mask_pred > 0
    true = mask_true > 0
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, np.logical_not(true)).sum())
    fn = int(np.logical_and(np.logical_not(pred), true).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(true)).sum())
    return tp, fp, fn, tn


def evaluate_segmentation(mask_pred: np.ndarray, mask_true: np.ndarray) -> dict[str, float]:
    tp, fp, fn, _ = confusion_from_masks(mask_pred, mask_true)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)
    iou = calculate_iou(mask_pred, mask_true)
    return {"Precision": precision, "Recall": recall, "F1-score": f1, "IoU": iou}

