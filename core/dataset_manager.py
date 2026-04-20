from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import kagglehub


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SPLITS = ("train", "validation", "test")


def download_dataset(handle: str = "wanderdust/coin-images", force_download: bool = False) -> Path:
    path = Path(kagglehub.dataset_download(handle, force_download=force_download))
    return path


def _has_expected_structure(root: Path) -> bool:
    data_dir = root / "data"
    return data_dir.exists() and all((data_dir / split).exists() for split in EXPECTED_SPLITS) and (
        data_dir / "cat_to_name.json"
    ).exists()


def find_dataset_root(download_path: Path) -> Path:
    candidates = [download_path, *(p for p in download_path.rglob("*") if p.is_dir())]
    for candidate in candidates:
        if _has_expected_structure(candidate):
            return candidate
    return download_path


def prepare_local_dataset(
    download_path: Path,
    target_root: Path = Path("datasets") / "coin-images",
    refresh_existing: bool = False,
) -> Path:
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        if refresh_existing:
            shutil.copytree(download_path, target_root, dirs_exist_ok=True)
        return target_root
    shutil.copytree(download_path, target_root)
    return target_root


def get_mask_root(dataset_root: Path | None = None) -> Path:
    if dataset_root is None:
        return Path("datasets") / "coin-images-masks"
    return dataset_root.parent / "coin-images-masks"


def ensure_mask_root_structure(mask_root: Path) -> None:
    for split in EXPECTED_SPLITS:
        (mask_root / split).mkdir(parents=True, exist_ok=True)


def validate_structure(dataset_root: Path) -> dict[str, Any]:
    data_dir = dataset_root / "data"
    info: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "valid": False,
        "errors": [],
        "splits": {},
        "num_classes": 0,
    }

    if not data_dir.exists():
        info["errors"].append("Folder 'data' tidak ditemukan.")
        return info

    for split in EXPECTED_SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            info["errors"].append(f"Folder split '{split}' tidak ditemukan.")
            continue

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        image_count = 0
        for class_dir in class_dirs:
            image_count += len([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

        info["splits"][split] = {
            "path": str(split_dir),
            "classes": len(class_dirs),
            "images": image_count,
        }

    mapping_file = data_dir / "cat_to_name.json"
    if not mapping_file.exists():
        info["errors"].append("File 'cat_to_name.json' tidak ditemukan.")

    info["num_classes"] = max((v.get("classes", 0) for v in info["splits"].values()), default=0)
    info["valid"] = len(info["errors"]) == 0
    return info


def load_label_mapping(dataset_root: Path) -> dict[str, str]:
    mapping_file = dataset_root / "data" / "cat_to_name.json"
    if not mapping_file.exists():
        return {}

    with mapping_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: dict[str, str] = {}
    for k, v in raw.items():
        mapping[str(k)] = str(v)
    return mapping


def list_classes(dataset_root: Path, split: str) -> list[str]:
    split_dir = dataset_root / "data" / split
    if not split_dir.exists():
        return []
    return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])


def list_images(dataset_root: Path, split: str, class_name: str | None = None) -> list[Path]:
    split_dir = dataset_root / "data" / split
    if not split_dir.exists():
        return []

    targets = []
    if class_name and class_name != "ALL":
        class_dir = split_dir / class_name
        if class_dir.exists():
            targets.append(class_dir)
    else:
        targets = [d for d in split_dir.iterdir() if d.is_dir()]

    images: list[Path] = []
    for target in targets:
        images.extend([p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    return sorted(images, key=lambda p: str(p).lower())


def summarize_split_counts(dataset_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in EXPECTED_SPLITS:
        counts[split] = len(list_images(dataset_root, split, class_name=None))
    return counts


def resolve_mask_path(image_path: Path, dataset_root: Path, mask_root: Path) -> Path:
    rel = image_path.relative_to(dataset_root / "data")
    return (mask_root / rel).with_suffix(".png")


def mask_coverage_summary(dataset_root: Path, mask_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SPLITS:
        images = list_images(dataset_root, split, class_name=None)
        total = len(images)
        has_mask = 0
        for image_path in images:
            if resolve_mask_path(image_path, dataset_root, mask_root).exists():
                has_mask += 1
        summary[split] = {
            "total_images": total,
            "has_mask": has_mask,
            "missing_mask": total - has_mask,
        }
    return summary
