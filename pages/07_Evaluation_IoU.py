from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.dataset_manager import get_mask_root, resolve_mask_path
from core.evaluation import (
    find_ground_truth_mask,
    iou_from_masks,
    load_mask,
    manual_iou_steps,
    overlay_segmentation,
    segmentation_mask_from_threshold,
)
from core.image_ops import bgr_to_rgb, read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe

init_state()


st.title("Evaluation Module - IoU")
st.caption("Hitung Intersection, Union, IoU, dan tampilkan overlay hasil segmentasi.")

images = st.session_state.available_images
dataset_root_raw = st.session_state.dataset_root
if not images or not dataset_root_raw:
    st.info("Siapkan dataset di Dataset Manager terlebih dahulu.")
    st.stop()

dataset_root = Path(dataset_root_raw)
mask_root = Path(st.session_state.mask_root) if st.session_state.mask_root else get_mask_root(dataset_root)
selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
st.session_state.selected_image = selected
img = resize_max(read_image(selected), 900)

threshold = st.slider("Threshold prediksi", 0, 255, 127)
pred = segmentation_mask_from_threshold(img, threshold)

expected_path = resolve_mask_path(Path(selected), dataset_root, mask_root)
gt_path = find_ground_truth_mask(Path(selected), dataset_root, mask_root)

if gt_path is None:
    st.warning("Ground truth mask tidak ditemukan untuk gambar ini.")
    st.caption(f"Path mask yang diharapkan: `{expected_path}`")
    st.caption("Buat file `.png` biner 0/255 pada path tersebut, lalu refresh halaman ini.")
    st.image(pred, caption="Predicted Binary Mask", clamp=True)
    st.stop()

st.success(f"Mask ditemukan: `{gt_path}`")
gt = load_mask(gt_path, target_shape=pred.shape)
metric = iou_from_masks(pred, gt)
overlay = overlay_segmentation(img, pred, gt)

c1, c2, c3 = st.columns(3)
c1.image(pred, caption="Predicted Mask", clamp=True)
c2.image(gt, caption=f"Ground Truth ({gt_path.name})", clamp=True)
c3.image(bgr_to_rgb(overlay), caption="Overlay (Hijau=Pred, Merah=GT)")

st.metric("Intersection", metric["intersection"])
st.metric("Union", metric["union"])
st.metric("IoU", f"{metric['iou']:.4f}")

patch = st.session_state.manual_patch
if patch is None:
    patch = extract_patch(to_gray(img), 0, 0)

pred_patch = (patch > threshold).astype("uint8") * 255
gt_patch = extract_patch(gt, 0, 0)
manual = manual_iou_steps(pred_patch, gt_patch)

st.subheader("Manual IoU 15x15")
st.dataframe(patch_dataframe(manual["intersection"]), use_container_width=True)
st.dataframe(patch_dataframe(manual["union"]), use_container_width=True)
for step in manual["steps"]:
    st.markdown(f"- {step}")

row = {
    "image": Path(selected).name,
    "threshold": threshold,
    "intersection": metric["intersection"],
    "union": metric["union"],
    "iou": metric["iou"],
}
st.session_state.metrics_rows = [r for r in st.session_state.metrics_rows if r["image"] != row["image"]] + [row]
st.session_state.last_results["evaluation"] = {
    "image_path": selected,
    "pred": pred,
    "gt": gt,
    "overlay": bgr_to_rgb(overlay),
    "metric": metric,
}
