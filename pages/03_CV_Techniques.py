from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.cv_techniques import (
    harris_corner_overlay,
    manual_harris_notes,
    manual_threshold_steps,
    threshold_global,
    threshold_otsu,
)
from core.image_ops import bgr_to_rgb, read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe

init_state()


st.title("Techniques Computer Vision")
st.caption("Thresholding (Global/Otsu) dan Harris Corner Detection.")

images = st.session_state.available_images
if not images:
    st.info("Pilih dataset dan sample di Dataset Manager terlebih dahulu.")
    st.stop()

selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
st.session_state.selected_image = selected
img = resize_max(read_image(selected), 900)

mode = st.radio("Mode thresholding", ["Global", "Otsu"], horizontal=True)
if mode == "Global":
    threshold_value = st.slider("Nilai threshold", 0, 255, 127)
    binary = threshold_global(img, threshold_value)
    threshold_used = threshold_value
else:
    binary, threshold_used = threshold_otsu(img)

harris_vis, corners = harris_corner_overlay(img)

c1, c2 = st.columns(2)
c1.image(binary, caption=f"Segmentasi ({mode}) - T={threshold_used}", clamp=True)
c2.image(bgr_to_rgb(harris_vis), caption=f"Harris Corner Overlay ({corners} corners)")

st.subheader("Manual Requirement")
patch = st.session_state.manual_patch
if patch is None:
    patch = extract_patch(to_gray(img), 0, 0)

manual_thr = manual_threshold_steps(patch, threshold_used)
st.write("Manual thresholding pada patch 15x15")
st.dataframe(patch_dataframe(manual_thr["result"]), use_container_width=True)
for step in manual_thr["steps"]:
    st.markdown(f"- {step}")

st.write("Catatan manual Harris")
for note in manual_harris_notes():
    st.markdown(f"- {note}")

st.session_state.last_results["cv_techniques"] = {
    "image_path": selected,
    "threshold_mode": mode,
    "threshold_value": threshold_used,
    "segmentation": binary,
    "harris": bgr_to_rgb(harris_vis),
    "corners": corners,
}
