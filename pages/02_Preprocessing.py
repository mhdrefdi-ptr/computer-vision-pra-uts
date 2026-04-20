from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.image_ops import bgr_to_rgb, histogram_figure, read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe
from core.preprocessing import contrast_enhancement, manual_minmax_steps, minmax_normalization, sharpening

init_state()


st.title("Preprocessing")
st.caption("Normalisasi Min-Max, contrast enhancement, sharpening, histogram, dan manual patch 15x15.")

images = st.session_state.available_images
if not images:
    st.info("Pilih dataset dan sample di Dataset Manager terlebih dahulu.")
    st.stop()

selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
st.session_state.selected_image = selected

method = st.radio("Metode preprocessing", ["Min-Max Normalization", "Contrast Enhancement", "Sharpening"], horizontal=True)

img = resize_max(read_image(selected), 900)
if method == "Min-Max Normalization":
    processed = minmax_normalization(img)
elif method == "Contrast Enhancement":
    processed = contrast_enhancement(img)
else:
    processed = sharpening(img)

c1, c2 = st.columns(2)
c1.image(bgr_to_rgb(img), caption="Before")
c2.image(processed, caption="After", clamp=True)

st.pyplot(histogram_figure(to_gray(img), processed), use_container_width=True)

st.subheader("Manual 15x15 (Normalisasi)")
gray = to_gray(img)
max_x = max(gray.shape[1] - 15, 0)
max_y = max(gray.shape[0] - 15, 0)
start_x = st.slider("Start X", 0, max_x, min(0, max_x))
start_y = st.slider("Start Y", 0, max_y, min(0, max_y))
patch = extract_patch(gray, start_x, start_y)
st.session_state.manual_patch = patch

manual = manual_minmax_steps(patch)
st.write("Patch asli")
st.dataframe(patch_dataframe(patch), use_container_width=True)
st.write("Hasil normalisasi patch")
st.dataframe(patch_dataframe(manual["result"]), use_container_width=True)
for step in manual["steps"]:
    st.markdown(f"- {step}")

st.session_state.last_results["preprocessing"] = {
    "image_path": selected,
    "method": method,
    "before": bgr_to_rgb(img),
    "after": processed,
}
