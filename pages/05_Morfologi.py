from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.cv_techniques import threshold_global
from core.image_ops import read_image, resize_max, to_gray
from core.manual_lab import extract_patch
from core.morphology import closing, fill_holes, manual_morphology_steps, opening

init_state()


st.title("Morfologi")
st.caption("Opening, Closing, Filling Holes pada citra biner.")

images = st.session_state.available_images
if not images:
    st.info("Pilih dataset dan sample di Dataset Manager terlebih dahulu.")
    st.stop()

selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
st.session_state.selected_image = selected
img = resize_max(read_image(selected), 900)

thr = st.slider("Threshold biner", 0, 255, 127)
bin_img = threshold_global(img, thr)
open_img, kernel = opening(bin_img, 3)
close_img, _ = closing(bin_img, 3)
fill_img = fill_holes(close_img)

c1, c2 = st.columns(2)
c1.image(bin_img, caption="Biner Awal", clamp=True)
c2.image(open_img, caption="Opening", clamp=True)

c3, c4 = st.columns(2)
c3.image(close_img, caption="Closing", clamp=True)
c4.image(fill_img, caption="Filling Holes", clamp=True)

for step in manual_morphology_steps("Opening/Closing", kernel):
    st.markdown(f"- {step}")

patch = st.session_state.manual_patch
if patch is None:
    patch = extract_patch(to_gray(img), 0, 0)
st.write("Patch 15x15 aktif digunakan untuk manual structuring element di halaman Manual Lab.")

st.session_state.last_results["morphology"] = {
    "image_path": selected,
    "binary": bin_img,
    "opening": open_img,
    "closing": close_img,
    "filled": fill_img,
}
