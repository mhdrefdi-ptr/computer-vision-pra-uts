from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.convolution import SOBEL_X, manual_convolution_steps
from core.cv_techniques import manual_threshold_steps
from core.image_ops import read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe
from core.morphology import closing, opening
from core.preprocessing import manual_minmax_steps

init_state()


st.title("Manual Calculation Lab")
st.caption("Patch 15x15, matriks piksel, dan langkah hitung manual per operasi.")

images = st.session_state.available_images
if not images:
    st.info("Siapkan dataset di Dataset Manager terlebih dahulu.")
    st.stop()

selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
img = resize_max(read_image(selected), 900)
gray = to_gray(img)

max_x = max(gray.shape[1] - 15, 0)
max_y = max(gray.shape[0] - 15, 0)
start_x = st.slider("Start X patch", 0, max_x, min(max_x, 0))
start_y = st.slider("Start Y patch", 0, max_y, min(max_y, 0))
patch = extract_patch(gray, start_x, start_y)
st.session_state.manual_patch = patch

st.write("Matriks patch 15x15")
st.dataframe(patch_dataframe(patch), use_container_width=True)

op = st.selectbox("Pilih operasi manual", ["preprocessing", "thresholding", "konvolusi", "morfologi"])

if op == "preprocessing":
    out = manual_minmax_steps(patch)
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
elif op == "thresholding":
    t = st.slider("Threshold manual", 0, 255, 127)
    out = manual_threshold_steps(patch, t)
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
elif op == "konvolusi":
    out = manual_convolution_steps(patch, SOBEL_X)
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
else:
    bpatch = (patch > 127).astype("uint8") * 255
    opened, _ = opening(bpatch)
    closed, _ = closing(opened)
    st.write("Before (binary)")
    st.dataframe(patch_dataframe(bpatch), use_container_width=True)
    st.write("After (opening + closing)")
    st.dataframe(patch_dataframe(closed), use_container_width=True)
    st.markdown("- Structuring element 3x3 digunakan untuk opening dan closing.")

st.session_state.last_results["manual_lab"] = {
    "image_path": selected,
    "operation": op,
    "patch": patch,
}
