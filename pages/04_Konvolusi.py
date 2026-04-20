from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.convolution import gaussian_blur, manual_convolution_steps, prewitt_edges, sobel_edges
from core.image_ops import bgr_to_rgb, read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe

init_state()


st.title("Konvolusi")
st.caption("Gaussian blur, Sobel, Prewitt, tampilan kernel, dan manual sliding window 15x15.")

images = st.session_state.available_images
if not images:
    st.info("Pilih dataset dan sample di Dataset Manager terlebih dahulu.")
    st.stop()

selected = st.selectbox("Pilih citra", images, index=images.index(st.session_state.selected_image) if st.session_state.selected_image in images else 0, format_func=lambda p: Path(p).name)
st.session_state.selected_image = selected
img = resize_max(read_image(selected), 900)

gaussian, g_kernel = gaussian_blur(img, ksize=5)
sobel, sobel_k = sobel_edges(img)
prewitt, prewitt_k = prewitt_edges(img)

r1c1, r1c2 = st.columns(2)
r1c1.image(gaussian, caption="Gaussian Blur", clamp=True)
r1c2.image(sobel, caption="Sobel", clamp=True)

r2c1, r2c2 = st.columns(2)
r2c1.image(prewitt, caption="Prewitt", clamp=True)
r2c2.image(bgr_to_rgb(img), caption="Original")

st.subheader("Kernel")
k1, k2, k3 = st.columns(3)
k1.write("Gaussian 5x5")
k1.dataframe(g_kernel)
k2.write("Sobel X")
k2.dataframe(sobel_k["kx"])
k3.write("Prewitt X")
k3.dataframe(prewitt_k["kx"])

st.subheader("Manual Konvolusi 15x15")
patch = st.session_state.manual_patch
if patch is None:
    patch = extract_patch(to_gray(img), 0, 0)
manual = manual_convolution_steps(patch, sobel_k["kx"])
st.dataframe(patch_dataframe(manual["result"]), use_container_width=True)
for step in manual["steps"]:
    st.markdown(f"- {step}")

st.session_state.last_results["convolution"] = {
    "image_path": selected,
    "gaussian": gaussian,
    "sobel": sobel,
    "prewitt": prewitt,
}
