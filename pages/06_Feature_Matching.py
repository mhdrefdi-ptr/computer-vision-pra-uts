from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.image_ops import bgr_to_rgb, read_image, resize_max
from core.matching import bf_match

init_state()


st.title("Feature Detection & Matching")
st.caption("ORB + BFMatcher, visual keypoints, descriptor info, dan garis matching.")

images = st.session_state.available_images
if len(images) < 2:
    st.info("Minimal butuh 2 citra. Pilih dataset di Dataset Manager.")
    st.stop()

pair = st.session_state.matching_pair if len(st.session_state.matching_pair) == 2 else images[:2]
idx_a = images.index(pair[0]) if pair[0] in images else 0
idx_b = images.index(pair[1]) if pair[1] in images else min(1, len(images) - 1)

a = st.selectbox("Image A", images, index=idx_a, format_func=lambda p: Path(p).name)
b = st.selectbox("Image B", images, index=idx_b, format_func=lambda p: Path(p).name)
st.session_state.matching_pair = [a, b]

img_a = resize_max(read_image(a), 900)
img_b = resize_max(read_image(b), 900)

show_a, show_b = st.columns(2)
show_a.image(bgr_to_rgb(img_a), caption="Image A")
show_b.image(bgr_to_rgb(img_b), caption="Image B")

if st.button("Jalankan ORB + BFMatcher", type="primary"):
    result = bf_match(img_a, img_b)

    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(result["vis_a"]), caption=f"Keypoints A: {len(result['keypoints_a'])}")
    c2.image(bgr_to_rgb(result["vis_b"]), caption=f"Keypoints B: {len(result['keypoints_b'])}")

    if result["descriptors_a"] is not None:
        st.write(f"Descriptor A shape: {result['descriptors_a'].shape}")
    if result["descriptors_b"] is not None:
        st.write(f"Descriptor B shape: {result['descriptors_b'].shape}")

    if result["match_vis"] is None:
        st.warning("Descriptor tidak cukup untuk matching.")
    else:
        st.image(bgr_to_rgb(result["match_vis"]), caption=f"Jumlah match: {len(result['matches'])}")

    st.session_state.last_results["matching"] = {
        "image_a": a,
        "image_b": b,
        "keypoints_a": len(result["keypoints_a"]),
        "keypoints_b": len(result["keypoints_b"]),
        "matches": len(result["matches"]),
        "match_vis": bgr_to_rgb(result["match_vis"]) if result["match_vis"] is not None else None,
    }
