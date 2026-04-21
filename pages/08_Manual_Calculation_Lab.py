from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core.app_state import init_state
from core.convolution import SOBEL_X, manual_convolution_steps
from core.cv_techniques import manual_threshold_steps
from core.image_ops import read_image, resize_max, to_gray
from core.manual_lab import extract_patch, patch_dataframe
from core.morphology import closing, opening
from core.preprocessing import manual_minmax_steps

init_state()


def matrix_to_csv_bytes(matrix: np.ndarray) -> bytes:
    buffer = io.StringIO()
    np.savetxt(buffer, matrix, delimiter=",", fmt="%g")
    return buffer.getvalue().encode("utf-8")


def matrix_to_png_bytes(matrix: np.ndarray) -> bytes:
    arr = matrix.astype(np.float32)
    if arr.min() < 0 or arr.max() > 255 or matrix.dtype != np.uint8:
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v - min_v == 0:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = (((arr - min_v) / (max_v - min_v)) * 255).astype(np.uint8)
    else:
        arr = matrix.astype(np.uint8)

    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Gagal encode PNG.")
    return encoded.tobytes()


def descriptor_to_bits(desc: np.ndarray) -> np.ndarray:
    return np.unpackbits(desc.astype(np.uint8))


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


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
before_matrix = patch.copy()
after_matrix = patch.copy()

st.write("Matriks patch 15x15")
st.dataframe(patch_dataframe(patch), use_container_width=True)

op = st.selectbox("Pilih operasi manual", ["preprocessing", "thresholding", "konvolusi", "morfologi", "feature_matching"])
before_text = None
after_text = None

if op == "preprocessing":
    out = manual_minmax_steps(patch)
    after_matrix = out["result"]
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
elif op == "thresholding":
    t = st.slider("Threshold manual", 0, 255, 127)
    out = manual_threshold_steps(patch, t)
    after_matrix = out["result"]
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
elif op == "konvolusi":
    out = manual_convolution_steps(patch, SOBEL_X)
    after_matrix = out["result"]
    st.dataframe(patch_dataframe(out["result"]), use_container_width=True)
    for step in out["steps"]:
        st.markdown(f"- {step}")
elif op == "morfologi":
    bpatch = (patch > 127).astype("uint8") * 255
    before_matrix = bpatch
    opened, _ = opening(bpatch)
    closed, _ = closing(opened)
    after_matrix = closed
    st.write("Before (binary)")
    st.dataframe(patch_dataframe(bpatch), use_container_width=True)
    st.write("After (opening + closing)")
    st.dataframe(patch_dataframe(closed), use_container_width=True)
    st.markdown("- Structuring element 3x3 digunakan untuk opening dan closing.")
else:
    data = st.session_state.manual_matching_data
    if not data:
        st.info("Belum ada data matching manual. Jalankan dulu menu Feature Matching (ORB + BFMatcher).")
    else:
        desc_a = np.array(data["descriptor_a"], dtype=np.uint8)
        desc_b = np.array(data["descriptor_b"], dtype=np.uint8)
        bits_a = descriptor_to_bits(desc_a)
        bits_b = descriptor_to_bits(desc_b)
        xor_bits = np.bitwise_xor(bits_a, bits_b)
        hamming_manual = int(xor_bits.sum())
        bf_distance = int(round(float(data["distance"])))

        st.write("Descriptor A (best query)")
        st.dataframe(patch_dataframe(desc_a.reshape(1, -1)), use_container_width=True)
        st.write("Descriptor B (best train)")
        st.dataframe(patch_dataframe(desc_b.reshape(1, -1)), use_container_width=True)

        st.write("Bit-level (manual Hamming)")
        st.dataframe(
            {
                "A_bits": ["".join(map(str, bits_a.tolist()))],
                "B_bits": ["".join(map(str, bits_b.tolist()))],
                "XOR": ["".join(map(str, xor_bits.tolist()))],
            },
            use_container_width=True,
        )

        st.markdown(f"- Query index: `{data['query_idx']}`")
        st.markdown(f"- Train index: `{data['train_idx']}`")
        st.markdown(f"- Hamming manual (jumlah bit 1 hasil XOR): **{hamming_manual}**")
        st.markdown(f"- BFMatcher distance: **{bf_distance}**")
        st.markdown(f"- Validasi: **{'SAMA' if hamming_manual == bf_distance else 'BERBEDA'}**")

        before_text = (
            f"image_a={data['image_a']}\n"
            f"image_b={data['image_b']}\n"
            f"query_idx={data['query_idx']}\n"
            f"train_idx={data['train_idx']}\n"
            f"descriptor_a={','.join(map(str, desc_a.tolist()))}\n"
            f"descriptor_b={','.join(map(str, desc_b.tolist()))}\n"
        )
        after_text = (
            f"xor_bits={''.join(map(str, xor_bits.tolist()))}\n"
            f"hamming_manual={hamming_manual}\n"
            f"bfmatcher_distance={bf_distance}\n"
            f"validation={'SAMA' if hamming_manual == bf_distance else 'BERBEDA'}\n"
        )

st.subheader("Download Before / After")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"manual_{op}_x{start_x}_y{start_y}_{timestamp}"

if op == "feature_matching":
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.download_button(
            label="Download Before (TXT)",
            data=text_to_bytes(before_text or "Data matching belum tersedia"),
            file_name=f"{base_name}_before.txt",
            mime="text/plain",
            key=f"before_txt_{base_name}",
        )
    with b_col2:
        st.download_button(
            label="Download Before (CSV)",
            data=text_to_bytes(before_text.replace("\n", ",\n") if before_text else "info,data matching belum tersedia\n"),
            file_name=f"{base_name}_before.csv",
            mime="text/csv",
            key=f"before_csv_{base_name}",
        )

    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.download_button(
            label="Download After (TXT)",
            data=text_to_bytes(after_text or "Data matching belum tersedia"),
            file_name=f"{base_name}_after.txt",
            mime="text/plain",
            key=f"after_txt_{base_name}",
        )
    with a_col2:
        st.download_button(
            label="Download After (CSV)",
            data=text_to_bytes(after_text.replace("\n", ",\n") if after_text else "info,data matching belum tersedia\n"),
            file_name=f"{base_name}_after.csv",
            mime="text/csv",
            key=f"after_csv_{base_name}",
        )
else:
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.download_button(
            label="Download Before (PNG)",
            data=matrix_to_png_bytes(before_matrix),
            file_name=f"{base_name}_before.png",
            mime="image/png",
            key=f"before_png_{base_name}",
        )
    with b_col2:
        st.download_button(
            label="Download Before (CSV)",
            data=matrix_to_csv_bytes(before_matrix),
            file_name=f"{base_name}_before.csv",
            mime="text/csv",
            key=f"before_csv_{base_name}",
        )

    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.download_button(
            label="Download After (PNG)",
            data=matrix_to_png_bytes(after_matrix),
            file_name=f"{base_name}_after.png",
            mime="image/png",
            key=f"after_png_{base_name}",
        )
    with a_col2:
        st.download_button(
            label="Download After (CSV)",
            data=matrix_to_csv_bytes(after_matrix),
            file_name=f"{base_name}_after.csv",
            mime="text/csv",
            key=f"after_csv_{base_name}",
        )

st.session_state.last_results["manual_lab"] = {
    "image_path": selected,
    "operation": op,
    "patch": patch,
}
