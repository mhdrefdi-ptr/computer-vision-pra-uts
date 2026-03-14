from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from modules.convolution import apply_gaussian_blur, apply_prewitt, apply_sobel
from modules.dataset_loader import create_thumbnail, get_image_list, load_image
from modules.evaluation import evaluate_segmentation
from modules.feature_matching import detect_orb_features, match_features
from modules.harris_corner import detect_harris_corners
from modules.morphology import apply_dilation, apply_erosion, apply_fill_holes
from modules.preprocessing import apply_clahe
from modules.thresholding import apply_binary_threshold
from utils.image_utils import bgr_to_rgb, load_grayscale_mask, resize_max


DATASET_DIR = Path("dataset/images")
MASK_DIR = Path("dataset/masks")


@st.cache_data
def cached_load_resized_image(path_str: str, max_dim: int = 512):
    image = load_image(path_str)
    return resize_max(image, max_dim=max_dim)


def find_mask_path(image_name: str) -> Path | None:
    image_rel = Path(image_name)
    image_stem_rel = image_rel.with_suffix("")

    # Prefer mirror structure under dataset/masks (e.g. images/a/x.jpg -> masks/a/x.png)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = MASK_DIR / image_stem_rel.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Backward-compatible fallback: flat masks folder by basename
    stem = image_rel.stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = MASK_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def process_pipeline(image_bgr, threshold_value: int):
    clahe_img = apply_clahe(image_bgr)
    threshold_img = apply_binary_threshold(clahe_img, threshold_value)
    harris_img = detect_harris_corners(image_bgr)
    gaussian_img = apply_gaussian_blur(clahe_img)
    sobel_img = apply_sobel(clahe_img)
    prewitt_img = apply_prewitt(clahe_img)
    erosion_img = apply_erosion(threshold_img)
    dilation_img = apply_dilation(erosion_img)
    filled_img = apply_fill_holes(dilation_img)

    return {
        "clahe": clahe_img,
        "threshold": threshold_img,
        "harris": harris_img,
        "gaussian": gaussian_img,
        "sobel": sobel_img,
        "prewitt": prewitt_img,
        "morphology": filled_img,
    }


st.set_page_config(page_title="Computer Vision Web App", layout="wide")
st.title("Computer Vision Web App")
st.caption("Streamlit + OpenCV: Contrast Enhancement, Thresholding, Harris, Convolution, Morphology, ORB, Evaluation")

image_names = get_image_list(DATASET_DIR)
if "selected_image" not in st.session_state:
    st.session_state.selected_image = image_names[0] if image_names else None

tabs = st.tabs(["Galeri Dataset", "Analisis Gambar", "Feature Matching", "Evaluasi"])


with tabs[0]:
    st.subheader("Dataset Gallery")
    if not image_names:
        st.info("Folder dataset/images masih kosong. Tambahkan gambar untuk mulai analisis.")
    else:
        cols_per_row = 4
        for start in range(0, len(image_names), cols_per_row):
            row = st.columns(cols_per_row)
            for idx, image_name in enumerate(image_names[start : start + cols_per_row]):
                image_path = DATASET_DIR / image_name
                image_bgr = cached_load_resized_image(str(image_path), max_dim=512)
                thumb = create_thumbnail(image_bgr, max_size=180)
                with row[idx]:
                    st.image(thumb, caption=image_name)
                    if st.button(f"Pilih {image_name}", key=f"pick_{image_name}"):
                        st.session_state.selected_image = image_name
                        st.success(f"{image_name} dipilih. Buka tab Analisis Gambar.")


with tabs[1]:
    st.subheader("Analisis Gambar")
    if not image_names:
        st.info("Belum ada gambar di dataset/images.")
    else:
        selected = st.selectbox(
            "Pilih gambar untuk dianalisis",
            image_names,
            index=image_names.index(st.session_state.selected_image)
            if st.session_state.selected_image in image_names
            else 0,
        )
        st.session_state.selected_image = selected
        threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=127)

        selected_path = DATASET_DIR / selected
        image_bgr = cached_load_resized_image(str(selected_path), max_dim=512)
        outputs = process_pipeline(image_bgr, threshold_value=threshold_value)

        col1, col2 = st.columns(2)
        col1.image(bgr_to_rgb(image_bgr), caption="Original Image")
        col2.image(outputs["clahe"], caption="Preprocessing (CLAHE)", clamp=True)

        col3, col4 = st.columns(2)
        col3.image(outputs["threshold"], caption="Thresholding", clamp=True)
        col4.image(bgr_to_rgb(outputs["harris"]), caption="Harris Corner")

        col5, col6 = st.columns(2)
        col5.image(outputs["gaussian"], caption="Gaussian Blur", clamp=True)
        col6.image(outputs["sobel"], caption="Sobel", clamp=True)

        col7, col8 = st.columns(2)
        col7.image(outputs["prewitt"], caption="Prewitt", clamp=True)
        col8.image(outputs["morphology"], caption="Morphology (Erosi + Dilasi + Fill Holes)", clamp=True)


with tabs[2]:
    st.subheader("Feature Matching (ORB)")
    if not image_names:
        st.info("Belum ada gambar di dataset/images.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            image_a = st.selectbox("Image A", image_names, key="image_a")
        with col_b:
            image_b = st.selectbox("Image B", image_names, key="image_b")

        path_a = DATASET_DIR / image_a
        path_b = DATASET_DIR / image_b
        img_a = cached_load_resized_image(str(path_a), max_dim=512)
        img_b = cached_load_resized_image(str(path_b), max_dim=512)

        show_a, show_b = st.columns(2)
        show_a.image(bgr_to_rgb(img_a), caption="Image A")
        show_b.image(bgr_to_rgb(img_b), caption="Image B")

        if st.button("Jalankan ORB Matching"):
            _, _, kp_vis_a = detect_orb_features(img_a)
            _, _, kp_vis_b = detect_orb_features(img_b)
            matches, match_vis = match_features(img_a, img_b, top_k=50)

            kp_col1, kp_col2 = st.columns(2)
            kp_col1.image(bgr_to_rgb(kp_vis_a), caption="Keypoints Image A")
            kp_col2.image(bgr_to_rgb(kp_vis_b), caption="Keypoints Image B")

            if match_vis is None:
                st.warning("Tidak ditemukan descriptor yang cukup untuk matching.")
            else:
                st.image(bgr_to_rgb(match_vis), caption=f"Feature Matching Result (Top {len(matches)} matches)")


with tabs[3]:
    st.subheader("Evaluasi")
    if not image_names:
        st.info("Belum ada gambar di dataset/images.")
    else:
        threshold_eval = st.slider("Threshold untuk evaluasi", min_value=0, max_value=255, value=127, key="threshold_eval")
        rows = []

        for image_name in image_names:
            image_path = DATASET_DIR / image_name
            mask_path = find_mask_path(image_name)
            if mask_path is None:
                continue

            image_bgr = cached_load_resized_image(str(image_path), max_dim=512)
            clahe_img = apply_clahe(image_bgr)
            threshold_mask = apply_binary_threshold(clahe_img, threshold_eval)
            morph_mask = apply_fill_holes(apply_dilation(apply_erosion(threshold_mask)))
            true_mask = load_grayscale_mask(mask_path, target_shape=threshold_mask.shape)

            eval_threshold = evaluate_segmentation(threshold_mask, true_mask)
            eval_morph = evaluate_segmentation(morph_mask, true_mask)

            rows.append(
                {
                    "Image": image_name,
                    "Method": "Thresholding",
                    "Precision": eval_threshold["Precision"],
                    "Recall": eval_threshold["Recall"],
                    "F1-score": eval_threshold["F1-score"],
                    "IoU": eval_threshold["IoU"],
                }
            )
            rows.append(
                {
                    "Image": image_name,
                    "Method": "Threshold+Morph",
                    "Precision": eval_morph["Precision"],
                    "Recall": eval_morph["Recall"],
                    "F1-score": eval_morph["F1-score"],
                    "IoU": eval_morph["IoU"],
                }
            )

        if not rows:
            st.info("Belum ada pasangan mask di dataset/masks dengan nama file yang sama seperti gambar.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            summary = df.groupby("Method")[["Precision", "Recall", "F1-score", "IoU"]].mean()
            st.write("Rata-rata metrik per metode")
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary)
