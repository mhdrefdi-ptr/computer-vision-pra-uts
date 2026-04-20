from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.dataset_manager import (
    download_dataset,
    ensure_mask_root_structure,
    find_dataset_root,
    get_mask_root,
    list_classes,
    list_images,
    load_label_mapping,
    mask_coverage_summary,
    prepare_local_dataset,
    resolve_mask_path,
    summarize_split_counts,
    validate_structure,
)
from core.image_ops import bgr_to_rgb, read_image, resize_max

init_state()


st.title("Dataset Manager")
st.caption("Download dataset Kaggle, validasi struktur, pilih split/kelas, dan pilih sample.")

col_a, col_b = st.columns([2, 1])
with col_a:
    force = st.checkbox("Force download (abaikan cache kagglehub)", value=False)
with col_b:
    if st.button("Download Dataset dari Kaggle", type="primary"):
        with st.spinner("Mengunduh dataset dari Kaggle..."):
            raw_path = download_dataset(force_download=force)
            detected = find_dataset_root(raw_path)
            local_path = prepare_local_dataset(detected, refresh_existing=force)
            structure = validate_structure(local_path)
            mask_root = get_mask_root(local_path)
            ensure_mask_root_structure(mask_root)

            st.session_state.dataset_root = str(local_path)
            st.session_state.mask_root = str(mask_root)
            st.session_state.dataset_structure = structure
            st.session_state.dataset_valid = structure["valid"]
            st.session_state.label_mapping = load_label_mapping(local_path)

        if structure["valid"]:
            st.success("Dataset berhasil diunduh dan tervalidasi.")
        else:
            st.error("Dataset terunduh, tetapi struktur belum sesuai PRD.")

if st.session_state.dataset_root:
    dataset_root = Path(st.session_state.dataset_root)
    mask_root = Path(st.session_state.mask_root) if st.session_state.mask_root else get_mask_root(dataset_root)
    ensure_mask_root_structure(mask_root)
    st.session_state.mask_root = str(mask_root)
    structure = st.session_state.dataset_structure or validate_structure(dataset_root)

    st.subheader("Ringkasan Dataset")
    st.json(structure)
    st.caption(f"Mask root persisten: `{mask_root}`")

    if not structure.get("valid", False):
        for err in structure.get("errors", []):
            st.error(err)
        st.stop()

    split_counts = summarize_split_counts(dataset_root)
    st.write("Jumlah citra per split")
    st.dataframe(split_counts, use_container_width=True)

    st.write("Coverage mask per split")
    coverage = mask_coverage_summary(dataset_root, mask_root)
    st.dataframe(coverage, use_container_width=True)

    split = st.selectbox("Pilih split", ["train", "validation", "test"], index=["train", "validation", "test"].index(st.session_state.active_split))
    st.session_state.active_split = split

    classes = list_classes(dataset_root, split)
    class_options = ["ALL", *classes]
    if st.session_state.active_class not in class_options:
        st.session_state.active_class = "ALL"
    selected_class = st.selectbox("Pilih kelas", class_options, index=class_options.index(st.session_state.active_class))
    st.session_state.active_class = selected_class

    images = list_images(dataset_root, split, selected_class)
    st.session_state.available_images = [str(p) for p in images]
    st.write(f"Total image terfilter: {len(images)}")

    if st.button("Buat folder mask missing untuk filter aktif"):
        created = 0
        for image_path in images:
            target = resolve_mask_path(image_path, dataset_root, mask_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            created += 1
        st.success(f"Folder mask dipastikan untuk {created} image terfilter.")

    preview_count = st.slider("Jumlah preview", min_value=6, max_value=60, value=18, step=6)
    cols = st.columns(6)
    for idx, image_path in enumerate(images[:preview_count]):
        with cols[idx % 6]:
            try:
                img = resize_max(read_image(image_path), 280)
                st.image(bgr_to_rgb(img), caption=image_path.name)
            except Exception as exc:
                st.warning(f"Gagal load {image_path.name}: {exc}")

    image_strs = [str(p) for p in images]
    default_samples = [p for p in st.session_state.selected_samples if p in image_strs][:3]

    samples = st.multiselect(
        "Pilih 2-3 citra sampel",
        image_strs,
        default=default_samples,
        max_selections=3,
        format_func=lambda p: Path(p).name,
    )
    st.session_state.selected_samples = samples

    if samples:
        st.session_state.selected_image = samples[0]

    st.markdown("### Pair untuk Feature Matching")
    if len(image_strs) >= 2:
        default_pair = st.session_state.matching_pair if len(st.session_state.matching_pair) == 2 else image_strs[:2]
        a = st.selectbox("Image A", image_strs, index=image_strs.index(default_pair[0]), format_func=lambda p: Path(p).name)
        b = st.selectbox("Image B", image_strs, index=image_strs.index(default_pair[1]), format_func=lambda p: Path(p).name)
        st.session_state.matching_pair = [a, b]
else:
    st.info("Dataset belum disiapkan. Klik tombol download terlebih dahulu.")
