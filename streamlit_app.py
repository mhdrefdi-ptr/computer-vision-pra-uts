from __future__ import annotations

import streamlit as st

from core.app_state import init_state


st.set_page_config(
    page_title="CV Workbench - Analisis Citra Pra-UTS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.title("CV Workbench - Analisis Citra Pra-UTS")
st.caption("Pipeline Computer Vision klasik untuk eksperimen, evaluasi IoU, dan manual calculation 15x15.")

with st.sidebar:
    st.subheader("Status Session")
    st.write(f"Dataset: {st.session_state.dataset_root or '-'}")
    st.write(f"Mask root: {st.session_state.mask_root or '-'}")
    st.write(f"Valid: {'Ya' if st.session_state.dataset_valid else 'Belum'}")
    st.write(f"Split aktif: {st.session_state.active_split}")
    st.write(f"Kelas aktif: {st.session_state.active_class}")
    st.write(f"Sample dipilih: {len(st.session_state.selected_samples)}")

st.markdown(
    """
### Cara pakai singkat
1. Buka **Dataset Manager** untuk download dan validasi dataset.
2. Pilih split, kelas, lalu pilih 2-3 sample.
3. Buka halaman preprocessing, teknik CV, konvolusi, morfologi, matching, evaluasi.
4. Buka **Manual Calculation Lab** untuk patch 15x15.
5. Buka **Export & Reporting** untuk unduh hasil.
"""
)
