from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_state import init_state
from core.exporter import create_export_dir, save_image, save_manual_matrix, save_metrics

init_state()


st.title("Export & Reporting")
st.caption("Ekspor gambar hasil, tabel metrik, dan matriks manual (CSV/Excel/PNG).")

if st.button("Generate Export Folder", type="primary"):
    export_dir = create_export_dir(prefix="cv_workbench")
    st.session_state.last_export_dir = str(export_dir)

if not st.session_state.last_export_dir:
    st.info("Klik 'Generate Export Folder' terlebih dahulu.")
    st.stop()

export_dir = Path(st.session_state.last_export_dir)
st.success(f"Folder export aktif: {export_dir}")

written_files: list[Path] = []

last_results = st.session_state.last_results
for module_name, payload in last_results.items():
    if not isinstance(payload, dict):
        continue
    for key, value in payload.items():
        if hasattr(value, "ndim"):
            filename = export_dir / f"{module_name}_{key}.png"
            written_files.append(save_image(filename, value))

if st.session_state.metrics_rows:
    csv_path, xlsx_path = save_metrics(export_dir, st.session_state.metrics_rows)
    written_files.extend([csv_path, xlsx_path])

if st.session_state.manual_patch is not None:
    csv_path, xlsx_path, png_path = save_manual_matrix(export_dir, st.session_state.manual_patch, "manual_patch_15x15")
    written_files.extend([csv_path, xlsx_path, png_path])

st.subheader("File yang dihasilkan")
for p in sorted(set(written_files), key=str):
    st.markdown(f"- `{p}`")

if written_files:
    for p in sorted(set(written_files), key=str):
        with p.open("rb") as f:
            st.download_button(
                label=f"Download {p.name}",
                data=f,
                file_name=p.name,
                mime="application/octet-stream",
                key=f"download_{p.name}",
            )
