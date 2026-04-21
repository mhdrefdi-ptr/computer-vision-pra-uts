from __future__ import annotations

import streamlit as st


DEFAULTS: dict[str, object] = {
    "dataset_root": None,
    "mask_root": None,
    "dataset_valid": False,
    "dataset_structure": {},
    "label_mapping": {},
    "active_split": "train",
    "active_class": "ALL",
    "available_images": [],
    "selected_samples": [],
    "matching_pair": [],
    "selected_image": None,
    "last_results": {},
    "metrics_rows": [],
    "manual_patch": None,
    "manual_matching_data": None,
    "last_export_dir": None,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
