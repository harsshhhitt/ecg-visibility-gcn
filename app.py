"""Streamlit demo: upload a photo/screenshot of a single ECG heartbeat and
classify it as Normal or a specific arrhythmia class (AAMI N/S/V/F/Q) using
the visibility-graph GCN pipeline.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import AAMI_CLASSES, CHECKPOINT_DIR  # noqa: E402
from image_digitize import digitize_waveform  # noqa: E402
from predict import predict_beat  # noqa: E402

CLASS_DESCRIPTIONS = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unclassifiable beat",
}

st.set_page_config(page_title="ECG Visibility-Graph GCN", page_icon="\U0001FAC0")
st.title("ECG Beat Classifier — Visibility Graph + GCN")
st.caption(
    "Upload a cropped image of a single heartbeat waveform (one P-QRS-T complex). "
    "The app digitizes the trace, builds a visibility graph from it, and classifies "
    "it with the trained GCN."
)

if not (CHECKPOINT_DIR / "best_model.pt").exists():
    st.error(
        f"No trained model found at {CHECKPOINT_DIR / 'best_model.pt'}. "
        "Run `python src/train.py` first."
    )
    st.stop()

st.warning(
    "Digitization is best-effort: without a calibration reference on the image, "
    "the recovered signal's amplitude scale is approximate, not a true pixel-to-mV "
    "conversion. Predictions should be read as a demo, not a diagnostic result.",
    icon="⚠️",
)

uploaded = st.file_uploader("ECG waveform image", type=["png", "jpg", "jpeg"])

threshold = st.slider(
    "Trace darkness threshold (lower = stricter, only very dark pixels count as trace)",
    min_value=30, max_value=220, value=130, step=5,
)

if uploaded is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

    try:
        waveform = digitize_waveform(uploaded, threshold=threshold)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with col2:
        st.subheader("Digitized waveform")
        fig, ax = plt.subplots()
        ax.plot(waveform)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Normalized amplitude")
        st.pyplot(fig)
        plt.close(fig)

    label, probs, data, nx_graph = predict_beat(waveform)

    st.subheader("Prediction")
    if label == "N":
        st.success(f"**Normal** (AAMI class N)")
    else:
        st.error(f"**Abnormal** — AAMI class **{label}**: {CLASS_DESCRIPTIONS[label]}")

    st.bar_chart({c: probs[c] for c in AAMI_CLASSES})

    with st.expander("Visibility graph structure"):
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = {i: (i, waveform[i]) for i in nx_graph.nodes}
        nx.draw(nx_graph, pos, ax=ax, node_size=15, width=0.5, with_labels=False)
        ax.set_title(f"{data.num_nodes} nodes, {data.num_edges // 2} edges")
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Upload an ECG waveform image to get a prediction.")
