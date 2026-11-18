"""Streamlit demo: upload a photo/screenshot of a single ECG heartbeat and
classify it as Normal or a specific arrhythmia class (AAMI N/S/V/F/Q) using
the visibility-graph GCN pipeline.

Run with:
    streamlit run app.py
"""

import io
import json
import tempfile
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import AAMI_CLASSES, CHECKPOINT_DIR  # noqa: E402
from image_digitize import digitize_waveform  # noqa: E402
from predict import predict_beat, predict_with_uncertainty, set_sensitivity  # noqa: E402
from visibility_graph import graph_stats  # noqa: E402
from report import build_report  # noqa: E402
from explain import node_saliency, plot_explanation  # noqa: E402

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

SAMPLES_DIR = Path(__file__).parent / "samples"
sample_files = sorted(SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
sample_choice = st.sidebar.selectbox("Or try a bundled sample", ["(none)"] + [p.name for p in sample_files])
tau = st.sidebar.slider(
    "Detection sensitivity", 0.0, 1.5, 1.0, step=0.1,
    help="0 = favour overall accuracy (mostly predicts the common class). 1 = class-balanced (default, "
         "chosen on held-out patients). Higher = flag rare beat types more readily, at the cost of more false alarms.")
set_sensitivity(tau)
mc_samples = st.sidebar.slider("MC-dropout passes (uncertainty)", 10, 100, 30, step=10)

uploaded = st.file_uploader("ECG waveform image", type=["png", "jpg", "jpeg"])
if uploaded is None and sample_choice != "(none)":
    uploaded = open(SAMPLES_DIR / sample_choice, "rb")

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
    st.metric("Confidence", f"{probs[label]:.1%}")

    mean_p, std_p, entropy = predict_with_uncertainty(waveform, n_samples=mc_samples)
    stats = graph_stats(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uncertainty (entropy)", f"{entropy:.2f}", help="0 = certain, 1 = maximally uncertain (MC dropout)")
    c2.metric("Graph nodes", stats["nodes"])
    c3.metric("Graph edges", stats["edges"])
    c4.metric("Mean degree", f"{stats['mean_degree']:.1f}")
    if entropy > 0.6:
        st.warning("High predictive uncertainty - treat this prediction with extra caution.")

    report = {"prediction": label, "probabilities": probs, "mc_dropout_mean": mean_p,
              "mc_dropout_std": std_p, "uncertainty_entropy": entropy, "graph": stats}
    st.download_button("Download JSON report", json.dumps(report, indent=2),
                       file_name="ecg_prediction.json", mime="application/json")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "report.pdf"
        uploaded.seek(0)
        build_report(waveform, pdf_path, source_name=getattr(uploaded, "name", "uploaded image"),
                     source_image=io.BytesIO(uploaded.read()), mc_samples=mc_samples)
        pdf_bytes = pdf_path.read_bytes()
    st.download_button("Download PDF report", pdf_bytes, file_name="ecg_report.pdf", mime="application/pdf")

    with st.expander("Why this prediction? (gradient x input saliency)", expanded=True):
        st.caption(
            "Importance of each ECG sample (graph node) to the predicted-class logit. "
            "Bright regions are the parts of the beat the GCN relied on most."
        )
        sal, _, _ = node_saliency(waveform)
        expl_fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_explanation(waveform, sal, label, probs, ax_list=axes)
        st.pyplot(expl_fig)
        plt.close(expl_fig)

    with st.expander("Visibility graph structure"):
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = {i: (i, waveform[i]) for i in nx_graph.nodes}
        nx.draw(nx_graph, pos, ax=ax, node_size=15, width=0.5, with_labels=False)
        ax.set_title(f"{data.num_nodes} nodes, {data.num_edges // 2} edges")
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Upload an ECG waveform image to get a prediction.")
