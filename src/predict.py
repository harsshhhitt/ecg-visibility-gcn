"""Single-beat inference: waveform array -> visibility graph -> GCN prediction.

Used by the Streamlit demo app (app.py) and usable standalone for scripting.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from config import AAMI_CLASSES, IDX_TO_CLASS
from ensemble import load_classifier, prior_adjust_bias
from preprocessing import Beat
from visibility_graph import add_degree_features, beat_to_graph, beat_to_networkx

logger = logging.getLogger(__name__)

_model_cache = {}


def load_model(checkpoint_path=None, device="cpu"):
    """Returns the calibrated ensemble classifier (falls back to the legacy single GCN)."""
    key = str(checkpoint_path)
    if key not in _model_cache:
        _model_cache[key] = load_classifier(checkpoint_path, device)
    return _model_cache[key]


def set_sensitivity(tau, checkpoint_path=None, device="cpu"):
    """Move the decision threshold: tau=0 favours overall accuracy (predict the common class),
    tau=1 is the class-balanced rule, larger tau favours catching rare classes at the cost of
    more false alarms. No-op for a legacy single-model checkpoint."""
    clf = load_model(checkpoint_path, device)
    counts = getattr(clf, "train_counts", None)
    if counts is not None:
        clf.bias.copy_(torch.tensor(prior_adjust_bias(counts, tau), dtype=torch.float))
    return clf


def build_graph(waveform, clf):
    """Waveform -> PyG Data with the node-feature layout the classifier was trained on."""
    beat = Beat(record_id=-1, r_peak_sample=len(waveform) // 2, waveform=waveform, label="N")
    data = beat_to_graph(beat)
    return add_degree_features(data) if clf.degree_features else data


def predict_beat(waveform: np.ndarray, checkpoint_path=None, device="cpu"):
    """Returns (predicted_label, class_probabilities dict, PyG Data, NetworkX graph)."""
    model = load_model(checkpoint_path, device)
    beat = Beat(record_id=-1, r_peak_sample=len(waveform) // 2, waveform=waveform, label="N")
    data = build_graph(waveform, model)
    nx_graph = beat_to_networkx(beat)

    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch.x, batch.edge_index, batch.batch), dim=1).squeeze(0).cpu().numpy()

    pred_label = IDX_TO_CLASS[int(probs.argmax())]
    prob_dict = {AAMI_CLASSES[i]: float(probs[i]) for i in range(len(AAMI_CLASSES))}
    return pred_label, prob_dict, data, nx_graph


def predict_with_uncertainty(waveform, n_samples=30, checkpoint_path=None, device="cpu"):
    """Monte-Carlo dropout over the whole ensemble: dropout stays active and n_samples
    passes are averaged. Returns (mean_probs dict, std_probs dict, predictive_entropy in [0, 1]).
    The std includes both dropout noise and disagreement between ensemble members."""
    model = load_model(checkpoint_path, device)
    batch = Batch.from_data_list([build_graph(waveform, model)]).to(device)
    model.train()  # enable dropout (no batchnorm in these models, so this is safe)
    try:
        with torch.no_grad():
            runs = torch.stack([
                torch.softmax(model(batch.x, batch.edge_index, batch.batch), dim=1).squeeze(0)
                for _ in range(n_samples)
            ]).cpu().numpy()
    finally:
        model.eval()
    mean, std = runs.mean(0), runs.std(0)
    entropy = float(-(mean * np.log(mean + 1e-12)).sum() / np.log(len(mean)))
    return ({AAMI_CLASSES[i]: float(mean[i]) for i in range(len(mean))},
            {AAMI_CLASSES[i]: float(std[i]) for i in range(len(std))}, entropy)


def main():
    """Batch CLI: classify every image in a file/folder and optionally write a CSV."""
    import argparse
    import csv

    from image_digitize import digitize_waveform

    parser = argparse.ArgumentParser(description="Classify ECG beat image(s).")
    parser.add_argument("path", help="Image file or folder of images")
    parser.add_argument("--csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    root = Path(args.path)
    files = sorted(p for p in ([root] if root.is_file() else root.iterdir())
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    rows = []
    for f in files:
        wf = digitize_waveform(str(f))
        label, probs, _, _ = predict_beat(wf)
        _, _, unc = predict_with_uncertainty(wf)
        rows.append({"file": f.name, "label": label, "confidence": round(probs[label], 4),
                     "uncertainty": round(unc, 4), **{f"p_{c}": round(v, 4) for c, v in probs.items()}})
        print(f"{f.name}: {label} ({probs[label]:.1%}, uncertainty {unc:.2f})")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit
    synthetic = np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float32)
    label, probs, _, _ = predict_beat(synthetic)
    logger.info("Predicted label=%s probs=%s", label, probs)
