"""Single-beat inference: waveform array -> visibility graph -> GCN prediction.

Used by the Streamlit demo app (app.py) and usable standalone for scripting.
"""

import logging

import numpy as np
import torch
from torch_geometric.data import Batch

from config import AAMI_CLASSES, CHECKPOINT_DIR, IDX_TO_CLASS
from model import ECGGCN
from preprocessing import Beat
from visibility_graph import beat_to_graph, beat_to_networkx

logger = logging.getLogger(__name__)

_model_cache = {}


def load_model(checkpoint_path=CHECKPOINT_DIR / "best_model.pt", device="cpu"):
    key = str(checkpoint_path)
    if key not in _model_cache:
        model = ECGGCN(in_channels=2, num_classes=len(AAMI_CLASSES)).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def predict_beat(waveform: np.ndarray, checkpoint_path=CHECKPOINT_DIR / "best_model.pt", device="cpu"):
    """Returns (predicted_label, class_probabilities dict, PyG Data, NetworkX graph)."""
    beat = Beat(record_id=-1, r_peak_sample=len(waveform) // 2, waveform=waveform, label="N")
    data = beat_to_graph(beat)
    nx_graph = beat_to_networkx(beat)

    model = load_model(checkpoint_path, device)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = IDX_TO_CLASS[pred_idx]
    prob_dict = {AAMI_CLASSES[i]: float(probs[i]) for i in range(len(AAMI_CLASSES))}
    return pred_label, prob_dict, data, nx_graph


if __name__ == "__main__":
    synthetic = np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float32)
    label, probs, _, _ = predict_beat(synthetic)
    logger.info("Predicted label=%s probs=%s", label, probs)
