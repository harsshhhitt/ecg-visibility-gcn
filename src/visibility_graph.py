"""Stage 4: Visibility graph construction — beat waveform -> PyG Data object.

Uses ts2vg's Natural Visibility Graph (amplitude-sensitive), chosen over the
Horizontal VG because HVG discards amplitude-scale information that's
diagnostically relevant (e.g. QRS amplitude differences between beat types).
"""

import logging

import networkx as nx
import numpy as np
import torch
from ts2vg import NaturalVG
from torch_geometric.data import Data

from config import CLASS_TO_IDX
from preprocessing import Beat

logger = logging.getLogger(__name__)


def beat_to_graph(beat: Beat) -> Data:
    """Convert a single beat waveform into a PyG Data object via NVG.

    Node features: [amplitude, normalized_time_position].
    Edges: undirected (both directions added to edge_index), since NVG
    visibility is symmetric even though ts2vg returns directed pairs.
    """
    waveform = beat.waveform
    n = len(waveform)

    vg = NaturalVG()
    vg.build(waveform)
    edges = vg.edges  # list of (i, j) pairs, i < j

    if edges:
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    amplitude = waveform.astype(np.float32)
    position = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = torch.tensor(np.stack([amplitude, position], axis=1), dtype=torch.float)

    y = torch.tensor([CLASS_TO_IDX[beat.label]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.record_id = beat.record_id
    data.r_peak_sample = beat.r_peak_sample
    return data


def beat_to_networkx(beat: Beat) -> nx.Graph:
    """Build a NetworkX graph for visualization only (not used in training)."""
    vg = NaturalVG()
    vg.build(beat.waveform)
    return vg.as_networkx()


def add_degree_features(data: Data) -> Data:
    """Per-beat z-score the amplitude channel (removes patient/lead gain and offset, which a
    visibility graph is already invariant to) and append normalized degree and log-degree
    (2 -> 4 channels). Idempotent: a graph that already has 4 channels is returned unchanged."""
    if data.x.shape[1] >= 4:
        return data
    n = data.num_nodes
    deg = torch.bincount(data.edge_index[0], minlength=n).float()
    extra = torch.stack([deg / max(n - 1, 1), torch.log1p(deg) / np.log(max(n, 2))], dim=1)
    amp = data.x[:, :1]
    amp = (amp - amp.mean()) / (amp.std() + 1e-6)
    data = data.clone()
    data.x = torch.cat([amp, data.x[:, 1:2], extra], dim=1)
    return data


def graph_stats(data: Data) -> dict:
    """Summary topology of a beat's visibility graph (edge_index holds both directions)."""
    n = data.num_nodes
    m = data.edge_index.shape[1] // 2
    deg = torch.bincount(data.edge_index[0], minlength=n).float()
    return {
        "nodes": n,
        "edges": m,
        "density": float(2 * m / max(n * (n - 1), 1)),
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "hub_position": float(deg.argmax()) / max(n - 1, 1),  # where the biggest hub sits (0..1)
    }


if __name__ == "__main__":
    import numpy as np

    synthetic = Beat(
        record_id=0, r_peak_sample=90,
        waveform=np.sin(np.linspace(0, 4 * np.pi, 200)),
        label="N",
    )
    data = beat_to_graph(synthetic)
    logger.info("Synthetic beat graph: %s", data)
