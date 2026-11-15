"""Explainability: which parts of a heartbeat drove the GCN's decision?

Computes per-node gradient x input saliency of the predicted-class logit with
respect to the node features (amplitude, time position), then overlays the
importance on the waveform and its visibility graph. Because every node maps
to one ECG sample, the saliency reads directly as "which segment of the beat
(P wave / QRS / T wave) the model relied on".
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from config import AAMI_CLASSES, CHECKPOINT_DIR, FIGURES_DIR, IDX_TO_CLASS
from model import ECGGCN
from predict import load_model
from preprocessing import Beat
from visibility_graph import beat_to_graph, beat_to_networkx


def node_saliency(waveform, checkpoint_path=CHECKPOINT_DIR / "best_model.pt", target=None, device="cpu"):
    """Returns (saliency[n_nodes] in [0, 1], predicted_label, probs dict)."""
    beat = Beat(record_id=-1, r_peak_sample=len(waveform) // 2, waveform=waveform, label="N")
    data = beat_to_graph(beat)
    model = load_model(checkpoint_path, device)

    batch = Batch.from_data_list([data]).to(device)
    batch.x.requires_grad_(True)
    logits = model(batch.x, batch.edge_index, batch.batch)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    cls = int(logits.argmax()) if target is None else int(target)
    model.zero_grad()
    logits[0, cls].backward()

    sal = (batch.x.grad * batch.x.detach()).abs().sum(dim=1).cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
    return sal, IDX_TO_CLASS[cls], {AAMI_CLASSES[i]: float(probs[i]) for i in range(len(probs))}


def plot_explanation(waveform, saliency, label, probs, out_path=None, ax_list=None):
    """Waveform coloured by saliency (left) + visibility graph with node size
    and colour scaled by saliency (right)."""
    import networkx as nx

    fig = None
    if ax_list is None:
        fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(len(waveform))
    ax_list[0].plot(t, waveform, color="0.7", lw=1, zorder=1)
    sc = ax_list[0].scatter(t, waveform, c=saliency, cmap="inferno", s=14, zorder=2)
    ax_list[0].set(title=f"Saliency on waveform — predicted {label} ({probs[label]:.0%})",
                   xlabel="Sample", ylabel="Amplitude")

    g = beat_to_networkx(Beat(-1, len(waveform) // 2, waveform, "N"))
    pos = {i: (i, waveform[i]) for i in g.nodes}
    nx.draw_networkx_edges(g, pos, ax=ax_list[1], width=0.2, alpha=0.25)
    nx.draw_networkx_nodes(g, pos, ax=ax_list[1], node_size=6 + 60 * saliency,
                           node_color=saliency, cmap="inferno")
    ax_list[1].set_title("Visibility graph (node size/colour = importance)")
    if fig is not None:
        fig.colorbar(sc, ax=ax_list, label="Normalized importance", shrink=0.85)
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    return fig


if __name__ == "__main__":
    from image_digitize import digitize_waveform

    parser = argparse.ArgumentParser(description="Explain a GCN prediction on an ECG beat image.")
    parser.add_argument("image", help="Path to a cropped single-beat ECG image")
    parser.add_argument("--out", default=str(FIGURES_DIR / "explanation.png"))
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    wf = digitize_waveform(args.image)
    sal, label, probs = node_saliency(wf)
    plot_explanation(wf, sal, label, probs, args.out)
    print(f"Predicted {label}; saved explanation to {args.out}")
