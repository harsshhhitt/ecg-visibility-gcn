"""Stage 5: Evaluation & reporting — metrics, confusion matrix, qualitative plots.

Runs the trained model on the held-out DS2 (inter-patient) test set and
writes outputs/reports/metrics.json + outputs/reports/confusion_matrix.png,
plus a handful of correctly/incorrectly classified waveform-vs-graph plots
per class into outputs/figures/.
"""

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from config import AAMI_CLASSES, CHECKPOINT_DIR, FIGURES_DIR, IDX_TO_CLASS, PROCESSED_DATA_DIR, REPORTS_DIR
from dataset import ECGVisibilityGraphDataset
from model import ECGGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        all_preds.append(out.argmax(dim=1).cpu())
        all_labels.append(batch.y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (DS2, inter-patient)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_waveform_and_graph(waveform, nx_graph, title, out_path):
    """Side-by-side waveform vs. visibility-graph plot for qualitative review."""
    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(waveform)
    axes[0].set_title(f"Waveform — {title}")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")

    pos = {i: (i, waveform[i]) for i in nx_graph.nodes}
    nx.draw(nx_graph, pos, ax=axes[1], node_size=15, width=0.5, with_labels=False)
    axes[1].set_title(f"Visibility Graph — {title}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(checkpoint_path=None, device=None, reference_macro_f1=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "best_model.pt")

    test_set = ECGVisibilityGraphDataset(root=str(PROCESSED_DATA_DIR), split="ds2")
    test_loader = DataLoader(test_set, batch_size=128)

    model = ECGGCN(in_channels=2, num_classes=len(AAMI_CLASSES)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, labels = predict(model, test_loader, device)

    class_indices = list(range(len(AAMI_CLASSES)))
    report = classification_report(
        labels, preds, labels=class_indices, target_names=AAMI_CLASSES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=class_indices)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: report[c] for c in AAMI_CLASSES},
        "reference_macro_f1": reference_macro_f1,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, AAMI_CLASSES, REPORTS_DIR / "confusion_matrix.png")

    logger.info("Accuracy=%.4f Macro-F1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    if reference_macro_f1 is not None:
        logger.info("Reference paper macro-F1=%.4f (delta=%.4f)",
                     reference_macro_f1, metrics["macro_f1"] - reference_macro_f1)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ECG GCN on the DS2 test split.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reference-macro-f1", type=float, default=None,
                         help="Published reference macro-F1 to compare against.")
    args = parser.parse_args()
    ckpt = args.checkpoint or (CHECKPOINT_DIR / "best_model.pt")
    evaluate(checkpoint_path=ckpt, reference_macro_f1=args.reference_macro_f1)
