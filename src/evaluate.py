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
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader

from config import AAMI_CLASSES, CHECKPOINT_DIR, FIGURES_DIR, IDX_TO_CLASS, PROCESSED_DATA_DIR, REPORTS_DIR
from dataset import ECGVisibilityGraphDataset
from ensemble import load_classifier
from visibility_graph import add_degree_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels, all_raw = [], [], []
    for batch in loader:
        batch = batch.to(device)
        raw = model.raw_logits(batch.x, batch.edge_index, batch.batch)  # ensemble-mean logits, uncalibrated
        out = raw / model.temperature + model.bias
        all_probs.append(torch.softmax(out, dim=1).cpu())
        all_raw.append(raw.cpu())
        all_labels.append(batch.y.cpu())
    probs = torch.cat(all_probs).numpy()
    predict.last_raw = torch.cat(all_raw).numpy()
    return probs.argmax(axis=1), torch.cat(all_labels).numpy(), probs


def per_class_sens_spec(cm):
    """One-vs-rest sensitivity (recall) and specificity from a confusion matrix."""
    total = cm.sum()
    out = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[i] = (tp / max(tp + fn, 1), tn / max(tn + fp, 1))
    return out


def plot_roc_pr(labels, probs, class_names, out_path):
    """One-vs-rest ROC and precision-recall curves; returns per-class AUROC/AUPRC."""
    y = label_binarize(labels, classes=list(range(len(class_names))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stats = {}
    for i, name in enumerate(class_names):
        if y[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(y[:, i], probs[:, i])
        auroc = roc_auc_score(y[:, i], probs[:, i])
        auprc = average_precision_score(y[:, i], probs[:, i])
        stats[name] = {"auroc": float(auroc), "auprc": float(auprc)}
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={auprc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (one-vs-rest, DS2)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (one-vs-rest, DS2)")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stats


def plot_confusion_matrix(cm, class_names, out_path, normalize=False):
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (DS2, inter-patient)" + (" — row-normalized" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j])), ha="center", va="center",
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
    model = load_classifier(checkpoint_path, device)  # ensemble if present, else legacy checkpoint

    test_set = ECGVisibilityGraphDataset(
        root=str(PROCESSED_DATA_DIR), split="ds2",
        transform=add_degree_features if model.degree_features else None)
    test_loader = DataLoader(test_set, batch_size=128)

    preds, labels, probs = predict(model, test_loader, device)

    class_indices = list(range(len(AAMI_CLASSES)))
    report = classification_report(
        labels, preds, labels=class_indices, target_names=AAMI_CLASSES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=class_indices)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORTS_DIR / "ds2_logits.npz", raw_logits=predict.last_raw, labels=labels,
                        temperature=float(model.temperature), bias=model.bias.numpy())

    sens_spec = per_class_sens_spec(cm)
    curve_stats = plot_roc_pr(labels, probs, AAMI_CLASSES, REPORTS_DIR / "roc_pr_curves.png")
    per_class = {c: dict(report[c]) for c in AAMI_CLASSES}
    for i, c in enumerate(AAMI_CLASSES):
        per_class[c]["sensitivity"], per_class[c]["specificity"] = map(float, sens_spec[i])
        per_class[c].update(curve_stats.get(c, {}))

    metrics = {
        "accuracy": report["accuracy"],
        "balanced_accuracy": float(np.mean([sens_spec[i][0] for i in class_indices if cm[i].sum() > 0])),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "mcc": float(matthews_corrcoef(labels, preds)),
        "macro_auroc": float(np.mean([v["auroc"] for v in curve_stats.values()])),
        "per_class": per_class,
        "reference_macro_f1": reference_macro_f1,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, AAMI_CLASSES, REPORTS_DIR / "confusion_matrix.png")
    plot_confusion_matrix(cm, AAMI_CLASSES, REPORTS_DIR / "confusion_matrix_normalized.png", normalize=True)

    logger.info("Accuracy=%.4f BalancedAcc=%.4f Macro-F1=%.4f MacroAUROC=%.4f MCC=%.4f",
                metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"],
                metrics["macro_auroc"], metrics["mcc"])
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
    from pathlib import Path
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    evaluate(checkpoint_path=ckpt, reference_macro_f1=args.reference_macro_f1)
