"""Accuracy vs. rare-class detection trade-off on DS2, from saved logits (no model re-run).

Sweeps the logit-adjustment strength tau. This is an *analysis* of the test set, shown so the
operating-point choice is transparent; the shipped tau was chosen on validation patients only.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score

from config import AAMI_CLASSES, REPORTS_DIR
from ensemble import prior_adjust_bias


def main():
    d = np.load(REPORTS_DIR / "ds2_logits.npz")
    raw, y, T = d["raw_logits"], d["labels"], float(d["temperature"])
    counts = json.load(open(REPORTS_DIR / "operating_point_val.json"))["train_counts"]
    taus = np.round(np.arange(0.0, 2.51, 0.1), 2)
    rows = []
    for tau in taus:
        pred = (raw / T + prior_adjust_bias(counts, tau)).argmax(1)
        rec = recall_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
        rows.append({"tau": float(tau), "accuracy": float((pred == y).mean()),
                     "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
                     "recall": dict(zip("NSVF", map(float, rec)))})
    json.dump(rows, open(REPORTS_DIR / "operating_curve_ds2.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(taus, [r["accuracy"] for r in rows], label="Accuracy", lw=2)
    axes[0].plot(taus, [r["macro_f1"] for r in rows], label="Macro-F1", lw=2)
    axes[0].axvline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set(xlabel="tau (rare-class boost)", title="Overall metrics vs. operating point (DS2)")
    for c in "NSVF":
        axes[1].plot(taus, [r["recall"][c] for r in rows], label=f"{c} recall", lw=2)
    axes[1].axvline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="tau (rare-class boost)", title="Per-class recall (dashed = shipped tau=1.0)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "operating_curve.png", dpi=150)
    for r in rows[::5]:
        print(r["tau"], round(r["accuracy"], 3), round(r["macro_f1"], 3), {k: round(v, 2) for k, v in r["recall"].items()})


if __name__ == "__main__":
    main()
