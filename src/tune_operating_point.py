"""Pick the decision-threshold strength `tau` on held-out *validation patients* (never DS2).

Logit adjustment shifts class biases to undo the training class prior (see
ensemble.prior_adjust_bias). tau is chosen to maximise balanced accuracy (mean
per-class recall over the classes present in validation), which, unlike overall
accuracy, does not reward simply predicting the majority class. The chosen bias is
written back into ensemble.pt; the temperature is left as fitted.
"""

import json
import logging

import numpy as np
import torch
from sklearn.metrics import recall_score
from torch_geometric.loader import DataLoader

from config import PROCESSED_DATA_DIR, REPORTS_DIR
from dataset import ECGVisibilityGraphDataset
from ensemble import ENSEMBLE_PATH, load_classifier, prior_adjust_bias
from train_ensemble import PRESENT, VAL_RECORDS
from visibility_graph import add_degree_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    clf = load_classifier()
    ds = ECGVisibilityGraphDataset(root=str(PROCESSED_DATA_DIR), split="ds1", transform=add_degree_features)
    rec, y_all = ds._data.record_id.numpy(), ds._data.y.numpy()
    is_val = np.isin(rec, VAL_RECORDS)
    train_counts = np.bincount(y_all[~is_val], minlength=5)
    val_set = ds[np.where(is_val)[0].tolist()]

    raws, ys = [], []
    with torch.no_grad():
        for b in DataLoader(val_set, batch_size=256):
            raws.append(clf.raw_logits(b.x, b.edge_index, b.batch))
            ys.append(b.y)
    raw, y = torch.cat(raws).numpy(), torch.cat(ys).numpy()
    T = float(clf.temperature)

    rows = []
    for tau in np.round(np.arange(0.0, 1.01, 0.1), 2):
        pred = (raw / T + prior_adjust_bias(train_counts, tau)).argmax(1)
        rec_c = recall_score(y, pred, labels=PRESENT, average=None, zero_division=0)
        rows.append({"tau": float(tau), "val_accuracy": float((pred == y).mean()),
                     "val_balanced_accuracy": float(rec_c.mean()), "val_recall": rec_c.round(3).tolist()})
        logger.info("tau=%.1f acc=%.3f balanced_acc=%.3f recall(NSVF)=%s", tau, rows[-1]["val_accuracy"],
                    rows[-1]["val_balanced_accuracy"], rows[-1]["val_recall"])

    best = max(rows, key=lambda r: r["val_balanced_accuracy"])
    bias = prior_adjust_bias(train_counts, best["tau"])
    logger.info("chosen tau=%.1f bias=%s", best["tau"], np.round(bias, 2).tolist())

    ck = torch.load(ENSEMBLE_PATH, map_location="cpu")
    ck["bias"] = bias.tolist()
    ck["meta"] = {**ck.get("meta", {}), "tau": best["tau"], "train_counts": train_counts.tolist()}
    torch.save(ck, ENSEMBLE_PATH)
    with open(REPORTS_DIR / "operating_point_val.json", "w") as f:
        json.dump({"chosen": best, "sweep": rows, "train_counts": train_counts.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
