"""Train a calibrated 3-model ensemble with an inter-patient validation split.

Differences from train.py (all aimed at generalising to unseen patients):
  * validation = whole held-out *patients* (not random beats of seen patients)
  * sqrt-inverse-frequency sampling (full balancing made rare classes over-predicted)
  * degree features on every node (structural information the GCN otherwise lacks)
  * mean+max readout, gradient clipping, cosine-free plateau LR schedule
  * ensemble of GCN + GraphSAGE + GATv2 and post-hoc temperature / class-bias
    calibration fitted on the held-out patients only (DS2 is never touched here)
"""

import argparse
import json
import logging
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from config import AAMI_CLASSES, CHECKPOINT_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from dataset import ECGVisibilityGraphDataset
from ensemble import ENSEMBLE_PATH, Classifier, save_ensemble
from model import ECGGCN
from visibility_graph import add_degree_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Held-out patients: contain S (96+107+73+31), V (16+210+473+47+71) and F (14+5+11) beats.
VAL_RECORDS = [118, 207, 223, 124, 205]
MEMBERS = [
    {"arch": "gcn", "pooling": "meanmax", "in_channels": 4, "hidden_channels": 64},
    {"arch": "sage", "pooling": "meanmax", "in_channels": 4, "hidden_channels": 64},
    {"arch": "gat", "pooling": "meanmax", "in_channels": 4, "hidden_channels": 64},
]
PRESENT = [0, 1, 2, 3]  # classes with validation support (Q has none)


def sqrt_sampler(labels, num_samples):
    counts = np.clip(np.bincount(labels, minlength=len(AAMI_CLASSES)), 1, None).astype(float)
    w = counts ** -0.5  # per-sample weight => class sampling prob proportional to sqrt(count)
    return WeightedRandomSampler([w[l] for l in labels], num_samples=num_samples, replacement=True)


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    out, ys = [], []
    for b in loader:
        out.append(model(b.x, b.edge_index, b.batch))
        ys.append(b.y)
    return torch.cat(out), torch.cat(ys)


def macro_f1(y, pred):
    return f1_score(y, pred, labels=PRESENT, average="macro", zero_division=0)


def train_member(cfg, train_set, train_labels, val_loader, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ECGGCN(num_classes=len(AAMI_CLASSES), **cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=1, factor=0.5)
    crit = torch.nn.CrossEntropyLoss()
    best, best_state, bad, hist = float("inf"), None, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loader = DataLoader(train_set, batch_size=args.batch_size,
                            sampler=sqrt_sampler(train_labels, args.samples_per_epoch))
        model.train()
        tot = 0.0
        for b in loader:
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b.num_graphs
        logits, y = collect_logits(model, val_loader)
        val_loss = torch.nn.functional.cross_entropy(logits, y).item()
        f1 = macro_f1(y.numpy(), logits.argmax(1).numpy())
        sched.step(val_loss)
        hist.append({"epoch": epoch, "train_loss": tot / args.samples_per_epoch, "val_loss": val_loss, "val_macro_f1": f1})
        logger.info("[%s] epoch %02d train_loss=%.4f val_loss=%.4f val_macroF1(NSVF)=%.4f (%.0fs)",
                    cfg["arch"], epoch, hist[-1]["train_loss"], val_loss, f1, time.time() - t0)
        if val_loss < best:  # select on val NLL: far less noisy than macro-F1 with ~30 F beats
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                logger.info("[%s] early stop at epoch %d (best val loss %.4f)", cfg["arch"], epoch, best)
                break
    model.load_state_dict(best_state)
    return model, best, hist


def fit_temperature(logits, y):
    """Minimise NLL over a single scalar temperature (log-parametrised)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def fit_biases(logits, y, temperature, passes=2, grid=np.arange(-1.5, 1.51, 0.25)):
    """Coordinate search of per-class logit biases maximising val macro-F1.
    Class 0 (N) is the reference (bias fixed at 0); Q has no val support, so it is left at 0."""
    z = (logits / temperature).numpy()
    yy = y.numpy()
    bias = np.zeros(len(AAMI_CLASSES))
    best = macro_f1(yy, z.argmax(1))
    for _ in range(passes):
        for c in (1, 2, 3):
            for g in grid:
                trial = bias.copy()
                trial[c] = g
                f = macro_f1(yy, (z + trial).argmax(1))
                if f > best + 1e-6:
                    best, bias = f, trial
    return bias, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--members", type=int, default=len(MEMBERS))
    args = ap.parse_args()

    ds = ECGVisibilityGraphDataset(root=str(PROCESSED_DATA_DIR), split="ds1", transform=add_degree_features)
    rec = ds._data.record_id.numpy()
    labels_all = ds._data.y.numpy()
    is_val = np.isin(rec, VAL_RECORDS)
    train_idx, val_idx = np.where(~is_val)[0].tolist(), np.where(is_val)[0].tolist()
    logger.info("train=%d val=%d | train classes=%s val classes=%s", len(train_idx), len(val_idx),
                np.bincount(labels_all[~is_val], minlength=5).tolist(),
                np.bincount(labels_all[is_val], minlength=5).tolist())
    train_set, val_set = ds[train_idx], ds[val_idx]
    train_labels = labels_all[~is_val].tolist()
    val_loader = DataLoader(val_set, batch_size=256)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfgs, states, histories, models = [], [], {}, []
    for k, cfg in enumerate(MEMBERS[: args.members]):
        model, best, hist = train_member(cfg, train_set, train_labels, val_loader, args, seed=42 + k)
        models.append(model)
        cfgs.append(cfg)
        states.append({k_: v.clone() for k_, v in model.state_dict().items()})
        histories[cfg["arch"]] = hist
        # checkpoint after every member so a long run is never lost
        save_ensemble(ENSEMBLE_PATH, cfgs, states, 1.0, [0] * 5, True, {"partial": True})

    ens = Classifier(models, degree_features=True).eval()
    members_logits = []
    for m in models:
        lg, y = collect_logits(m, val_loader)
        members_logits.append(lg)
        logger.info("member %s val acc=%.4f macroF1=%.4f", m.conv1.__class__.__name__,
                    (lg.argmax(1) == y).float().mean(), macro_f1(y.numpy(), lg.argmax(1).numpy()))
    logits = torch.stack(members_logits).mean(0)
    raw_f1 = macro_f1(y.numpy(), logits.argmax(1).numpy())
    T = fit_temperature(logits, y)
    bias, cal_f1 = fit_biases(logits, y, T)
    nll_raw = torch.nn.functional.cross_entropy(logits, y).item()
    nll_cal = torch.nn.functional.cross_entropy(logits / T + torch.tensor(bias, dtype=torch.float), y).item()
    logger.info("ensemble val: macroF1 raw=%.4f calibrated=%.4f | NLL raw=%.4f calibrated=%.4f | T=%.3f bias=%s",
                raw_f1, cal_f1, nll_raw, nll_cal, T, np.round(bias, 2).tolist())

    save_ensemble(ENSEMBLE_PATH, cfgs, states, T, bias, True,
                  {"val_records": VAL_RECORDS, "val_macro_f1_raw": raw_f1, "val_macro_f1_cal": cal_f1,
                   "val_nll_raw": nll_raw, "val_nll_cal": nll_cal})
    with open(REPORTS_DIR / "ensemble_training.json", "w") as f:
        json.dump({"history": histories, "temperature": T, "bias": bias.tolist(),
                   "val_macro_f1_raw": raw_f1, "val_macro_f1_cal": cal_f1,
                   "val_nll_raw": nll_raw, "val_nll_cal": nll_cal}, f, indent=2)
    logger.info("Saved ensemble to %s", ENSEMBLE_PATH)


if __name__ == "__main__":
    main()
