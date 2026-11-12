"""Stage 4: Training loop — inter-patient split, class-balanced oversampling,
checkpointing, early stopping on validation macro-F1.
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from config import AAMI_CLASSES, CHECKPOINT_DIR, PROCESSED_DATA_DIR
from dataset import ECGVisibilityGraphDataset
from model import ECGGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def safe_stratified_split(indices, labels, **split_kwargs):
    """train_test_split with stratify, falling back to a plain random split
    if any class has fewer members than needed to stratify (e.g. AAMI class
    Q has only a handful of examples in DS1)."""
    counts = np.bincount(labels)
    if counts[counts > 0].min() >= 2:
        return train_test_split(indices, stratify=labels, **split_kwargs)
    logger.warning(
        "Falling back to non-stratified split — a class has <2 members (counts=%s)",
        counts.tolist(),
    )
    return train_test_split(indices, **split_kwargs)


def class_counts(labels, num_classes=len(AAMI_CLASSES)):
    counts = np.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    return np.clip(counts, 1, None)  # avoid div-by-zero for absent classes


def make_oversampler(labels, num_classes=len(AAMI_CLASSES)):
    """WeightedRandomSampler that oversamples minority classes so each
    training batch is drawn roughly class-balanced, instead of relying on
    loss weighting alone (which leaves rare classes under-represented in
    every gradient step)."""
    counts = class_counts(labels, num_classes)
    class_weight = counts.sum() / (num_classes * counts)
    sample_weights = [class_weight[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            all_preds.append(out.argmax(dim=1).cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(
        all_labels, all_preds, labels=list(range(len(AAMI_CLASSES))),
        average="macro", zero_division=0,
    )
    return total_loss / len(loader.dataset), macro_f1


def train(
    epochs=100,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    val_fraction=0.1,
    device=None,
    checkpoint_dir=CHECKPOINT_DIR,
    max_samples=None,
    resume_from=None,
):
    """max_samples: if set, stratified-subsample DS1 down to this many graphs
    before splitting — for a fast demo run rather than a full training pass.
    resume_from: path to a checkpoint's state_dict to initialize weights from
    (e.g. continuing after an interrupted run) — optimizer/scheduler state is
    not restored, only model weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_full = ECGVisibilityGraphDataset(root=str(PROCESSED_DATA_DIR), split="ds1")
    labels = [int(d.y.item()) for d in train_full]

    if max_samples is not None and max_samples < len(train_full):
        keep_idx, _ = safe_stratified_split(
            range(len(train_full)), labels, train_size=max_samples, random_state=42
        )
        train_full = train_full[keep_idx]
        labels = [labels[i] for i in keep_idx]
    train_idx, val_idx = safe_stratified_split(
        range(len(train_full)), labels, test_size=val_fraction, random_state=42
    )
    train_set = train_full[train_idx]
    val_set = train_full[val_idx]
    train_labels = [labels[i] for i in train_idx]

    sampler = make_oversampler(train_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = ECGGCN(in_channels=2, num_classes=len(AAMI_CLASSES)).to(device)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        logger.info("Resumed weights from %s", resume_from)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1, epochs_no_improve = -1.0, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch, train_loss, train_f1, val_loss, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1, epochs_no_improve = val_f1, 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
                break

        torch.save(model.state_dict(), checkpoint_dir / "last_model.pt")

    return model, best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ECG visibility-graph GCN.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Subsample DS1 to this many graphs for a fast demo run.")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Checkpoint path to initialize weights from before training.")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
          patience=args.patience, max_samples=args.max_samples, resume_from=args.resume_from)
