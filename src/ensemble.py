"""Calibrated ensemble of graph networks.

An ensemble averages the logits of several differently-structured GNNs
(GCN / GraphSAGE / GATv2), then applies a single temperature T and per-class
logit biases fitted on *held-out patients* (never on the DS2 test set):

    p = softmax(mean_k(logits_k) / T + bias)

Averaging reduces variance (better accuracy, and a disagreement-based
uncertainty signal); temperature scaling makes the confidence honest.
`load_classifier` also wraps a legacy single-model checkpoint so every
consumer (predict / explain / evaluate / app) uses one interface.
"""

import torch

from config import AAMI_CLASSES, CHECKPOINT_DIR
from model import ECGGCN

ENSEMBLE_PATH = CHECKPOINT_DIR / "ensemble.pt"


class Classifier(torch.nn.Module):
    """Uniform interface: forward(x, edge_index, batch) -> calibrated logits."""

    def __init__(self, models, temperature=1.0, bias=None, degree_features=False):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("bias", torch.zeros(len(AAMI_CLASSES)) if bias is None else torch.tensor(bias, dtype=torch.float))
        self.degree_features = degree_features

    def member_logits(self, x, edge_index, batch):
        return torch.stack([m(x, edge_index, batch) for m in self.models])  # [K, B, C]

    def raw_logits(self, x, edge_index, batch):
        return self.member_logits(x, edge_index, batch).mean(0)

    def forward(self, x, edge_index, batch):
        return self.raw_logits(x, edge_index, batch) / self.temperature + self.bias

    def probs_with_spread(self, x, edge_index, batch):
        """Calibrated ensemble probs plus the across-member std (disagreement)."""
        member = self.member_logits(x, edge_index, batch)
        p = torch.softmax(member.mean(0) / self.temperature + self.bias, dim=-1)
        member_p = torch.softmax(member / self.temperature + self.bias, dim=-1)
        return p, member_p.std(0)


def save_ensemble(path, configs, state_dicts, temperature, bias, degree_features, meta=None):
    torch.save({"configs": configs, "state_dicts": state_dicts, "temperature": float(temperature),
                "bias": list(map(float, bias)), "degree_features": degree_features, "meta": meta or {}}, path)


def load_classifier(path=None, device="cpu"):
    """Load the ensemble if present; otherwise wrap the legacy single GCN checkpoint."""
    path = path or ENSEMBLE_PATH
    if path.exists():
        ck = torch.load(path, map_location=device)
        models = []
        for cfg, sd in zip(ck["configs"], ck["state_dicts"]):
            m = ECGGCN(num_classes=len(AAMI_CLASSES), **cfg)
            m.load_state_dict(sd)
            models.append(m)
        clf = Classifier(models, ck["temperature"], ck["bias"], ck["degree_features"])
        clf.train_counts = ck.get("meta", {}).get("train_counts")
        clf.default_tau = ck.get("meta", {}).get("tau")
    else:
        legacy = CHECKPOINT_DIR / "best_model.pt"
        m = ECGGCN(in_channels=2, num_classes=len(AAMI_CLASSES))
        m.load_state_dict(torch.load(legacy, map_location=device))
        clf = Classifier([m])
    return clf.to(device).eval()


def prior_adjust_bias(train_counts, tau, sampling_power=0.5):
    """Logit-adjustment bias that undoes the class prior the model was trained under.

    With sqrt sampling the effective training prior is proportional to count**0.5, so
    bias_c = tau * (log p_N - log p_c) (N is the reference, bias 0). tau=0 keeps the
    trained prior, tau=1 targets a class-balanced decision rule. The Q class (a handful of
    training beats) is left at 0 rather than being boosted on noise."""
    import numpy as np
    p = np.clip(np.asarray(train_counts, dtype=float), 1, None) ** sampling_power
    p = p / p.sum()
    bias = tau * (np.log(p[0]) - np.log(p))
    bias[len(bias) - 1] = 0.0
    return bias
