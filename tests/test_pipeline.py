import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import ECGGCN  # noqa: E402
from preprocessing import Beat  # noqa: E402
from train import class_counts, make_oversampler, safe_stratified_split  # noqa: E402
from visibility_graph import beat_to_graph  # noqa: E402
from evaluate import per_class_sens_spec  # noqa: E402


def _beat(label="N", n=200):
    return Beat(record_id=0, r_peak_sample=n // 2, waveform=np.sin(np.linspace(0, 4 * np.pi, n)), label=label)


def test_graph_shape_and_symmetry():
    d = beat_to_graph(_beat())
    assert d.x.shape == (200, 2)
    assert d.edge_index.shape[0] == 2
    edges = set(map(tuple, d.edge_index.t().tolist()))
    assert all((j, i) in edges for i, j in edges)  # undirected
    # consecutive samples are always mutually visible in a natural VG
    assert all((i, i + 1) in edges for i in range(199))


def test_model_forward_shape():
    from torch_geometric.data import Batch
    b = Batch.from_data_list([beat_to_graph(_beat()), beat_to_graph(_beat("V", 150))])
    out = ECGGCN()(b.x, b.edge_index, b.batch)
    assert out.shape == (2, 5)


def test_oversampler_balances_classes():
    labels = [0] * 900 + [1] * 90 + [2] * 10
    torch.manual_seed(0)
    drawn = np.array([labels[i] for i in make_oversampler(labels, 3)])
    frac = np.bincount(drawn, minlength=3) / len(drawn)
    assert frac.min() > 0.25


def test_class_counts_clips_absent_classes():
    assert class_counts([0, 0, 1], 3).tolist() == [2, 1, 1]


def test_safe_split_falls_back_when_class_too_small():
    labels = [0] * 20 + [1]
    tr, te = safe_stratified_split(list(range(21)), labels, test_size=0.2, random_state=0)
    assert len(tr) + len(te) == 21


def test_sens_spec():
    cm = np.array([[8, 2], [1, 9]])
    ss = per_class_sens_spec(cm)
    assert ss[0] == pytest.approx((0.8, 0.9))


@pytest.mark.parametrize("arch", ["gcn", "sage", "gat"])
@pytest.mark.parametrize("pooling", ["mean", "meanmax"])
def test_model_variants(arch, pooling):
    from torch_geometric.data import Batch
    b = Batch.from_data_list([beat_to_graph(_beat()), beat_to_graph(_beat("V", 150))])
    assert ECGGCN(arch=arch, pooling=pooling)(b.x, b.edge_index, b.batch).shape == (2, 5)


def test_graph_stats():
    from visibility_graph import graph_stats
    s = graph_stats(beat_to_graph(_beat()))
    assert s["nodes"] == 200 and s["edges"] >= 199 and 0 < s["density"] <= 1
