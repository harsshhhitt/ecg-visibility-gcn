# ECG Visibility-Graph GCN

Classify MIT-BIH heartbeats (AAMI 5-class: N/S/V/F/Q) by converting each
beat's waveform into a **visibility graph** and classifying it with a
**Graph Convolutional Network**, rather than treating it as raw-signal
time-series classification.

See `prd.md` (parent directory) for full design rationale.

## Pipeline

```
ingestion.py  ->  preprocessing.py  ->  visibility_graph.py  ->  dataset.py  ->  train.py  ->  evaluate.py
 (wfdb pull)      (filter, segment)      (ts2vg -> PyG Data)     (cache .pt)    (GCN)        (metrics, plots)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Build the cached graph dataset (slow, run once; downloads MIT-BIH on first run)
python src/dataset.py --split all

# 2. Train (uses DS1 records, inter-patient split with a held-out val fraction)
python src/train.py --epochs 100 --batch-size 128

# 3. Evaluate on DS2 (held-out patients)
python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

Or run `notebooks/colab_training.ipynb` on a free-tier Colab/Kaggle GPU.

## Repository structure

```
data/            raw PhysioNet downloads + cached PyG graph datasets (gitignored)
src/             pipeline stages (see above)
notebooks/       Colab GPU training entry point
outputs/         checkpoints, figures, metrics reports (gitignored)
```

## Status

Stages 1–5 (ingestion through evaluation) are implemented per `prd.md`
Section 4. Milestone 2 (pulling the full real MIT-BIH dataset and building
the full cached graph dataset) requires actually running `src/dataset.py`
against PhysioNet, which needs network access and has not been executed in
this environment yet — code is written and ready to run.
