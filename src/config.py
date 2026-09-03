"""Shared constants for the ECG visibility-graph GCN pipeline."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "outputs" / "checkpoints"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# --- Signal / dataset constants ---
FS = 360  # MIT-BIH sampling rate (Hz)
PRE_SAMPLES = 90   # samples before R-peak
POST_SAMPLES = 110  # samples after R-peak
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 40.0  # Hz
FILTER_ORDER = 4
SIGNAL_CHANNEL = "MLII"

# --- AAMI 5-class mapping ---
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "Q": "Q", "/": "Q", "f": "Q",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# --- Inter-patient split (de Chazal et al. DS1 / DS2) ---
# DS1: used for training/validation. DS2: held out for testing.
DS1_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
    223, 230,
]
DS2_RECORDS = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
    233, 234,
]
ALL_RECORDS = sorted(DS1_RECORDS + DS2_RECORDS)

# Records using a paced/non-MLII lead or otherwise excluded from the
# standard AAMI inter-patient benchmark (102, 104, 107, 217 use paced signals).
EXCLUDED_RECORDS = [102, 104, 107, 217]

PHYSIONET_DB = "mitdb"
