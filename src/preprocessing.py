"""Stage 2 & 3: Signal filtering and beat segmentation.

- Butterworth bandpass filter (zero-phase via filtfilt) to remove baseline
  wander and powerline noise.
- Fixed-window segmentation around cardiologist-annotated R-peaks.
- AAMI 5-class label mapping; beats with symbols outside the AAMI table or
  that don't represent a beat annotation (e.g. rhythm change markers '+',
  or non-beat symbols like '~', '|') are dropped.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

from config import (
    AAMI_MAP,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FILTER_ORDER,
    FS,
    POST_SAMPLES,
    PRE_SAMPLES,
)
from ingestion import RecordData

logger = logging.getLogger(__name__)


@dataclass
class Beat:
    record_id: int
    r_peak_sample: int
    waveform: np.ndarray  # length PRE_SAMPLES + POST_SAMPLES
    label: str            # AAMI class: N/S/V/F/Q
    raw_symbol: str = field(default="")


def bandpass_filter(signal: np.ndarray, fs: int = FS,
                     low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
                     order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def segment_beats(record: RecordData, pre: int = PRE_SAMPLES,
                   post: int = POST_SAMPLES) -> list:
    """Filter the record's signal and slice out fixed windows around each
    annotated R-peak, dropping beats near the record boundary and any
    annotation symbol that isn't in the AAMI mapping (e.g. rhythm labels)."""
    filtered = bandpass_filter(record.signal, fs=record.fs)
    n = len(filtered)

    beats = []
    for sample, symbol in zip(record.ann_samples, record.ann_symbols):
        if symbol not in AAMI_MAP:
            continue
        start, end = sample - pre, sample + post
        if start < 0 or end > n:
            continue
        beats.append(
            Beat(
                record_id=record.record_id,
                r_peak_sample=int(sample),
                waveform=filtered[start:end].copy(),
                label=AAMI_MAP[symbol],
                raw_symbol=symbol,
            )
        )
    logger.info(
        "Record %s: %d beats segmented (%d annotations total)",
        record.record_id, len(beats), len(record.ann_samples),
    )
    return beats


if __name__ == "__main__":
    from ingestion import load_record

    rec = load_record(100)
    beats = segment_beats(rec)
    logger.info("First beat label=%s waveform shape=%s", beats[0].label, beats[0].waveform.shape)
