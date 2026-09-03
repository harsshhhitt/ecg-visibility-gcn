"""Stage 1: Data ingestion — pull MIT-BIH records (signal + annotations) via wfdb.

Downloads are cached under data/raw/ on first access; subsequent calls reuse
the local copy (wfdb.rdrecord/rdann do this automatically when pb_dir is
omitted and files already exist locally).
"""

import logging
from dataclasses import dataclass

import numpy as np
import wfdb

from config import ALL_RECORDS, PHYSIONET_DB, RAW_DATA_DIR, SIGNAL_CHANNEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RecordData:
    record_id: int
    signal: np.ndarray          # 1D float array, single lead
    fs: int
    ann_samples: np.ndarray     # sample index of each annotated beat
    ann_symbols: list           # annotation symbol per beat


def download_record(record_id: int, target_dir=RAW_DATA_DIR) -> None:
    """Download a single record's .dat/.hea/.atr files if not already cached."""
    target_dir.mkdir(parents=True, exist_ok=True)
    record_path = target_dir / f"{record_id}.hea"
    if record_path.exists():
        logger.debug("Record %s already cached at %s", record_id, target_dir)
        return
    logger.info("Downloading record %s from PhysioNet (%s)...", record_id, PHYSIONET_DB)
    files = [f"{record_id}.hea", f"{record_id}.dat", f"{record_id}.atr"]
    wfdb.dl_files(PHYSIONET_DB, str(target_dir), files, keep_subdirs=False)


def load_record(record_id: int, data_dir=RAW_DATA_DIR) -> RecordData:
    """Load a single record's signal + beat annotations, downloading if needed."""
    download_record(record_id, data_dir)
    record_str = str(data_dir / str(record_id))

    record = wfdb.rdrecord(record_str)
    ann = wfdb.rdann(record_str, "atr")

    try:
        channel_idx = record.sig_name.index(SIGNAL_CHANNEL)
    except ValueError:
        raise ValueError(
            f"Record {record_id} does not contain a {SIGNAL_CHANNEL} channel "
            f"(available: {record.sig_name})"
        )

    signal = record.p_signal[:, channel_idx].astype(np.float64)

    return RecordData(
        record_id=record_id,
        signal=signal,
        fs=record.fs,
        ann_samples=np.asarray(ann.sample),
        ann_symbols=list(ann.symbol),
    )


def load_records(record_ids=ALL_RECORDS, data_dir=RAW_DATA_DIR):
    """Yield RecordData for each requested record id, skipping ones that fail to load."""
    for rid in record_ids:
        try:
            yield load_record(rid, data_dir)
        except Exception as exc:  # noqa: BLE001 - log and continue across a batch job
            logger.error("Failed to load record %s: %s", rid, exc)


if __name__ == "__main__":
    rec = load_record(100)
    logger.info(
        "Loaded record 100: %d samples @ %dHz, %d annotations",
        len(rec.signal), rec.fs, len(rec.ann_samples),
    )
