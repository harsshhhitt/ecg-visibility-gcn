"""Digitize a photo/screenshot of a single-beat ECG waveform into a 1D signal
that can be fed through the existing visibility-graph -> GCN pipeline.

This is inherently a best-effort reconstruction: without a known pixel-to-mV
calibration reference on the image, the recovered signal's absolute amplitude
scale is approximate. We rescale the digitized trace to match the mean
per-beat amplitude std observed in the real MIT-BIH training data
(~0.36, see TARGET_STD) so the GCN sees inputs in a familiar numeric range,
but this is a heuristic, not a calibration.

Assumes the uploaded image is cropped to roughly one heartbeat (a single
P-QRS-T complex) with the trace as the darkest curve in the image (darker
than any grid lines).
"""

import logging

import numpy as np
from PIL import Image

from config import POST_SAMPLES, PRE_SAMPLES

logger = logging.getLogger(__name__)

# Mean per-beat amplitude std across real filtered MIT-BIH training beats
# (computed empirically from the cached DS1 dataset).
TARGET_STD = 0.36

DARKNESS_THRESHOLD = 130  # 0-255 grayscale; pixels darker than this are "trace"


def _extract_trace_columns(gray: np.ndarray, threshold: int = DARKNESS_THRESHOLD) -> np.ndarray:
    """For each image column, return the darkness-weighted average row of
    trace pixels, or NaN if no pixel in that column is dark enough."""
    height, width = gray.shape
    darkness = np.clip(255.0 - gray.astype(np.float64), 0, None)
    dark_mask = gray < threshold

    rows = np.arange(height).reshape(-1, 1)
    y_coords = np.full(width, np.nan)

    col_has_trace = dark_mask.any(axis=0)
    weights = darkness * dark_mask
    weight_sums = weights.sum(axis=0)
    weighted_rows = (weights * rows).sum(axis=0)

    valid = col_has_trace & (weight_sums > 0)
    y_coords[valid] = weighted_rows[valid] / weight_sums[valid]
    return y_coords


def _fill_gaps(y_coords: np.ndarray) -> np.ndarray:
    """Linearly interpolate over columns with no detected trace pixel."""
    x = np.arange(len(y_coords))
    valid = ~np.isnan(y_coords)
    if valid.sum() < 2:
        raise ValueError(
            "Could not detect a waveform trace in the image — try a higher-contrast "
            "crop with the trace clearly darker than the background/grid."
        )
    if not valid.all():
        y_coords = y_coords.copy()
        y_coords[~valid] = np.interp(x[~valid], x[valid], y_coords[valid])
    return y_coords


def digitize_waveform(image_path, target_length: int = PRE_SAMPLES + POST_SAMPLES,
                       threshold: int = DARKNESS_THRESHOLD) -> np.ndarray:
    """Load an ECG waveform image and return a 1D float array of length
    `target_length`, amplitude-normalized to roughly match training-data scale."""
    img = Image.open(image_path).convert("L")
    gray = np.array(img)

    y_coords = _extract_trace_columns(gray, threshold=threshold)
    y_coords = _fill_gaps(y_coords)

    # Flip vertically: image rows increase downward, ECG amplitude increases upward.
    signal = -y_coords

    # Resample to the fixed beat length used throughout the pipeline.
    src_x = np.linspace(0.0, 1.0, len(signal))
    dst_x = np.linspace(0.0, 1.0, target_length)
    resampled = np.interp(dst_x, src_x, signal)

    # Zero-center, then rescale to the empirical training amplitude std —
    # heuristic amplitude calibration, not a true pixel-to-mV conversion.
    resampled = resampled - resampled.mean()
    std = resampled.std()
    if std > 1e-8:
        resampled = resampled / std * TARGET_STD

    return resampled.astype(np.float32)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python image_digitize.py <path_to_ecg_image>")
        sys.exit(1)
    wf = digitize_waveform(sys.argv[1])
    logger.info("Digitized waveform: shape=%s mean=%.4f std=%.4f", wf.shape, wf.mean(), wf.std())
