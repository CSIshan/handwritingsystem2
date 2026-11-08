"""
Estimate actual scan DPI from known template line spacing, since
embedded image metadata (e.g. 72 DPI) is often a meaningless default,
not the real scan resolution.

Raises if measured DPI < 300 (too low to trust downstream pixel math).
Otherwise returns the measured DPI so the caller can set config.dpi.
"""
import logging
import cv2
import numpy as np
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


class DPITooLowError(Exception):
    pass


def estimate_dpi(grey: np.ndarray, config, min_dpi: float = 300.0) -> float:
    """
    Measures pixel spacing between the first few template lines and
    back-solves DPI using config.pair_gap_mm / config.inter_pair_mm.

    grey: RAW grayscale scan (NOT binarize()'s output -- binarize() removes
          template lines entirely, so there'd be nothing left to measure).
          We threshold at config.line_threshold ourselves, which captures
          ink + template lines together (same value binarize.py uses for
          its own "combined_mask").
    """
    _, combined_mask = cv2.threshold(
        grey, config.line_threshold, 255, cv2.THRESH_BINARY_INV
    )
    density = np.sum(combined_mask > 0, axis=1).astype(float)
    kernel = np.ones(9) / 9
    density_smooth = np.convolve(density, kernel, mode='same')

    # Lines = local minima in ink density. Invert so find_peaks finds valleys.
    inverted = -density_smooth
    peaks, _ = find_peaks(inverted, distance=10, prominence=np.std(density_smooth) * 0.5)

    if len(peaks) < 3:
        raise DPITooLowError(
            f"Could not detect enough template lines to measure DPI "
            f"(found {len(peaks)} candidate lines, need at least 3)."
        )

    # First three detected lines should be: top_0, baseline_0, top_1
    # gap0 = pair_gap_mm (top_0 -> baseline_0)
    # gap1 = inter_pair_mm (baseline_0 -> top_1)
    gap0_px = peaks[1] - peaks[0]
    gap1_px = peaks[2] - peaks[1]

    dpi_from_gap0 = (gap0_px / config.pair_gap_mm) * 25.4
    dpi_from_gap1 = (gap1_px / config.inter_pair_mm) * 25.4

    measured_dpi = float(np.mean([dpi_from_gap0, dpi_from_gap1]))

    log.info(
        f"DPI estimate: gap0={dpi_from_gap0:.1f}, gap1={dpi_from_gap1:.1f}, "
        f"averaged={measured_dpi:.1f}"
    )

    if abs(dpi_from_gap0 - dpi_from_gap1) > 20:
        log.warning(
            f"DPI estimates disagree by >{abs(dpi_from_gap0 - dpi_from_gap1):.1f} "
            f"-- line detection may be unreliable, treat this measurement with caution."
        )

    if measured_dpi < min_dpi:
        raise DPITooLowError(
            f"Measured DPI ({measured_dpi:.1f}) is below minimum ({min_dpi}). "
            f"Scan quality too low for reliable pixel-based processing."
        )

    return measured_dpi
