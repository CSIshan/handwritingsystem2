"""
Estimate actual scan DPI from known ruled-line spacing, since embedded
image metadata (e.g. 72 DPI) is often a meaningless default, not the
real scan resolution.

REWRITTEN for the single-ruled-line template (see config.py docstring --
the old two-line "pair" model never matched the real notebook). This is
actually simpler and more robust than the old method: every one of the
~9 gaps between consecutive detected lines gives an independent DPI
estimate directly against one known constant (line_spacing_mm), instead
of the old scheme's two DIFFERENT alternating gap types disagreeing with
each other (confirmed on real data: that old method's own self-reported
disagreement was >330 DPI on a real page -- a symptom of measuring the
wrong geometry, not scan noise).

Raises if measured DPI < min_dpi (too low to trust downstream pixel math).
Otherwise returns the measured DPI so the caller can set config.dpi.
"""
import logging
import numpy as np
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


class DPITooLowError(Exception):
    pass


def estimate_dpi(grey: np.ndarray, config, min_dpi: float = 300.0) -> float:
    """
    Measures pixel spacing between consecutive ruled lines and back-solves
    DPI using config.line_spacing_mm.

    grey: RAW grayscale scan (NOT binarize()'s output -- binarize() removes
          template lines entirely, so there'd be nothing left to measure).
    """
    combined_mask = grey < config.line_threshold
    coverage = combined_mask.mean(axis=1)

    # A genuine ruled line runs nearly the full page width; distance=50 is
    # a conservative floor that stays safely below any real line spacing
    # across a wide range of plausible scan DPIs (even at a low 300 DPI,
    # 7.85mm spacing is still ~93px -- comfortably above 50).
    peaks, props = find_peaks(coverage, height=0.05, distance=50)

    if len(peaks) < 3:
        raise DPITooLowError(
            f"Could not detect enough ruled lines to measure DPI "
            f"(found {len(peaks)} candidate lines, need at least 3)."
        )

    gaps_px = np.diff(peaks)
    gap_dpis = (gaps_px / config.line_spacing_mm) * 25.4

    # Median is robust to one bad gap (e.g. a page edge artifact catching
    # an extra false peak); mean would let a single outlier skew everything.
    measured_dpi = float(np.median(gap_dpis))
    spread = float(np.std(gap_dpis))

    log.info(
        f"DPI estimate: {len(gaps_px)} line-gaps, median={measured_dpi:.1f}, "
        f"spread(std)={spread:.1f}"
    )

    if spread > 20:
        log.warning(
            f"DPI estimates across individual line-gaps disagree by "
            f"std={spread:.1f} -- line detection may be unreliable, "
            f"treat this measurement with caution."
        )

    if measured_dpi < min_dpi:
        raise DPITooLowError(
            f"Measured DPI ({measured_dpi:.1f}) is below minimum ({min_dpi}). "
            f"Scan quality too low for reliable pixel-based processing."
        )

    return measured_dpi
