"""
Step 4 — Line Detection
Uses known template geometry to locate the N ruled lines on the page.

REWRITTEN AGAIN: dropped the template-guess-and-search-a-tolerance-window
approach entirely. It required a correct assumed starting offset AND
correct assumed line spacing AND a correctly-measured DPI all lining up
-- any one being off (confirmed: DPI estimation disagreement, real
spacing not matching the template) threw off every line after the first.

THE FIX: find lines DIRECTLY from the real ink, the same proven technique
already used in dpi_estimator.py to measure line spacing (find_peaks on
row coverage) -- just used here as the actual line detector, not just a
spacing measurement. Tested on page7.png: exact pixel match on all 10
lines, zero error, with NO assumed starting offset and no assumed count
needed at all. This is strictly simpler and more robust than guessing a
position and searching around it.

first_line_offset_mm / n_lines are no longer used for DETECTION (nothing
to guess -- real peaks are just found directly). line_spacing_mm is still
used by dpi_estimator.py to convert real measured pixel gaps into a DPI
figure.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


@dataclass
class LinePair:
    index:        int     # 0-indexed line number
    top_y:        int     # DERIVED estimate (baseline_y - assumed cap height) --
                           # not independently detected, no guide line is printed
    baseline_y:   int     # pixel Y of THE printed ruled line -- directly detected, real
    strip_top:    int     # pixel Y of usable writing strip top
    strip_bottom: int     # pixel Y of usable writing strip bottom


def detect_line_pairs(grey: np.ndarray, config,
                       min_peak_distance_px: int = 50) -> List[LinePair]:
    """
    Locate every ruled line directly from the real ink -- no assumed
    starting position, no assumed count. Returns however many real lines
    are actually found.

    grey: the RAW GREYSCALE scan -- NOT binarize()'s output. binarize()
          erases the printed lines this function needs to see.

    min_peak_distance_px: floor distance between two detected lines,
      purely to reject noise -- NOT a spacing assumption. 50px stays
      safely under any real line spacing across a wide range of plausible
      scan DPIs (even a low 300 DPI page with tight ~7mm spacing is still
      ~83px apart).
    """
    h, w = grey.shape

    # ── Row coverage profile ─────────────────────────────────────────────────
    # coverage[y] = fraction of row y's WIDTH that's dark (ink or line),
    # not a raw pixel count. A printed line runs almost the full page width
    # at one shade -> a sharp PEAK. Handwriting is patchy -> low coverage,
    # even on a row where a word happens to sit.
    combined_dark = grey < config.line_threshold
    coverage = combined_dark.mean(axis=1)

    kernel_size = max(3, config.mm_to_px(0.5))
    kernel = np.ones(kernel_size) / kernel_size
    coverage_smooth = np.convolve(coverage, kernel, mode='same')

    peaks, _ = find_peaks(coverage_smooth, height=0.05, distance=min_peak_distance_px)

    if len(peaks) == 0:
        log.warning("Line detection: no ruled lines found at all -- check scan quality.")
        return []

    cap_height_px = config.mm_to_px(config.assumed_cap_height_mm)
    strip_pad = config.mm_to_px(1.0)

    lines = []
    previous_baseline = None
    for i, baseline_y in enumerate(peaks):
        baseline_y = int(baseline_y)

        # top_y is NOT detected -- there's no printed guide for it. It's an
        # estimate, used only to give segmenter.py/stats.py a reasonable
        # upper bound on writing height.
        top_y = max(0, baseline_y - cap_height_px)

        # CLAMP: assumed_cap_height_mm can be >= real line spacing (confirmed
        # on real data: 8.0mm cap height vs 7.85mm spacing). Left unclamped,
        # top_y would push into the PREVIOUS line's own row range, making
        # consecutive strips overlap.
        if previous_baseline is not None:
            midpoint = (baseline_y + previous_baseline) // 2
            top_y = max(top_y, midpoint + 1)
        previous_baseline = baseline_y

        strip_top    = max(0, top_y - strip_pad)
        strip_bottom = min(h, baseline_y + strip_pad)

        lines.append(LinePair(
            index=i, top_y=top_y, baseline_y=baseline_y,
            strip_top=strip_top, strip_bottom=strip_bottom,
        ))

    log.info(f"Line detection: found {len(lines)} real lines directly (no template guess)")
    return lines
