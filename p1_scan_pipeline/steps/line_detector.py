"""
Step 4 — Line Pair Detection
Uses known template geometry to locate the 10 writing pairs.
Each pair = (top_line_y, bottom_line_y) in pixels.
Bottom line = baseline for that pair.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import List

log = logging.getLogger(__name__)


@dataclass
class LinePair:
    index:        int     # 0-indexed pair number
    top_y:        int     # pixel Y of top line (cap height guide)
    baseline_y:   int     # pixel Y of bottom line (THE baseline)
    strip_top:    int     # pixel Y of usable writing strip top
    strip_bottom: int     # pixel Y of usable writing strip bottom


def detect_line_pairs(binary: np.ndarray, config) -> List[LinePair]:
    """
    Locate all 10 writing pairs using template geometry + horizontal
    projection profile to fine-tune positions within tolerance.

    Returns list of LinePair, one per pair.
    """
    h, w = binary.shape
    px = config.px_per_mm
    tol = config.mm_to_px(config.position_tol_mm)

    # ── Horizontal projection profile ────────────────────────────────────────
    # Sum ink pixels across each row → density[y] = ink pixels on row y
    density = np.sum(binary > 0, axis=1).astype(float)

    # Smooth slightly to handle minor row-to-row variation
    kernel_size = max(3, config.mm_to_px(0.5))
    kernel = np.ones(kernel_size) / kernel_size
    density_smooth = np.convolve(density, kernel, mode='same')

    # ── Expected line positions from template geometry ────────────────────────
    # Template: margin_top + top_offset → first top line
    # Then alternating pair_gap (within pair) and inter_pair+pair_gap (between pairs)
    margin_top_px  = config.mm_to_px(config.margin_top_mm)
    top_offset_px  = config.mm_to_px(config.top_offset_mm)
    pair_gap_px    = config.mm_to_px(config.pair_gap_mm)
    inter_pair_px  = config.mm_to_px(config.inter_pair_mm)

    pairs = []
    for i in range(config.n_pairs):
        # Expected positions from template
        expected_top = int(
            margin_top_px + top_offset_px
            + i * (pair_gap_px + inter_pair_px)
        )
        expected_baseline = expected_top + pair_gap_px

        # Fine-tune: find actual ink density peak near expected position
        actual_top      = _find_line_near(density_smooth, expected_top,      tol)
        actual_baseline = _find_line_near(density_smooth, expected_baseline,  tol)

        # Writing strip: between top line and baseline + small padding
        strip_pad = config.mm_to_px(1.0)
        strip_top    = max(0, (actual_top      or expected_top)      - strip_pad)
        strip_bottom = min(h, (actual_baseline or expected_baseline) + strip_pad)

        pair = LinePair(
            index       = i,
            top_y       = actual_top      or expected_top,
            baseline_y  = actual_baseline or expected_baseline,
            strip_top   = strip_top,
            strip_bottom= strip_bottom,
        )
        pairs.append(pair)

        log.debug(
            f"Pair {i}: top={pair.top_y}px baseline={pair.baseline_y}px "
            f"(expected top={expected_top} baseline={expected_baseline})"
        )

    log.info(f"Line pair detection: found {len(pairs)} pairs")
    return pairs


def _find_line_near(density: np.ndarray, expected_y: int, tol: int) -> int | None:
    """
    Find the row with minimum ink density near expected_y.
    Template lines are grey and partially removed by binarize, so they
    appear as valleys (low ink density) in the projection profile.
    Writing sits above/below lines, so lines = local minima.

    Returns refined Y or None if no clear valley found.
    """
    lo = max(0, expected_y - tol)
    hi = min(len(density) - 1, expected_y + tol)

    if lo >= hi:
        return None

    region = density[lo:hi]

    # Line position = local minimum in density (fewest ink pixels)
    min_idx = int(np.argmin(region))
    refined_y = lo + min_idx

    # Only accept if it's meaningfully lower than surroundings
    local_mean = float(np.mean(region))
    local_min  = float(region[min_idx])

    if local_mean > 0 and local_min < local_mean * 0.6:
        return refined_y

    # No clear valley — fall back to expected position
    return None
