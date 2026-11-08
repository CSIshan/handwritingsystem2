"""
Step 5 — Word Segmentation
Splits each line pair's content into individual word image crops.
Uses vertical projection profile to find inter-word valleys.
"""
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
from line_detector import LinePair

log = logging.getLogger(__name__)


@dataclass
class WordCrop:
    pair_index:   int
    word_index:   int          # position in line, left to right
    image:        np.ndarray   # binary crop of just this word
    x_start:      int          # pixel X of left edge in full page
    x_end:        int          # pixel X of right edge in full page
    baseline_y:   int          # pixel Y of baseline in full page
    top_y:        int          # pixel Y of top of writing strip
    flagged:      bool = False
    flag_reason:  Optional[str] = None


def segment_words(
    binary: np.ndarray,
    pairs:  List[LinePair],
    config,
) -> List[WordCrop]:
    """
    For each line pair, extract individual word crops.
    Returns flat list of WordCrop across all pairs.
    """
    all_words = []
    h, w = binary.shape

    for pair in pairs:
        # Extract the strip for this pair
        strip = binary[pair.strip_top:pair.strip_bottom, :]

        if strip.shape[0] == 0:
            log.warning(f"Pair {pair.index}: empty strip, skipping")
            continue

        words = _segment_strip(strip, pair, config, w)
        all_words.extend(words)
        log.debug(f"Pair {pair.index}: {len(words)} words extracted")

    log.info(f"Segmentation: {len(all_words)} total word crops across {len(pairs)} pairs")
    return all_words


def _segment_strip(
    strip:  np.ndarray,
    pair:   LinePair,
    config,
    page_w: int,
) -> List[WordCrop]:
    """Segment one strip into word crops using vertical projection."""

    # Vertical projection: sum ink pixels down each column
    v_proj = np.sum(strip > 0, axis=0).astype(float)

    min_gap_px    = config.mm_to_px(config.min_gap_width_mm)
    min_word_px   = config.mm_to_px(config.min_word_width_mm)
    max_word_px   = config.mm_to_px(config.max_word_width_mm)
    min_height_px = config.mm_to_px(config.min_word_height_mm)
    max_height_px = config.mm_to_px(config.max_word_height_mm)
    margin_px     = config.mm_to_px(config.margin_left_mm)

    # Find valleys: columns with zero (or near-zero) ink = gaps between words
    threshold = max(1.0, np.max(v_proj) * 0.02)  # 2% of max density
    is_gap = v_proj <= threshold

    # Find contiguous ink regions (words)
    regions = _find_regions(is_gap, min_gap_px)

    words = []
    for word_idx, (x_start, x_end) in enumerate(regions):
        width_px = x_end - x_start

        # Get word image crop
        crop = strip[:, x_start:x_end]

        # Find actual ink extent vertically (tight bbox)
        rows_with_ink = np.any(crop > 0, axis=1)
        if not np.any(rows_with_ink):
            continue

        ink_rows = np.where(rows_with_ink)[0]
        ink_height_px = int(ink_rows[-1] - ink_rows[0])

        # ── Flagging ──────────────────────────────────────────────────────────
        flagged = False
        flag_reason = None

        if width_px < min_word_px:
            flagged, flag_reason = True, f"too narrow ({config.px_to_mm(width_px):.1f}mm)"
        elif width_px > max_word_px:
            flagged, flag_reason = True, f"too wide ({config.px_to_mm(width_px):.1f}mm) — possible merged words"
        elif ink_height_px < min_height_px:
            flagged, flag_reason = True, f"too short ({config.px_to_mm(ink_height_px):.1f}mm)"
        elif ink_height_px > max_height_px:
            flagged, flag_reason = True, f"too tall ({config.px_to_mm(ink_height_px):.1f}mm) — possible crossed lines"

        wc = WordCrop(
            pair_index  = pair.index,
            word_index  = word_idx,
            image       = crop,
            x_start     = x_start + margin_px,   # back to page coordinates
            x_end       = x_end   + margin_px,
            baseline_y  = pair.baseline_y,
            top_y       = pair.top_y,
            flagged     = flagged,
            flag_reason = flag_reason,
        )
        words.append(wc)

        if flagged:
            log.debug(f"Pair {pair.index} word {word_idx}: flagged — {flag_reason}")

    return words


def _find_regions(is_gap: np.ndarray, min_gap: int) -> List[tuple]:
    """
    Find contiguous non-gap regions in is_gap array.
    Gaps shorter than min_gap are bridged (treated as part of word).
    Returns list of (start, end) column indices.
    """
    n = len(is_gap)

    # Bridge short gaps: if gap is narrower than min_gap, treat as ink
    bridged = is_gap.copy()
    i = 0
    while i < n:
        if bridged[i]:
            # Find end of this gap
            j = i
            while j < n and bridged[j]:
                j += 1
            gap_len = j - i
            if gap_len < min_gap:
                bridged[i:j] = False   # bridge it
            i = j
        else:
            i += 1

    # Find ink regions
    regions = []
    in_word = False
    start = 0
    for i in range(n):
        if not bridged[i] and not in_word:
            start = i
            in_word = True
        elif bridged[i] and in_word:
            regions.append((start, i))
            in_word = False
    if in_word:
        regions.append((start, n))

    return regions
