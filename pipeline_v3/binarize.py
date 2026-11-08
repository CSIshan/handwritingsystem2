"""
Step 3 — Binarize + Remove Template Lines (v2)
Separates black gel ink from grey printed template lines and white paper.

v2 fix: brightness alone can't distinguish faint ink from a printed line
(their grey ranges overlap). Two extra constraints now protect real ink:

1. SPATIAL: only pixels near a known template-line row are even considered
   "candidate line" -- computed directly from config's template geometry,
   so ink anywhere else is never at risk regardless of its brightness.

2. CONNECTIVITY: a candidate mid-grey pixel is only actually classified as
   "line" if it is NOT 8-connected to unambiguous dark ink. A faint letter
   crossing the ruled line stays connected to darker ink just above/below
   it, so it survives; an isolated mid-grey run with no dark ink attached
   is a real printed line and gets removed.

Requires config.dpi to be set correctly BEFORE calling this (run DPI
estimation first).
"""
import logging
import numpy as np
import cv2

log = logging.getLogger(__name__)


def _expected_line_rows(config) -> list:
    """
    FALLBACK ONLY -- used when real detected line positions aren't
    available yet. Y-positions (pixels) of every expected printed ruled
    line, guessed from template geometry alone.

    Prefer passing real detected baselines to binarize() via
    known_baselines instead -- line_detector.py can now find them
    directly from the raw scan (proven zero-error on real data), which
    is strictly more reliable than this template guess.
    """
    first_line_offset_px = config.mm_to_px(config.first_line_offset_mm)
    line_spacing_px       = config.mm_to_px(config.line_spacing_mm)

    rows = []
    for i in range(config.n_lines):
        rows.append(first_line_offset_px + i * line_spacing_px)
    return rows


def binarize(grey: np.ndarray, config, line_band_mm: float = 2.0,
             known_baselines: list | None = None) -> np.ndarray:
    """
    Returns binary image: 255 = ink, 0 = background.
    Printed template lines are removed; ink crossing a line is protected.
    """
    h, w = grey.shape

    # ── Strong ink: unambiguously dark, definitely real ink ──────────────────
    _, ink_mask_u8 = cv2.threshold(grey, config.ink_threshold, 255, cv2.THRESH_BINARY_INV)
    ink_mask = ink_mask_u8 > 0

    # ── Mid-grey candidates: could be faint ink OR a printed line ────────────
    _, combined_u8 = cv2.threshold(grey, config.line_threshold, 255, cv2.THRESH_BINARY_INV)
    combined_mask = combined_u8 > 0
    candidate_line = combined_mask & ~ink_mask

    # ── Constraint 1: spatially restrict candidates to known line rows ──────
    band_px = config.mm_to_px(line_band_mm)
    row_band = np.zeros(h, dtype=bool)
    line_rows = known_baselines if known_baselines is not None else _expected_line_rows(config)
    for row_y in line_rows:
        lo = max(0, row_y - band_px)
        hi = min(h, row_y + band_px)
        row_band[lo:hi] = True

    candidate_line_in_band = candidate_line & row_band[:, None]

    if not np.any(row_band):
        log.warning("No expected line rows fall inside this image's row range "
                    "-- check config.dpi / first_line_offset_mm are correct for this crop.")

    # ── Constraint 2: protect candidates connected to strong ink ─────────────
    cc_input = ((candidate_line_in_band | ink_mask).astype(np.uint8)) * 255
    num_labels, labels = cv2.connectedComponents(cc_input, connectivity=8)

    ink_labels = set(np.unique(labels[ink_mask]).tolist())
    ink_labels.discard(0)  # 0 = background, never protect it

    protect_mask = np.isin(labels, list(ink_labels)) if ink_labels else np.zeros_like(labels, dtype=bool)

    # Real template line = candidate, in-band, NOT connected to strong ink
    true_line_mask = candidate_line_in_band & ~protect_mask

    # ── Dilate only the confirmed line pixels (safe now: band + connectivity
    #     filtered already, so this can no longer eat into real strokes) ──────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    true_line_u8 = true_line_mask.astype(np.uint8) * 255
    true_line_dilated = cv2.dilate(true_line_u8, kernel, iterations=1) > 0

    clean_ink = (ink_mask & ~true_line_dilated).astype(np.uint8) * 255

    # ── Denoise: remove single-pixel speckles ─────────────────────────────────
    denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean_ink = cv2.morphologyEx(clean_ink, cv2.MORPH_OPEN, denoise_kernel)

    ink_pixels = int(np.sum(clean_ink > 0))
    log.info(f"Binarize: {ink_pixels:,} ink pixels after line removal "
             f"({len(ink_labels)} ink-connected components protected)")

    if ink_pixels < 1000:
        log.warning("Very few ink pixels detected. Check scan quality or thresholds.")

    return clean_ink
