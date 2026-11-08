"""
Contour-Based Stroke Extraction (Alternative)
================================================
An alternative to the skeleton pixel-walking approach in stroke_extractor.py.

Instead of skeletonizing then walking pixel-by-pixel (which struggles at
branch points — letter joins, loop crossings — because there's no robust
way to know which branch arm continues the "real" pen path), this method:

  1. Finds the outer contour of the ink blob (cv2.findContours)
  2. Simplifies it with Douglas-Peucker (cv2.approxPolyDP)
  3. Walks the simplified contour as the stroke path

This sacrifices true single-pixel-centerline topology (a contour traces
the OUTLINE of the ink, not its centerline, so strokes appear ~2x pen
width "hollow" rather than a single center path) but is far more robust:
contours never have to make an ambiguous choice at a branch point because
there ARE no branch points on an outer boundary — it's always a single
closed loop per connected ink region.

This is intentionally kept separate from stroke_extractor.py so both can
be run side-by-side for comparison on real data before deciding which
becomes the production path.
"""
import logging
import numpy as np
import cv2
from typing import List
from .stroke_extractor import StrokePoint

log = logging.getLogger(__name__)


def extract_strokes_contour(
    crop:       np.ndarray,
    baseline_y: int,
    strip_top:  int,
    config,
    epsilon_px: float = 1.5,
) -> List[StrokePoint]:
    """
    Contour-based alternative to extract_strokes() in stroke_extractor.py.
    Same signature and return format — drop-in comparable.

    epsilon_px: Douglas-Peucker simplification tolerance in pixels.
                Higher = smoother/simpler path, lower = more faithful
                to the exact contour (but noisier).
    """
    if crop is None or crop.size == 0:
        return []

    bool_img = crop > 0
    if not np.any(bool_img):
        return []

    # Find contours of the ink blob(s)
    contours, _ = cv2.findContours(
        crop.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Sort contours by area, largest first — draw the main body before
    # small disconnected dots (e.g. the dot on an 'i')
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    baseline_in_crop = baseline_y - strip_top
    px = config.px_per_mm

    # Compute x_offset from ALL contour points up front, not just the
    # first-processed (largest) contour. A short stroke like a t-bar or
    # an i-dot can sit further left than the main body; anchoring on
    # only the largest contour's first point would shift everything
    # else's relative x position incorrectly.
    all_raw_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x_offset = int(all_raw_pts[:, 0].min())

    all_points = []   # list of StrokePoint

    for contour in contours:
        # Simplify the contour to remove pixel-level jitter
        simplified = cv2.approxPolyDP(contour, epsilon_px, closed=True)
        pts = simplified.reshape(-1, 2)   # [[x,y], [x,y], ...]

        # A short, near-straight stroke (t-bar, i-dot, single tick mark)
        # legitimately simplifies down to 2 points after Douglas-Peucker.
        # Previously these were dropped entirely (len(pts) < 3 check),
        # silently discarding real ink. 2 points is still a valid line
        # segment — only truly degenerate single-point contours (noise
        # specks after simplification) should be skipped.
        if len(pts) < 2:
            continue

        for i, (px_x, px_y) in enumerate(pts):
            x_mm =  (px_x - x_offset) / px
            y_mm = -(px_y - baseline_in_crop) / px
            pen_up = 1 if i == len(pts) - 1 else 0   # lift at end of each contour loop
            all_points.append(StrokePoint(x=float(x_mm), y=float(y_mm), pen_up=pen_up))

    return all_points
