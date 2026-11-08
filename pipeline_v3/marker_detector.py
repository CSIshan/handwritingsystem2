"""
Marker Detection — Red Start-Dots and Blue Exclusion Boxes
==============================================================
Detects two kinds of hand-added digital markers on an otherwise-grayscale
scan:

  RED dot   — marks the intended starting point of a stroke/word.
  BLUE box  — a hollow outline around a mistake; the enclosed content
              should be excluded entirely (whole word dropped).

Thresholds below were measured directly on a real reference page
(page7.png), not guessed:
  - Real ink pixels are PERFECTLY neutral (R == G == B exactly) --
    confirmed because the page was fully grayscaled before markers were
    digitally added. This means ANY color difference between channels is
    unambiguously a marker, never real ink.
  - Blue marker core color measured ~BGR(246,134,79): B-R gap up to ~170.
  - Red marker core color measured ~BGR(37,64,234): R-B gap up to ~197.
  - A threshold of 40 on the channel difference separates all three
    (ink, red marker, blue marker) with enormous margin -- not a fragile
    cutoff.
  - The red dot measured ~6x6px, a single small connected blob.
  - Blue boxes are HOLLOW OUTLINES, not filled -- confirmed by sampling
    each box's exact center pixel and finding blank paper there, not
    solid blue. Detection must fill the traced outline to get the
    excluded interior, not just use the outline pixels themselves.
"""
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple

log = logging.getLogger(__name__)


@dataclass
class PageMarkers:
    exclusion_mask: np.ndarray            # bool, HxW, True = inside a blue box (drop this content)
    start_points:   List[Tuple[int, int]] = field(default_factory=list)  # (y, x) page coords, one per red dot


def detect_markers(
    color_img: np.ndarray,
    channel_diff_threshold: int = 40,
    min_dot_area: int = 4,
    max_dot_area: int = 400,
) -> PageMarkers:
    """
    color_img: BGR color scan (as loaded by cv2.imread(..., IMREAD_COLOR)).

    Returns PageMarkers with the exclusion mask (interior of every blue
    box, filled) and a list of red dot centroids in page (y, x) coords.
    """
    if color_img.ndim != 3 or color_img.shape[2] != 3:
        raise ValueError("color_img must be a 3-channel BGR image")

    b = color_img[:, :, 0].astype(int)
    g = color_img[:, :, 1].astype(int)
    r = color_img[:, :, 2].astype(int)

    blue_mask = (b - r) > channel_diff_threshold
    red_mask = (r - b) > channel_diff_threshold

    exclusion_mask = _fill_blue_outlines(blue_mask)
    start_points = _find_red_dot_centroids(red_mask, min_dot_area, max_dot_area)

    log.info(
        f"Marker detection: {len(start_points)} start-dot(s), "
        f"exclusion area covers {int(exclusion_mask.sum())}px"
    )

    return PageMarkers(exclusion_mask=exclusion_mask, start_points=start_points)


def _fill_blue_outlines(blue_mask: np.ndarray) -> np.ndarray:
    """
    Blue boxes are hollow outlines. Trace each outline's contour and fill
    its interior -- the excluded region is everything INSIDE the box,
    not just the thin blue border pixels themselves.
    """
    blue_u8 = (blue_mask.astype(np.uint8)) * 255
    # Close small gaps in the traced outline (anti-aliasing can leave the
    # border a pixel or two thin/broken in places) so findContours sees
    # one clean closed loop per box instead of a leaky boundary.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(blue_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros(blue_mask.shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    return filled > 0


def _find_red_dot_centroids(
    red_mask: np.ndarray, min_dot_area: int, max_dot_area: int
) -> List[Tuple[int, int]]:
    """
    Each red dot is one small connected blob. Filter by area to reject
    both single-pixel noise and anything implausibly large (which would
    suggest a misdetection, not a genuine small hand-placed dot).
    """
    red_u8 = (red_mask.astype(np.uint8)) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(red_u8, 8)

    points = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_dot_area <= area <= max_dot_area:
            cx, cy = centroids[label]
            points.append((int(round(cy)), int(round(cx))))  # (y, x)
        else:
            log.warning(
                f"Red-colored region at label {label} rejected as a start-dot "
                f"candidate: area={area}px is outside the expected "
                f"[{min_dot_area}, {max_dot_area}] range."
            )

    return points


def remove_markers_from_image(color_img: np.ndarray, markers: PageMarkers) -> np.ndarray:
    """
    Returns a COLOR image with the exclusion regions AND the red dots
    themselves blanked to plain white paper -- ready to convert to
    grayscale for the rest of the pipeline, which should never see marker
    ink or excluded mistake-ink at all.

    Safe to do precisely here (unlike a hypothetical physical marker on
    physical ink): real ink in this workflow is provably always perfectly
    neutral, so any colored pixel is unambiguously marker, never ink --
    there is no risk of this accidentally erasing real handwriting.
    """
    working = color_img.copy()
    working[markers.exclusion_mask] = (255, 255, 255)

    b = color_img[:, :, 0].astype(int)
    r = color_img[:, :, 2].astype(int)
    red_mask = (r - b) > 40
    # Dilate slightly to also catch the dot's anti-aliased fringe pixels,
    # which fall just under the strict threshold but are still visibly
    # part of the dot, not paper.
    red_u8 = (red_mask.astype(np.uint8)) * 255
    red_dilated = cv2.dilate(red_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    working[red_dilated > 0] = (255, 255, 255)

    return working
