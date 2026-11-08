"""
Hybrid Stroke Extraction
==========================
Combines the strengths of both prior approaches:

  - Contour-based cleaning (robust to noise, preserves loop holes)
  - Skeleton centerline extraction (true single-pixel pen path, not an
    outline — required for actual plotting, unlike raw contours)
  - The existing direction-aware pixel walker (already correct once it
    isn't confused by noisy branch points)

Pipeline:
    1. Find contours with RETR_CCOMP (separates outer boundaries from
       inner holes per connected component — critical so loops like 'o'
       don't get solid-filled and lose their hole)
    2. Rebuild a CLEAN binary mask: fill outer contours solid, then
       re-punch inner hole contours back out. This removes small noise
       specks and pixel-level jitter that were creating spurious branch
       points in skeletonization, while correctly preserving real loops.
    3. Skeletonize the cleaned mask (same Zhang-Suen thinning as before)
    4. Walk the skeleton with the existing direction-aware tracer
    5. Normalize to baseline=0 mm space (same as before)

This produces a true centerline (plottable), not an outline, while
benefiting from the noise-robustness that made the contour comparison
look visually cleaner.
"""
import logging
import numpy as np
import cv2
from typing import List
from skimage.morphology import medial_axis

from .stroke_extractor import (
    StrokePoint,
    _skeleton_to_paths,
    _order_paths,
    _normalize,
    _smooth_strokes,
    _smooth_positions,
    _reconnect_compatible,
    _filter_short,
)

log = logging.getLogger(__name__)


def _build_clean_mask(crop: np.ndarray) -> np.ndarray:
    """
    Rebuild a denoised binary mask from contours, preserving holes.

    RETR_CCOMP gives a two-level hierarchy: outer boundaries at one
    level, inner holes (e.g. the inside of an 'o' or 'e' loop) at the
    next level. We fill all outer contours solid, then subtract the
    inner hole contours back out — so loops stay correctly hollow
    instead of becoming solid blobs.
    """
    binary = (crop > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours or hierarchy is None:
        return binary

    clean = np.zeros_like(binary)
    hierarchy = hierarchy[0]   # shape [N, 4]: next, prev, first_child, parent

    for i, contour in enumerate(contours):
        parent = hierarchy[i][3]
        if parent == -1:
            # Outer contour — fill solid
            cv2.drawContours(clean, contours, i, 255, thickness=cv2.FILLED)

    for i, contour in enumerate(contours):
        parent = hierarchy[i][3]
        if parent != -1:
            # Inner hole contour — punch back out
            cv2.drawContours(clean, contours, i, 0, thickness=cv2.FILLED)

    return clean


def _prune_skeleton_spurs(
    skeleton:           np.ndarray,   # binary skeleton image (0/255)
    distance:           np.ndarray,   # same-shape distance transform from medial_axis
    min_branch_factor:  float = 1.5,
    max_iterations:     int = 10,
) -> np.ndarray:
    """
    Remove short spurious branches from a medial-axis skeleton.

    A "branch" is the run of skeleton pixels between an endpoint
    (degree 1) and the nearest junction (degree >= 3). Its pixel length
    is compared against the LOCAL stroke radius — the medial_axis
    distance-transform value at the junction it attaches to — rather
    than a fixed pixel threshold, since stroke width varies across a
    word. Genuine strokes are long relative to their local thickness;
    a noise-induced spur (the artifact responsible for medial axis's
    inflated branch-point counts vs Zhang-Suen) is typically short
    relative to it. Branches shorter than min_branch_factor * radius
    are removed.

    Iterative: removing a branch can drop what used to be a degree>=3
    junction down to degree 2 (no longer a junction) or expose a new,
    shorter spur at it, so we repeat until nothing more qualifies for
    removal or max_iterations is hit.

    Validated on synthetic data (see extract_strokes_medial_axis_pruned
    docstring) — roughly halves branch-point count vs unpruned medial
    axis, in one case dropping it below Zhang-Suen's count entirely,
    though it does not fully close the gap on a perfectly clean shape.
    """
    skel = (skeleton > 0).copy()

    def neighbours(pixel_set, x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in pixel_set:
                    yield n

    for _ in range(max_iterations):
        ys, xs = np.where(skel)
        if len(xs) == 0:
            break
        pixel_set = set(zip(xs.tolist(), ys.tolist()))

        def degree(x, y):
            return sum(1 for _ in neighbours(pixel_set, x, y))

        endpoints = [(x, y) for x, y in pixel_set if degree(x, y) == 1]
        to_remove = set()

        for ep in endpoints:
            if ep in to_remove:
                continue
            branch_pixels  = [ep]
            visited_branch = {ep}
            current = ep
            while True:
                nbrs = [n for n in neighbours(pixel_set, *current)
                         if n not in visited_branch]
                if not nbrs:
                    break
                if len(nbrs) == 1:
                    nxt = nbrs[0]
                    branch_pixels.append(nxt)
                    visited_branch.add(nxt)
                    if degree(*nxt) >= 3:
                        break
                    current = nxt
                else:
                    break

            attach = branch_pixels[-1]
            if degree(*attach) < 3:
                # Walked all the way to a real endpoint, not a junction
                # (e.g. an isolated short stroke like an i-dot or t-bar
                # crossing fragment) — not a spur, leave it alone.
                continue

            branch_len = len(branch_pixels) - 1
            radius     = distance[attach[1], attach[0]]
            threshold  = max(2.0, min_branch_factor * radius)

            if branch_len < threshold:
                for p in branch_pixels[:-1]:
                    to_remove.add(p)

        if not to_remove:
            break
        for (x, y) in to_remove:
            skel[y, x] = False

    return (skel.astype(np.uint8)) * 255


def _medial_axis_pipeline(
    mask:       np.ndarray,
    baseline_y: int,
    strip_top:  int,
    config,
    prune:      bool = False,
) -> List[StrokePoint]:
    """
    Shared back half of the medial-axis pipeline: skeletonize a binary
    mask via medial_axis, optionally prune spurious branches, walk it,
    normalize, smooth, filter. Used by extract_strokes_hybrid() (mask
    pre-cleaned via contours), extract_strokes_medial_axis_raw() (mask
    used as-is, no pruning), and extract_strokes_medial_axis_pruned()
    (mask used as-is, with pruning) — so all three stay in sync and
    only the preprocessing/pruning differs.
    """
    skel_bool, distance = medial_axis(mask > 0, return_distance=True)
    skeleton = skel_bool.astype(np.uint8) * 255

    if prune:
        skeleton = _prune_skeleton_spurs(skeleton, distance)

    pixel_paths = _skeleton_to_paths(skeleton)
    if not pixel_paths:
        return []

    ordered = _order_paths(pixel_paths, config)

    baseline_in_crop = baseline_y - strip_top
    strokes = _normalize(ordered, baseline_in_crop, config)
    strokes = _smooth_strokes(strokes)
    strokes = _smooth_positions(strokes)
    strokes = _reconnect_compatible(strokes)
    strokes = _filter_short(strokes, config.min_stroke_points)

    return strokes


def extract_strokes_hybrid(
    crop:       np.ndarray,
    baseline_y: int,
    strip_top:  int,
    config,
) -> List[StrokePoint]:
    """
    Hybrid extraction: contour-cleaned mask → medial axis → direction-aware
    trace → normalized mm strokes. Same signature/return type as
    extract_strokes() in stroke_extractor.py — drop-in comparable.
    """
    if crop is None or crop.size == 0:
        return []

    bool_img = crop > 0
    if not np.any(bool_img):
        return []

    # ── Clean the mask via contours (denoise, preserve loop holes) ───────────
    clean_mask = _build_clean_mask(crop)

    if not np.any(clean_mask > 0):
        # Cleaning removed everything (e.g. crop was pure noise) —
        # fall back to the original crop rather than returning nothing.
        clean_mask = (crop > 0).astype(np.uint8) * 255

    return _medial_axis_pipeline(clean_mask, baseline_y, strip_top, config)


def extract_strokes_medial_axis_raw(
    crop:       np.ndarray,
    baseline_y: int,
    strip_top:  int,
    config,
) -> List[StrokePoint]:
    """
    Medial axis WITHOUT contour-cleaning preprocessing — applied
    directly to the raw binarized crop. Added to isolate whether the
    branch-point sprawl seen in extract_strokes_hybrid() comes from the
    contour-cleaning step or is inherent to medial_axis itself on
    rasterized handwriting strokes.

    Empirical check (synthetic data, see project notes): contour
    cleaning made it slightly WORSE (46 branch points vs 40 without
    cleaning, on the same noisy synthetic stroke), and even on a
    perfectly clean synthetic curve with zero boundary noise, raw
    medial_axis still produced 23 spurious branch points vs 0 for
    Zhang-Suen skeletonize. This variant exists so that can be verified
    visually on real scan data via compare_extraction.py, rather than
    only on synthetic shapes.
    """
    if crop is None or crop.size == 0:
        return []

    bool_img = crop > 0
    if not np.any(bool_img):
        return []

    raw_mask = (crop > 0).astype(np.uint8) * 255
    return _medial_axis_pipeline(raw_mask, baseline_y, strip_top, config)


def extract_strokes_medial_axis_pruned(
    crop:       np.ndarray,
    baseline_y: int,
    strip_top:  int,
    config,
) -> List[StrokePoint]:
    """
    Medial axis on the raw mask (no contour cleaning), WITH spur
    pruning applied to the resulting skeleton before tracing.

    This is the fair-comparison variant: extract_strokes_medial_axis_raw
    showed medial axis's branch-point sprawl is inherent to the
    algorithm, not caused by contour-cleaning preprocessing. Pruning is
    the standard, principled fix for that specific weakness — using the
    distance transform medial_axis already computes (and which the
    other two variants discard) to tell genuine strokes apart from
    noise-induced spurs.

    Empirical check (synthetic data): on a noisy synthetic stroke,
    pruning took medial axis from 38-40 branch points down to 16 —
    actually below Zhang-Suen skeletonize's 38 on that case. On a
    perfectly clean synthetic curve (zero boundary noise), pruning took
    it from 23 down to 12 — a real improvement, but it does not fully
    close the gap to Zhang-Suen's 0 on that idealized case. Wired into
    compare_extraction.py as a 5th column so this can be checked
    against real handwriting, not just synthetic shapes.
    """
    if crop is None or crop.size == 0:
        return []

    bool_img = crop > 0
    if not np.any(bool_img):
        return []

    raw_mask = (crop > 0).astype(np.uint8) * 255
    return _medial_axis_pipeline(raw_mask, baseline_y, strip_top, config, prune=True)
