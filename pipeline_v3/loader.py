"""
Step 1 — Load & Validate
Loads scan, checks DPI, detects/removes markers, converts to greyscale.
"""
import logging
import numpy as np
import cv2

from .marker_detector import detect_markers, remove_markers_from_image, PageMarkers

log = logging.getLogger(__name__)


class LoadError(ValueError):
    pass


def load_image(image_path: str, config) -> tuple[np.ndarray, "PageMarkers"]:
    """
    Load image from path, validate, detect/remove red-dot and blue-box
    markers, return (greyscale array, PageMarkers).

    CHANGED: now loads COLOR, not grayscale directly -- markers (red
    start-dots, blue exclusion boxes) are only distinguishable from real
    ink in the original color data; converting to grayscale first (the
    old behaviour) makes them indistinguishable from ink and unusable.

    Raises LoadError on bad input.
    """
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color is None:
        raise LoadError(f"Could not load image: {image_path}")

    h, w = color.shape[:2]
    log.info(f"Loaded {image_path}: {w}×{h}px color")

    # Estimate DPI from image dimensions
    # A4 = 210×297mm. If width ≈ 210mm worth of pixels, derive DPI
    estimated_dpi = int(w / (210 / 25.4))
    if estimated_dpi < 400:
        log.warning(
            f"Estimated DPI ~{estimated_dpi} is below recommended 600. "
            f"Stroke extraction quality may suffer."
        )

    # Sanity check: image should be roughly A4 portrait proportions
    aspect = h / w
    if not (1.3 < aspect < 1.6):
        log.warning(
            f"Image aspect ratio {aspect:.2f} doesn't match A4 portrait (1.41). "
            f"Is the page rotated or cropped?"
        )

    markers = detect_markers(color)
    cleaned = remove_markers_from_image(color, markers)
    grey = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    return grey, markers
