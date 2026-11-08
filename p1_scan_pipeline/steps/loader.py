"""
Step 1 — Load & Validate
Loads scan, checks DPI, converts to greyscale.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)


class LoadError(ValueError):
    pass


def load_image(image_path: str, config) -> np.ndarray:
    """
    Load image from path, validate, return greyscale numpy array.
    Raises LoadError on bad input.
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise LoadError(f"Could not load image: {image_path}")

    h, w = img.shape
    log.info(f"Loaded {image_path}: {w}×{h}px greyscale")

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

    return img
