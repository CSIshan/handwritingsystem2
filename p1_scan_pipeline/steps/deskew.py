"""
Step 2 — Deskew
Detects page tilt via Hough line transform and corrects it.
Typical scanner misalignment: ±1–3 degrees.
"""
import logging
import numpy as np
import cv2

log = logging.getLogger(__name__)


def deskew(grey: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Correct page tilt. Returns (corrected_image, angle_degrees_corrected).
    Positive angle = image was rotated clockwise, we corrected CCW.
    """
    # Threshold loosely to get all ink for angle detection
    _, binary = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY_INV)

    # Hough line transform on edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None or len(lines) == 0:
        log.warning("Deskew: no lines detected, skipping rotation correction")
        return grey, 0.0

    # Collect angles of near-horizontal lines (within ±10° of horizontal)
    angles = []
    for line in lines:
        rho, theta = line[0]
        # theta=0 is vertical, theta=pi/2 is horizontal
        angle_from_horizontal = np.degrees(theta) - 90
        if abs(angle_from_horizontal) < 10:
            angles.append(angle_from_horizontal)

    if not angles:
        log.info("Deskew: no near-horizontal lines found, skipping")
        return grey, 0.0

    # Use median angle — robust to outlier lines
    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.1:
        log.info(f"Deskew: angle {median_angle:.2f}° negligible, skipping")
        return grey, 0.0

    log.info(f"Deskew: correcting {median_angle:.2f}° rotation")

    # Rotate image to correct the tilt
    h, w = grey.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
    corrected = cv2.warpAffine(
        grey, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected, median_angle
