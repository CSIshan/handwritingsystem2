"""
P2 Renderer
===========
ASCII rendering of word image crops and stroke path animation.
Designed for terminal display — no external dependencies.
"""
import time
import numpy as np
from typing import List, Dict, Any


# ASCII brightness ramp — darkest to lightest
_RAMP = " .:-=+*#%@"


def render_image_ascii(
    image:      np.ndarray,
    width:      int = 40,
    height:     int = 8,
) -> List[str]:
    """
    Render a greyscale word image as ASCII art lines.
    Returns list of strings, one per row.
    """
    if image is None or image.size == 0:
        return ["[no image]"]

    import cv2
    # Resize to target dimensions
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Invert: ink=255 in our binary, but we want ink=dark in ASCII
    # Our binary: 255=ink(dark), 0=background(light)
    # ASCII ramp: index 0=space(light), index 9=@(dark)
    lines = []
    for row in resized:
        line = ""
        for px in row:
            # px=255 → ink → dark character
            # px=0   → background → space
            idx = int((px / 255.0) * (len(_RAMP) - 1))
            line += _RAMP[idx]
        lines.append(line)

    return lines


def render_strokes_frame(
    strokes:    List[List],   # [[x, y, pen_up], ...]
    width:      int = 40,
    height:     int = 8,
    up_to:      int = None,   # draw only first N points (for animation)
) -> List[str]:
    """
    Render stroke sequence as ASCII art up to point index `up_to`.
    Returns list of strings.
    """
    if not strokes:
        return ["[no strokes]"] + [""] * (height - 1)

    points = strokes if up_to is None else strokes[:up_to]

    if not points:
        return [" " * width] * height

    # Extract XY, find bounds
    xs = [p[0] for p in points if p[2] == 0]  # only pen-down points
    ys = [p[1] for p in points if p[2] == 0]

    if not xs:
        return [" " * width] * height

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)

    # Build grid
    grid = [[" "] * width for _ in range(height)]

    prev_pen_up = True
    for i, (x, y, pen_up) in enumerate(points):
        if pen_up:
            prev_pen_up = True
            continue

        # Map to grid coords
        col = int((x - x_min) / x_range * (width  - 1))
        row = int((y_max - y) / y_range * (height - 1))  # flip Y
        col = max(0, min(width  - 1, col))
        row = max(0, min(height - 1, row))

        # Mark current point
        if i == len(points) - 1:
            grid[row][col] = "●"   # current pen position
        elif prev_pen_up:
            grid[row][col] = "○"   # pen-down start
        else:
            grid[row][col] = "·"   # stroke point

        prev_pen_up = False

    return ["".join(row) for row in grid]


def animate_strokes(
    strokes:    List[List],
    draw_func,              # callable(lines: List[str]) — renders to screen
    width:      int  = 40,
    height:     int  = 8,
    delay:      float = 0.025,  # seconds per point
    step:       int  = 3,       # points per frame (skip for speed)
):
    """
    Animate stroke sequence by progressively revealing points.
    draw_func is called with each frame's lines.
    """
    n = len(strokes)
    for i in range(0, n + 1, step):
        frame = render_strokes_frame(strokes, width, height, up_to=i)
        draw_func(frame)
        time.sleep(delay)

    # Final frame — all points
    final = render_strokes_frame(strokes, width, height)
    draw_func(final)
