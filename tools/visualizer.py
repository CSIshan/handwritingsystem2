"""
Universal pipeline visualizer.

Usage:
    from tools.visualizer import visualize
    visualize(components, background=word_crop, save_path="out.png")

Dispatches based on the type of `data` to the right renderer.
Add a new `_render_x()` function + dispatch check as new module outputs
(lines, words, curves, sequences) come online.
"""

import cv2
import numpy as np

_PALETTE = [
    (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
]


def visualize(data, background: np.ndarray, save_path: str = None, show: bool = True,
              show_skeleton: bool = False, skeleton: np.ndarray = None):
    """
    data: list of module-output objects (supports list[Component], list[Curve])
    background: the original crop/page image (grayscale or BGR) to draw over
    save_path: if given, saves the annotated image to this path
    show: if True, displays the image inline (matplotlib)
    show_skeleton: if True and data is list[Curve], overlays the raw skeleton
                   faintly underneath the fitted curves (needs `skeleton` arg)
    skeleton: uint8 skeleton array, required only if show_skeleton=True
    """
    if len(data) > 0 and hasattr(data[0], "mask") and hasattr(data[0], "bbox"):
        canvas = _render_components(data, background)
    elif len(data) > 0 and hasattr(data[0], "dense_points") and hasattr(data[0], "origin_offset"):
        canvas = _render_curves(data, background, show_skeleton=show_skeleton, skeleton=skeleton)
    else:
        raise NotImplementedError(f"No renderer for type: {type(data[0]) if data else type(data)}")

    if save_path:
        cv2.imwrite(save_path, canvas)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return canvas


def _render_components(components, background: np.ndarray) -> np.ndarray:
    if background.ndim == 2:
        canvas = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        canvas = background.copy()

    for i, comp in enumerate(components):
        color = _PALETTE[i % len(_PALETTE)]
        x, y, w, h = comp.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            canvas, f"a={comp.area}", (x, max(y - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
        )

    return canvas


def _render_curves(curves, background: np.ndarray, show_skeleton: bool = False,
                    skeleton: np.ndarray = None) -> np.ndarray:
    if background.ndim == 2:
        canvas = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        canvas = background.copy()

    canvas = (canvas * 0.35).astype(np.uint8)

    if show_skeleton:
        if skeleton is None:
            raise ValueError("show_skeleton=True requires the `skeleton` argument")
        sk_color = (90, 90, 90)
        canvas[skeleton > 0] = sk_color

    for i, curve in enumerate(curves):
        color = _PALETTE[i % len(_PALETTE)]

        if curve.degenerate:
            pt = curve.dense_points[0]
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, color, -1)
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)
        else:
            pts = curve.dense_points.astype(np.int32)
            for j in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), color, 1, cv2.LINE_AA)

    return canvas
