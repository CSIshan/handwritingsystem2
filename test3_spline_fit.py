"""Run a selected real word through the curve-fitting path and inspect it.

Example:
    python test3_spline_fit.py --line-index 0 --word-index 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np

from p1_scan_pipeline.config import ScanConfig
from p1_scan_pipeline.component_extractor import ComponentExtractor
from p1_scan_pipeline.steps.binarize import binarize
from p1_scan_pipeline.steps.dpi_estimator import DPITooLowError, estimate_dpi
from p1_scan_pipeline.steps.line_detector import detect_line_pairs
from p1_scan_pipeline.steps.segmenter import segment_words
from tuning_kit.curve_fitter import (
    BSplineFitter, HybridExtractor, RDPSuppressor, _find_branch_and_endpoints,
    _split_into_segments, bridge_missing_strokes, finalize_topology_smoothing,
    estimate_stroke_width, prune_spurs, recenter_skeleton,
)


DEFAULT_PAGE = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"
COLORS = [(50, 220, 50), (255, 180, 40), (230, 80, 220), (60, 190, 255),
          (255, 90, 90), (100, 240, 240)]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", default=DEFAULT_PAGE)
    parser.add_argument("--line-index", type=int, default=0)
    parser.add_argument("--word-index", type=int, default=8)
    parser.add_argument("--rdp-epsilon", type=float, default=2.0)
    parser.add_argument("--sample-spacing", type=float, default=0.5)
    parser.add_argument("--smoothing-per-point", type=float, default=0.35,
                        help="Gentle spline smoothing per RDP control point; 0 keeps exact interpolation.")
    parser.add_argument("--no-open", action="store_true",
                        help="Save outputs without opening the PNG in Preview.")
    return parser.parse_args()


def _selected_word(page_path: str, line_index: int, word_index: int):
    grey = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    config = ScanConfig()
    config.dpi = estimate_dpi(grey, config, min_dpi=300.0)
    binary = binarize(grey, config)
    pairs = detect_line_pairs(binary, config)
    chosen_line = [pair for pair in pairs if pair.index == line_index]
    if not chosen_line:
        raise IndexError(f"Line {line_index} was not detected")
    words = segment_words(binary, chosen_line, config)
    if not 0 <= word_index < len(words):
        raise IndexError(f"Word {word_index} is outside this line's {len(words)} words")
    return words[word_index], config


def _fit_component(component, rdp: RDPSuppressor, fitter: BSplineFitter):
    skeleton = HybridExtractor().extract(component.mask)
    skeleton = prune_spurs(skeleton)
    skeleton = recenter_skeleton(component.mask, skeleton)
    skeleton = bridge_missing_strokes(component.mask, skeleton)
    skeleton = finalize_topology_smoothing(skeleton)
    skeleton = prune_spurs(skeleton)
    branches, endpoints = _find_branch_and_endpoints(skeleton)
    curves = []
    for path in _split_into_segments(skeleton, branches, endpoints):
        controls = rdp.suppress(path)
        curves.append((controls, fitter.fit(controls)))
    return skeleton, curves


def _overlay(word, components, fitted) -> np.ndarray:
    scale = 3
    base = cv2.cvtColor(word.image, cv2.COLOR_GRAY2BGR)
    base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    controls_panel = base.copy()
    final_panel = base.copy()
    segment = 0
    for component, (skeleton, curves) in zip(components, fitted):
        # ComponentExtractor receives ``word.image``, so its bbox is already
        # relative to the selected word.  ``page_offset`` is metadata for
        # later page-coordinate output, not something to subtract here.
        x0, y0 = int(component.bbox[0]), int(component.bbox[1])
        ys, xs = np.where(skeleton > 0)
        valid = ((ys + y0 >= 0) & (ys + y0 < word.image.shape[0]) &
                 (xs + x0 >= 0) & (xs + x0 < word.image.shape[1]))
        controls_panel[(ys[valid] + y0) * scale, (xs[valid] + x0) * scale] = (165, 165, 165)
        for controls, curve in curves:
            color = COLORS[segment % len(COLORS)]
            segment += 1
            dense = np.rint(curve.dense_points).astype(int)
            dense[:, 0] = np.clip(dense[:, 0] + y0, 0, word.image.shape[0] - 1)
            dense[:, 1] = np.clip(dense[:, 1] + x0, 0, word.image.shape[1] - 1)
            # Keep input controls separate from the final curve. Otherwise
            # their large orange markers visually cut the fitted line apart.
            for y, x in np.rint(controls).astype(int):
                cv2.circle(controls_panel, ((x + x0) * scale, (y + y0) * scale), 3,
                           (0, 140, 255), -1, cv2.LINE_AA)
            cv2.polylines(final_panel, [dense[:, ::-1] * scale], False,
                          color, 3, cv2.LINE_AA)
    cv2.putText(controls_panel, "RDP controls + skeleton", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(final_panel, "Final fitted splines", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    gap = 12
    return np.hstack([controls_panel,
                      np.zeros((base.shape[0], gap, 3), dtype=np.uint8),
                      final_panel])


def _write_svg(word, components, fitted, output: str) -> None:
    """Write the fitted centre curves as black, pen-width SVG paths."""
    height, width = word.image.shape
    paths = []
    for component, (skeleton, curves) in zip(components, fitted):
        x0, y0 = int(component.bbox[0]), int(component.bbox[1])
        # estimate_stroke_width returns the radius / half-width. SVG's
        # stroke-width is the full diameter, which matches the original ink.
        stroke_width = 2 * estimate_stroke_width(component.mask, skeleton)
        for _, curve in curves:
            points = curve.dense_points
            if len(points) == 1:
                x, y = points[0][1] + x0, points[0][0] + y0
                paths.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{stroke_width / 2:.3f}"/>')
                continue
            d = " ".join(
                (f"M {points[0, 1] + x0:.3f} {points[0, 0] + y0:.3f}",
                 *[f"L {x + x0:.3f} {y + y0:.3f}" for y, x in points[1:]])
            )
            paths.append(
                f'<path d="{d}" fill="none" stroke="black" '
                f'stroke-width="{stroke_width:.3f}" stroke-linecap="round" '
                'stroke-linejoin="round"/>'
            )
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        *paths,
        '</svg>',
    ])
    Path(output).write_text(svg, encoding="utf-8")


def main() -> None:
    args = _arguments()
    try:
        word, config = _selected_word(args.page, args.line_index, args.word_index)
    except DPITooLowError as error:
        raise SystemExit(f"Page DPI is too low: {error}") from error
    extractor = ComponentExtractor(min_component_area=5,
        max_merge_height_px=config.mm_to_px(config.pair_gap_mm))
    components = extractor.merge_nearby(extractor.filter(
        extractor.extract(word.image, page_offset=(word.x_start, word.top_y))))
    rdp = RDPSuppressor(args.rdp_epsilon)
    fitter = BSplineFitter(args.sample_spacing,
                           smoothing_per_point=args.smoothing_per_point)
    fitted = [_fit_component(component, rdp, fitter) for component in components]
    result = _overlay(word, components, fitted)
    output = f"debug_spline_line{args.line_index}_word{args.word_index}.png"
    svg_output = f"debug_spline_line{args.line_index}_word{args.word_index}.svg"
    cv2.imwrite(output, result)
    _write_svg(word, components, fitted, svg_output)
    print(f"Saved {output}; components={len(components)}; "
          f"curves={sum(len(curves) for _, curves in fitted)}")
    print(f"Saved {svg_output}; black paths use each component's measured ink width")
    print(f"Spline smoothing: {args.smoothing_per_point} per RDP control point "
          "(pass --smoothing-per-point 0 for exact interpolation)")
    if not args.no_open:
        # Launch Preview independently: this script can exit immediately while
        # the image remains open for inspection.
        subprocess.Popen(["open", output], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
