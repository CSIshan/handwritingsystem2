"""
P4 Preview — Quick-look Debug SVG
===================================
Renders generated strokes to a simple SVG for visual inspection.
Not the full P9 SVG composer — just raw stroke paths for debugging.

Usage:
    from handwriting_system.p4_inference.preview import preview_word, preview_words
    preview_word(engine, "hello", "debug_hello.svg")
    preview_words(engine, ["hello", "world", "the"], "debug_words.svg")
"""
import logging
from pathlib import Path
from typing import List, Optional
from .engine import InferenceEngine, StrokePoint

log = logging.getLogger(__name__)

# SVG page dimensions for debug preview (px at 96dpi)
PREVIEW_W    = 800
PREVIEW_H    = 300
MARGIN       = 40
SCALE        = 8.0    # mm → px scale for debug view
STROKE_COLOR = "#1a1a1a"
STROKE_WIDTH = 1.5
BASELINE_COLOR = "#cccccc"
GUIDE_COLOR    = "#eeeeee"


def _strokes_to_path(strokes: List[StrokePoint], x_off: float, y_off: float) -> str:
    """Convert stroke points to SVG path data string."""
    if not strokes:
        return ""

    parts  = []
    pen_up = True

    for pt in strokes:
        px = x_off + pt.x * SCALE
        py = y_off - pt.y * SCALE   # flip Y: positive Y = up in SVG

        if pen_up:
            parts.append(f"M {px:.2f} {py:.2f}")
        else:
            parts.append(f"L {px:.2f} {py:.2f}")

        pen_up = bool(pt.pen_up)

    return " ".join(parts)


def preview_word(
    engine:      InferenceEngine,
    word:        str,
    output_path: str,
    temperature: float = 1.0,
    n_variants:  int   = 3,
):
    """
    Generate n_variants of a word and render them stacked vertically.
    Useful for checking variation across generations.
    """
    output_path = Path(output_path)
    h = MARGIN * 2 + n_variants * (PREVIEW_H // n_variants)

    paths  = []
    labels = []

    for i in range(n_variants):
        strokes = engine.generate(word, temperature=temperature)
        y_off   = MARGIN + (i + 1) * (h // (n_variants + 1))
        path_d  = _strokes_to_path(strokes, MARGIN, y_off)
        if path_d:
            paths.append((path_d, y_off))
            bbox = engine.get_bbox(word)
            labels.append(
                f"var {i+1}: {len(strokes)} pts  "
                f"w={bbox[0]:.1f}mm asc={bbox[1]:.1f}mm desc={bbox[2]:.1f}mm"
            )

    svg = _build_svg(PREVIEW_W, h, paths, labels, word)
    output_path.write_text(svg, encoding="utf-8")
    log.info(f"Preview saved: {output_path}")
    return str(output_path)


def preview_words(
    engine:      InferenceEngine,
    words:       List[str],
    output_path: str,
    temperature: float = 1.0,
):
    """
    Generate one variant per word, laid out left-to-right on one line.
    Useful for checking a sentence or word list at a glance.
    """
    output_path = Path(output_path)
    baseline_y  = PREVIEW_H // 2

    paths  = []
    labels = []
    x_cursor = MARGIN

    for word in words:
        strokes = engine.generate(word, temperature=temperature)
        path_d  = _strokes_to_path(strokes, x_cursor, baseline_y)
        if path_d:
            paths.append((path_d, baseline_y))
            labels.append(word)

        # Advance cursor by word width + gap
        if strokes:
            word_w = max(s.x for s in strokes) * SCALE
        else:
            word_w = 40
        x_cursor += word_w + MARGIN * 0.5

    # Expand canvas if words overflow
    w = max(PREVIEW_W, int(x_cursor) + MARGIN)

    svg = _build_svg(w, PREVIEW_H, paths, labels, " ".join(words))
    output_path.write_text(svg, encoding="utf-8")
    log.info(f"Preview saved: {output_path}")
    return str(output_path)


def _build_svg(
    w:      int,
    h:      int,
    paths:  List[tuple],
    labels: List[str],
    title:  str,
) -> str:
    """Build complete SVG string."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" '
        f'style="background:#fafafa;font-family:monospace">',
        f'  <title>Handwriting Preview: {title}</title>',
    ]

    # Title
    lines.append(
        f'  <text x="{MARGIN}" y="20" '
        f'font-size="12" fill="#888">{title}</text>'
    )

    # Baseline guides and paths
    for i, (path_d, y_off) in enumerate(paths):
        # Baseline guide
        lines.append(
            f'  <line x1="{MARGIN//2}" y1="{y_off}" '
            f'x2="{w - MARGIN//2}" y2="{y_off}" '
            f'stroke="{BASELINE_COLOR}" stroke-width="1" '
            f'stroke-dasharray="4,4"/>'
        )

        # Stroke path
        lines.append(
            f'  <path d="{path_d}" '
            f'stroke="{STROKE_COLOR}" stroke-width="{STROKE_WIDTH}" '
            f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        )

        # Label
        if i < len(labels):
            lines.append(
                f'  <text x="{MARGIN}" y="{y_off + 25}" '
                f'font-size="10" fill="#aaa">{labels[i]}</text>'
            )

    lines.append('</svg>')
    return "\n".join(lines)
