"""
P1 Scan Pipeline — Configuration
=================================
All measurements derived from the actual writing template.
Change these if the template changes — nothing else needs updating.

TEMPLATE GEOMETRY, updated: the real notebook uses plain, evenly-spaced
SINGLE ruled lines (confirmed against a real scan, page7.png) -- not the
two-line-per-row (cap-height guide + baseline) structure this file
originally assumed. That old "pair" model never matched the real
template and has been replaced below.

Measured directly from page7.png:
  - line_spacing_mm: 9 consecutive line-to-line gaps averaged 223.1px;
    given real spacing 7.8-7.9mm, that implies DPI ~717-727 -- i.e. this
    is a real, working, self-consistent measurement, not a guess.
  - first_line_offset_mm: the first ruled line sits at y=438px on that
    same scan (~15.4mm at the DPI above).
  - assumed_cap_height_mm: NOT measurable from the template itself --
    there's no printed guide for it. 8.0mm per your own estimate
    ("less than or only sometimes greater than the line width").
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    # ── Scan properties ───────────────────────────────────────────────────────
    dpi:                  int   = 350   # always overwritten per-page by dpi_estimator

    # ── Template geometry (single ruled lines -- see module docstring) ────────
    n_lines:               int   = 10     # lines per page
    line_spacing_mm:        float = 7.85   # baseline-to-baseline spacing (measured)
    first_line_offset_mm:   float = 15.4   # page top to first ruled line (measured)
    assumed_cap_height_mm:  float = 8.0    # NOT measured -- see docstring; tune later
                                            # against real word-height stats
    margin_bottom_mm:     float = 20.0
    margin_left_mm:       float = 20.0
    margin_right_mm:      float = 20.0

    # ── Binarization ──────────────────────────────────────────────────────────
    # Grey value below which a pixel is considered ink (gel pen)
    ink_threshold:        int   = 80
    # Grey value below which a pixel is considered printed line or ink
    # Printed grey lines sit around 160-180; ink sits below 80
    # Anything between 80-180 = printed template line → discard
    line_threshold:       int   = 180

    # ── Word segmentation ─────────────────────────────────────────────────────
    min_word_width_mm:    float = 3.0
    max_word_width_mm:    float = 60.0
    min_word_height_mm:   float = 2.0
    max_word_height_mm:   float = 18.0
    min_gap_width_mm:     float = 1.5    # minimum valley width to split words

    # ── Line detection tolerance ──────────────────────────────────────────────
    # Must stay comfortably under half of line_spacing_mm (3.925mm) or the
    # search window for one line can wander into its neighbor's territory.
    # Old default (4.0mm) was sized for the old, much wider pair spacing and
    # is unsafe here.
    position_tol_mm:      float = 2.0    # how far a detected line can be from expected

    # ── Stroke extraction ─────────────────────────────────────────────────────
    min_stroke_points:    int   = 3      # discard strokes shorter than this
    max_jump_mm:          float = 0.25   # max gap to treat as continuous stroke vs pen-up

    # ── Outlier detection (IQR-based) ─────────────────────────────────────────
    outlier_iqr_factor:   float = 2.0   # pages > 2×IQR from median are flagged

    # ── Derived: pixels per mm ────────────────────────────────────────────────
    @property
    def px_per_mm(self) -> float:
        return self.dpi / 25.4

    def mm_to_px(self, mm: float) -> int:
        return int(round(mm * self.px_per_mm))

    def px_to_mm(self, px: float) -> float:
        return px / self.px_per_mm
