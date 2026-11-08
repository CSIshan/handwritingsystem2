"""
P1 Scan Pipeline — Configuration
=================================
All measurements derived from the actual writing template.
Change these if the template changes — nothing else needs updating.p
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    # ── Scan properties ───────────────────────────────────────────────────────
    dpi:                  int   = 350

    # ── Template geometry (must match writing_template.pdf) ───────────────────
    n_pairs:              int   = 10
    pair_gap_mm:          float = 8.0    # bottom to top line within pair
    inter_pair_mm:        float = 16.7   # bottom line of pair N to top line of pair N+1
    top_offset_mm:        float = 10.0   # from usable top to first line
    margin_top_mm:        float = 20.0
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
    position_tol_mm:      float = 4.0    # how far a detected line can be from expected

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
