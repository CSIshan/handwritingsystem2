"""
Tests for contour-based stroke extraction.
Specifically guards against the missing-strokes regression where short
straight segments (t-bars, dots) simplified to 2 points and were dropped.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import cv2
from handwriting_system.p1_scan_pipeline.config import ScanConfig
from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_contour import extract_strokes_contour


def make_config():
    return ScanConfig(dpi=150)


class TestContourExtraction(unittest.TestCase):

    def test_empty_crop_returns_empty(self):
        cfg = make_config()
        crop = np.zeros((20, 20), dtype=np.uint8)
        result = extract_strokes_contour(crop, baseline_y=15, strip_top=5, config=cfg)
        self.assertEqual(result, [])

    def test_short_straight_line_not_dropped(self):
        """
        Regression test: a perfectly straight short stroke (t-bar-like)
        simplifies to exactly 2 points after approxPolyDP. The old
        `len(pts) < 3` check silently discarded these, making t-bars
        and similar marks vanish entirely from extraction.
        """
        cfg = make_config()
        crop = np.zeros((20, 60), dtype=np.uint8)
        cv2.line(crop, (10, 10), (50, 10), 255, 2)
        result = extract_strokes_contour(crop, baseline_y=15, strip_top=5, config=cfg)
        self.assertGreater(len(result), 0,
            "Short straight stroke was dropped — missing-strokes regression")

    def test_tiny_dot_not_dropped(self):
        cfg = make_config()
        crop = np.zeros((20, 20), dtype=np.uint8)
        cv2.circle(crop, (10, 10), 2, 255, -1)
        result = extract_strokes_contour(crop, baseline_y=15, strip_top=5, config=cfg)
        self.assertGreater(len(result), 0, "Tiny dot was dropped")

    def test_loop_produces_points(self):
        cfg = make_config()
        crop = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(crop, (20, 20), 12, 255, 3)
        result = extract_strokes_contour(crop, baseline_y=32, strip_top=5, config=cfg)
        self.assertGreater(len(result), 3)

    def test_multiple_disconnected_strokes_all_present(self):
        """
        Word-like crop with a main body plus a separate small mark
        that does NOT touch it (simulating an i-dot floating above its
        stem, or a t-bar drawn slightly apart from the stem).
        Both pieces should appear in the output.
        """
        cfg = make_config()
        crop = np.zeros((40, 80), dtype=np.uint8)
        # Main vertical stem
        cv2.line(crop, (20, 15), (20, 35), 255, 3)
        # Genuinely disconnected short horizontal mark, well clear of the stem
        cv2.line(crop, (10, 5), (30, 5), 255, 2)

        result = extract_strokes_contour(crop, baseline_y=35, strip_top=5, config=cfg)

        pen_ups = sum(1 for p in result if p.pen_up == 1)
        self.assertGreaterEqual(pen_ups, 2,
            "Expected at least 2 separate strokes (main body + mark), "
            "got fewer pen-lifts than disconnected shapes drawn")

    def test_x_offset_uses_leftmost_point_across_all_contours(self):
        """
        Regression test: x_offset was previously taken from only the
        first-processed (largest) contour's first point, which could
        misplace a smaller contour that extends further left/right.
        """
        cfg = make_config()
        crop = np.zeros((40, 100), dtype=np.uint8)
        # Large main body, starts at x=40
        cv2.circle(crop, (50, 20), 15, 255, 3)
        # Small mark further LEFT than the main body, at x=5
        cv2.line(crop, (5, 10), (15, 10), 255, 2)

        result = extract_strokes_contour(crop, baseline_y=35, strip_top=5, config=cfg)
        xs = [p.x for p in result]
        # With correct global x_offset, no point should be negative
        # (everything anchored to the true leftmost point across all contours)
        self.assertGreaterEqual(min(xs), -0.5,
            f"Found unexpectedly negative x ({min(xs):.2f}) — x_offset "
            f"likely anchored to wrong contour")

    def test_pen_up_values_valid(self):
        cfg = make_config()
        crop = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(crop, (20, 20), 10, 255, 2)
        result = extract_strokes_contour(crop, baseline_y=32, strip_top=5, config=cfg)
        for p in result:
            self.assertIn(p.pen_up, [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
