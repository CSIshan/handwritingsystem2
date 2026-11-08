"""
P1 Test Suite
=============
Tests using synthetic images — no real scans needed.
Generates controlled binary images to verify each pipeline step.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import unittest
import numpy as np
import cv2
from handwriting_system.p1_scan_pipeline.config import ScanConfig
from handwriting_system.p1_scan_pipeline.steps.binarize import binarize
from handwriting_system.p1_scan_pipeline.steps.line_detector import detect_line_pairs
from handwriting_system.p1_scan_pipeline.steps.segmenter import segment_words
from handwriting_system.p1_scan_pipeline.steps.stroke_extractor import (
    extract_strokes, StrokePoint, _skeleton_to_paths, _smooth_strokes,
    _smooth_positions, _reconnect_compatible, _order_paths,
)
from handwriting_system.p1_scan_pipeline.stats import measure_page_stats, _percentile_dist
from handwriting_system.p1_scan_pipeline.output import compute_bbox_mm


def make_config():
    return ScanConfig(dpi=150)   # lower DPI for faster tests


def synthetic_page(config: ScanConfig) -> np.ndarray:
    """
    Generate a synthetic A4 greyscale page with:
    - White background
    - Grey template lines at known positions
    - Black ink blobs simulating words
    """
    h  = config.mm_to_px(297)
    w  = config.mm_to_px(210)
    img = np.ones((h, w), dtype=np.uint8) * 240   # off-white paper

    margin_top = config.mm_to_px(config.margin_top_mm)
    top_offset = config.mm_to_px(config.top_offset_mm)
    pair_gap   = config.mm_to_px(config.pair_gap_mm)
    inter_pair = config.mm_to_px(config.inter_pair_mm)

    for i in range(config.n_pairs):
        y_top      = int(margin_top + top_offset + i * (pair_gap + inter_pair))
        y_baseline = y_top + pair_gap
        cv2.line(img, (0, y_top),      (w, y_top),      160, 2)
        cv2.line(img, (0, y_baseline), (w, y_baseline), 160, 2)

    margin_left = config.mm_to_px(config.margin_left_mm)
    for i in range(config.n_pairs):
        y_top      = int(margin_top + top_offset + i * (pair_gap + inter_pair))
        y_baseline = y_top + pair_gap
        y_mid      = (y_top + y_baseline) // 2
        x = margin_left
        for j in range(3):
            word_w = config.mm_to_px(12)
            word_h = config.mm_to_px(4)
            cv2.rectangle(img,
                (x, y_mid - word_h // 2),
                (x + word_w, y_mid + word_h // 2),
                30, -1)
            x += word_w + config.mm_to_px(5)

    return img


class TestConfig(unittest.TestCase):

    def test_px_per_mm(self):
        cfg = make_config()
        self.assertAlmostEqual(cfg.px_per_mm, 150 / 25.4, places=2)

    def test_mm_to_px_roundtrip(self):
        cfg = make_config()
        self.assertAlmostEqual(cfg.px_to_mm(cfg.mm_to_px(10.0)), 10.0, places=0)


class TestBinarize(unittest.TestCase):

    def setUp(self):
        self.cfg = make_config()
        self.page = synthetic_page(self.cfg)

    def test_output_is_binary(self):
        result = binarize(self.page, self.cfg)
        unique = set(np.unique(result))
        self.assertTrue(unique.issubset({0, 255}))

    def test_ink_detected(self):
        result = binarize(self.page, self.cfg)
        ink_pixels = np.sum(result > 0)
        self.assertGreater(ink_pixels, 100)

    def test_grey_lines_removed(self):
        h, w = (self.cfg.mm_to_px(297), self.cfg.mm_to_px(210))
        grey_only = np.ones((h, w), dtype=np.uint8) * 240
        cv2.line(grey_only, (0, 100), (w, 100), 160, 2)
        result = binarize(grey_only, self.cfg)
        ink_pixels = np.sum(result > 0)
        self.assertLess(ink_pixels, 50)

    def test_ink_survives_binarization(self):
        h, w = (self.cfg.mm_to_px(297), self.cfg.mm_to_px(210))
        ink_only = np.ones((h, w), dtype=np.uint8) * 240
        cv2.rectangle(ink_only, (50, 50), (150, 100), 30, -1)
        result = binarize(ink_only, self.cfg)
        ink_pixels = np.sum(result > 0)
        self.assertGreater(ink_pixels, 100)


class TestLineDetection(unittest.TestCase):

    def setUp(self):
        self.cfg  = make_config()
        self.page = synthetic_page(self.cfg)
        self.binary = binarize(self.page, self.cfg)

    def test_correct_number_of_pairs(self):
        pairs = detect_line_pairs(self.binary, self.cfg)
        self.assertEqual(len(pairs), self.cfg.n_pairs)

    def test_pairs_in_order(self):
        pairs = detect_line_pairs(self.binary, self.cfg)
        baselines = [p.baseline_y for p in pairs]
        self.assertEqual(baselines, sorted(baselines))

    def test_baseline_below_top_line(self):
        pairs = detect_line_pairs(self.binary, self.cfg)
        for pair in pairs:
            self.assertGreater(pair.baseline_y, pair.top_y)

    def test_pair_gap_approximately_correct(self):
        pairs = detect_line_pairs(self.binary, self.cfg)
        expected_gap_px = self.cfg.mm_to_px(self.cfg.pair_gap_mm)
        tol = self.cfg.mm_to_px(self.cfg.position_tol_mm)
        for pair in pairs:
            actual_gap = pair.baseline_y - pair.top_y
            self.assertAlmostEqual(actual_gap, expected_gap_px, delta=tol)

    def test_strip_encompasses_writing_area(self):
        pairs = detect_line_pairs(self.binary, self.cfg)
        for pair in pairs:
            self.assertLessEqual(pair.strip_top,    pair.top_y)
            self.assertGreaterEqual(pair.strip_bottom, pair.baseline_y)


class TestWordSegmentation(unittest.TestCase):

    def setUp(self):
        self.cfg    = make_config()
        self.page   = synthetic_page(self.cfg)
        self.binary = binarize(self.page, self.cfg)
        self.pairs  = detect_line_pairs(self.binary, self.cfg)

    def test_words_detected(self):
        words = segment_words(self.binary, self.pairs, self.cfg)
        self.assertGreater(len(words), 0)

    def test_words_have_correct_pair_indices(self):
        words = segment_words(self.binary, self.pairs, self.cfg)
        pair_indices = set(w.pair_index for w in words)
        valid = set(range(self.cfg.n_pairs))
        self.assertTrue(pair_indices.issubset(valid))

    def test_words_ordered_left_to_right_within_pair(self):
        words = segment_words(self.binary, self.pairs, self.cfg)
        for pair_idx in set(w.pair_index for w in words):
            pair_words = [w for w in words if w.pair_index == pair_idx]
            x_starts = [w.x_start for w in pair_words]
            self.assertEqual(x_starts, sorted(x_starts))

    def test_word_crops_are_non_empty(self):
        words = segment_words(self.binary, self.pairs, self.cfg)
        for w in words:
            self.assertGreater(w.image.size, 0)


class TestStrokeExtraction(unittest.TestCase):

    def setUp(self):
        self.cfg = make_config()

    def _make_line_crop(self):
        h, w = 30, 80
        crop = np.zeros((h, w), dtype=np.uint8)
        cv2.line(crop, (5, 15), (75, 15), 255, 2)
        return crop

    def test_strokes_produced(self):
        crop = self._make_line_crop()
        strokes = extract_strokes(crop, baseline_y=20, strip_top=5, config=self.cfg)
        self.assertGreater(len(strokes), 0)

    def test_strokes_are_stroke_points(self):
        crop = self._make_line_crop()
        strokes = extract_strokes(crop, baseline_y=20, strip_top=5, config=self.cfg)
        for s in strokes:
            self.assertIsInstance(s, StrokePoint)

    def test_pen_up_values_valid(self):
        crop = self._make_line_crop()
        strokes = extract_strokes(crop, baseline_y=20, strip_top=5, config=self.cfg)
        for s in strokes:
            self.assertIn(s.pen_up, [0, 1])

    def test_baseline_normalized(self):
        crop = self._make_line_crop()
        strokes = extract_strokes(crop, baseline_y=20, strip_top=5, config=self.cfg)
        y_values = [s.y for s in strokes if s.pen_up == 0]
        if y_values:
            median_y = float(np.median(y_values))
            self.assertAlmostEqual(median_y, 0.0, delta=2.0)

    def test_empty_crop_returns_empty(self):
        empty = np.zeros((20, 50), dtype=np.uint8)
        strokes = extract_strokes(empty, baseline_y=15, strip_top=5, config=self.cfg)
        self.assertEqual(strokes, [])

    def test_smooth_curve_produces_smooth_path(self):
        """
        Regression test for the zigzag bug: tracing used to pick neighbours
        in fixed scan order rather than by direction of travel, producing
        a back-and-forth jittery path on what should be a smooth stroke.
        This checks that path length stays close to the geometric extent
        of the curve, not inflated by zigzagging.
        """
        crop = np.zeros((60, 150), dtype=np.uint8)
        pts = []
        for t in np.linspace(0, 1, 100):
            x = int(10 + t * 120)
            y = int(30 + 15 * np.sin(t * 3.14))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(crop, pts[i], pts[i + 1], 255, 2)

        strokes = extract_strokes(crop, baseline_y=45, strip_top=5, config=self.cfg)
        self.assertGreater(len(strokes), 10)

        xs = [s.x for s in strokes]
        ys = [s.y for s in strokes]
        width  = max(xs) - min(xs)
        height = max(ys) - min(ys)
        diag   = (width ** 2 + height ** 2) ** 0.5

        n = len(strokes)
        path_len = sum(
            ((strokes[i].x - strokes[i - 1].x) ** 2
             + (strokes[i].y - strokes[i - 1].y) ** 2) ** 0.5
            for i in range(1, n)
        )

        # A zigzagging path inflates path_len far beyond the geometric
        # extent (diag). A clean trace keeps diag/path_len well above 0.7;
        # the pre-fix scan-order trace produced ratios around 0.2-0.4.
        straightness = diag / path_len if path_len > 0 else 0
        self.assertGreater(straightness, 0.7,
            f"Path looks zigzaggy (straightness={straightness:.2f}) — "
            f"stroke ordering may have regressed to fixed scan-order traversal")

    def test_skeleton_to_paths_stops_at_junctions(self):
        """
        _skeleton_to_paths must stop exactly at every junction pixel and
        return exactly one fragment per edge in the skeleton graph.
        A 3-way junction (stem + 2 arms) must produce exactly 3 fragments,
        all sharing the junction pixel as an endpoint.
        """
        h, w = 20, 20
        skel = np.zeros((h, w), dtype=np.uint8)
        stem  = [(x, 5) for x in range(0, 12)]   # includes junction at (11,5)
        arm_a = [(11, 5), (12, 4), (13, 3), (14, 2), (15, 1)]
        arm_b = [(11, 5), (12, 6), (13, 6), (14, 6), (15, 6)]
        for (x, y) in stem + arm_a + arm_b:
            skel[y, x] = 255

        paths = _skeleton_to_paths(skel)
        self.assertEqual(len(paths), 3,
            f"3-way junction must produce exactly 3 fragments, got {len(paths)}")
        junction = (11, 5)
        for p in paths:
            self.assertTrue(p[0] == junction or p[-1] == junction,
                f"every fragment at a 3-way junction must share the junction "
                f"pixel {junction}; got endpoints {p[0]} and {p[-1]}")

    def test_order_paths_no_false_joins(self):
        """
        _order_paths must never emit a pen-DOWN step larger than one
        diagonal pixel (~1.41px) within the skeleton graph.  Larger
        pen-down steps are structurally impossible in the exact-graph CPP
        approach because pen-down connections only occur along real skeleton
        edges; the only large steps in the output are pen-UP transitions
        (which are correct — they're marked with pen_up=True on the
        departure point).
        """
        from handwriting_system.p1_scan_pipeline.config import ScanConfig
        # Two disjoint horizontal fragments separated by a gap
        frag_a = [(x, 5) for x in range(0, 8)]
        frag_b = [(x, 5) for x in range(15, 23)]
        ordered = _order_paths([frag_a, frag_b], ScanConfig())

        for i in range(1, len(ordered)):
            pt_prev, up_prev = ordered[i-1]
            pt_cur,  _       = ordered[i]
            d_px = ((pt_prev[0]-pt_cur[0])**2 + (pt_prev[1]-pt_cur[1])**2)**0.5
            if not up_prev:
                self.assertLessEqual(d_px, 1.5,
                    f"pen-DOWN step from {pt_prev} to {pt_cur} is {d_px:.2f}px — "
                    f"larger than one diagonal pixel means a false join")

    def test_smooth_strokes_breaks_anomalous_internal_jump(self):
        """
        _smooth_strokes should convert a large anomalous internal jump
        within a segment (tracer artifact: wrong branch arm -> dead end
        -> _order_paths jumps back across the word) into a pen-up break,
        rather than leaving it as a visible straight line drawn within a
        single pen-down segment.
        """
        seg = [StrokePoint(x=i * 0.1, y=0, pen_up=0) for i in range(5)]
        seg.append(StrokePoint(x=seg[-1].x + 5.0, y=0, pen_up=0))  # anomalous 5mm jump
        seg += [StrokePoint(x=seg[-1].x + i * 0.1, y=0, pen_up=0) for i in range(1, 5)]
        seg[-1] = StrokePoint(x=seg[-1].x, y=seg[-1].y, pen_up=1)

        smoothed = _smooth_strokes(seg)

        self.assertEqual(len(smoothed), len(seg),
            "Smoothener should only mark pen_up, never drop points")
        pen_up_count = sum(1 for p in smoothed if p.pen_up)
        self.assertGreaterEqual(pen_up_count, 2,
            "Anomalous jump should produce an extra pen-up break, on "
            "top of the segment's natural end-of-segment pen-up")


    def test_smooth_strokes_floor_does_not_suppress_submm_anomalies(self):
        """
        The 0.15mm absolute floor in _smooth_strokes should NOT protect
        sub-1mm jumps that are still clearly anomalous relative to their
        segment's local median step size.

        Real-data root cause: fine cursive segments have median step
        ~0.04-0.06mm; a 0.98mm internal jump is ~23x that median — a
        genuine artifact — but was passing the old 1.0mm absolute floor
        silently. This test confirms the new 0.15mm floor doesn't
        interfere with the 8x-median relative detector catching such cases.
        """
        # Simulate a segment with tiny step sizes (~0.05mm each), containing
        # a sub-1mm anomalous jump (0.7mm — well above 8x the 0.05mm median,
        # but below the old 1.0mm floor that was suppressing detection).
        seg = [StrokePoint(x=i * 0.05, y=0, pen_up=0) for i in range(10)]
        seg.append(StrokePoint(x=seg[-1].x + 0.7, y=0, pen_up=0))  # 0.7mm jump (~14x median)
        seg += [StrokePoint(x=seg[-1].x + i * 0.05, y=0, pen_up=0) for i in range(1, 6)]
        seg[-1] = StrokePoint(x=seg[-1].x, y=seg[-1].y, pen_up=1)

        smoothed = _smooth_strokes(seg)

        pen_ups = sum(1 for p in smoothed if p.pen_up)
        self.assertGreaterEqual(pen_ups, 2,
            "A sub-1mm jump that is 14x the segment's local median step "
            "should still be flagged as anomalous — the absolute floor "
            "must not suppress the relative (8x-median) detector")

    def test_smooth_positions_reduces_wobble_without_dropping_points(self):
        """
        _smooth_positions should:
        1. Return the same number of points (no drops).
        2. Measurably reduce perpendicular deviation along a noisy line.
        3. Leave endpoints of each segment untouched (pen_up positions are
           letter-alignment-critical and must not shift).
        """
        import math
        # Build a straight-ish line with realistic sub-pixel jitter (+/-0.03mm)
        rng = np.random.default_rng(42)
        n = 40
        pts = [
            StrokePoint(
                x=i * 0.05,
                y=rng.uniform(-0.03, 0.03),
                pen_up=(1 if i == n-1 else 0)
            )
            for i in range(n)
        ]

        smoothed = _smooth_positions(pts, radius=3, weight=0.6)

        self.assertEqual(len(smoothed), len(pts),
            "_smooth_positions must not drop or add points")

        # Endpoint (pen_up) must be untouched
        self.assertAlmostEqual(smoothed[-1].x, pts[-1].x, places=6)
        self.assertAlmostEqual(smoothed[-1].y, pts[-1].y, places=6)
        self.assertEqual(smoothed[-1].pen_up, 1)

        # Wobble (y-deviation from y=0 for a nominally-horizontal path)
        # should be reduced for interior points
        before_rms = math.sqrt(sum(p.y**2 for p in pts[3:-3]) / (n-6))
        after_rms  = math.sqrt(sum(p.y**2 for p in smoothed[3:-3]) / (n-6))
        self.assertLess(after_rms, before_rms,
            "_smooth_positions should reduce y-deviation on a noisy horizontal segment")

    def test_smooth_positions_skips_short_segments(self):
        """Short segments (< 2*radius+1 points) should pass through unchanged."""
        pts = [StrokePoint(x=i * 0.1, y=float(i % 2) * 0.05, pen_up=(1 if i == 3 else 0))
               for i in range(4)]   # 4 points < 7 (min for radius=3)
        smoothed = _smooth_positions(pts, radius=3, weight=0.6)
        for a, b in zip(pts, smoothed):
            self.assertAlmostEqual(a.x, b.x, places=6)
            self.assertAlmostEqual(a.y, b.y, places=6)

    def test_morphological_closing_does_not_break_clean_stroke(self):
        """
        The kernel=2 closing added before skeletonize should not
        introduce any new holes or gaps in a stroke that was already
        clean — the skeleton of the closed mask should have no more
        endpoints than the original (closing a solid stroke may slightly
        extend its ends, but must never fragment it or introduce gaps).
        """
        from skimage.morphology import skeletonize as sk_thin

        def count_endpoints(skel):
            ys, xs = np.where(skel > 0)
            pixel_set = set(zip(xs.tolist(), ys.tolist()))
            return sum(
                1 for x, y in pixel_set
                if sum(1 for dx in (-1,0,1) for dy in (-1,0,1)
                       if not (dx==0 and dy==0) and (x+dx,y+dy) in pixel_set) == 1
            )

        # Clean solid horizontal bar — no gaps
        mask = np.zeros((30, 80), dtype=np.uint8)
        mask[10:20, 10:70] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        skel_orig   = sk_thin(mask > 0).astype(np.uint8) * 255
        skel_closed = sk_thin(closed > 0).astype(np.uint8) * 255

        ep_orig   = count_endpoints(skel_orig)
        ep_closed = count_endpoints(skel_closed)

        self.assertLessEqual(ep_closed, ep_orig,
            "Closing a gap-free stroke should never increase skeleton endpoint "
            f"count (before={ep_orig}, after={ep_closed})"
        )

    def test_order_paths_cpp_no_false_pen_down_joins(self):
        """
        The CPP-based _order_paths must never emit a pen-DOWN step larger
        than a single diagonal skeleton pixel (~1.41px), because pen-down
        connections only occur along real skeleton edges (fragments), never
        across empty space.  Inter-component jumps are always pen-up.

        Two disjoint vertical fragments 8px apart (~0.34mm at 23.62px/mm):
        the CPP correctly identifies them as separate connected components
        and connects them pen-UP, not pen-DOWN.
        """
        from handwriting_system.p1_scan_pipeline.config import ScanConfig
        frag_a = [(5, y) for y in range(5, 16)]    # vertical, left
        frag_b = [(13, y) for y in range(5, 16)]   # vertical, right, 8px gap
        ordered = _order_paths([frag_a, frag_b], ScanConfig())

        for i in range(1, len(ordered)):
            pt_prev, up_prev = ordered[i-1]
            pt_cur,  _       = ordered[i]
            d_px = ((pt_prev[0]-pt_cur[0])**2 + (pt_prev[1]-pt_cur[1])**2)**0.5
            if not up_prev:
                self.assertLessEqual(d_px, 1.5,
                    f"pen-DOWN step {pt_prev}→{pt_cur} is {d_px:.1f}px: "
                    f"CPP should never pen-DOWN across empty space")

        # Must also confirm a pen-up was produced somewhere
        self.assertTrue(any(up for _, up in ordered),
            "disjoint components must produce at least one pen-up")

    def test_reconnect_compatible_merges_direction_aligned_gap(self):
        """
        _reconnect_compatible should reconnect a small pen-up gap when both
        sides of the gap are going in roughly the same direction, even if
        there is a singleton bridge segment between them (as created by
        _smooth_strokes from _order_paths pen-up points).
        """
        # Simulate: seg_A going right, singleton bridge, seg_B going right.
        # This matches the 'ou' continuation pattern seen in 'you'.
        seg_a = [StrokePoint(x=float(i)*0.05, y=0.0, pen_up=(1 if i==9 else 0))
                 for i in range(10)]                          # 10 pts, pen_up on last
        singleton = [StrokePoint(x=0.52, y=0.01, pen_up=1)] # bridge singleton
        seg_b = [StrokePoint(x=0.56 + i*0.05, y=0.01, pen_up=(1 if i==9 else 0))
                 for i in range(10)]                          # 10 pts going right
        strokes = seg_a + singleton + seg_b

        result = _reconnect_compatible(strokes, max_gap_mm=1.0, min_dot=0.6)
        pen_ups_after = sum(1 for p in result if p.pen_up)
        # Should have merged the gap: fewer pen-ups than original 3
        self.assertLess(pen_ups_after, 3,
            "_reconnect_compatible should merge a direction-compatible "
            "gap (dot ≈ 1.0) through a singleton bridge segment")
        self.assertEqual(len(result), len(strokes),
            "_reconnect_compatible must not drop any points")

    def test_reconnect_compatible_preserves_direction_incompatible_gap(self):
        """
        _reconnect_compatible must NOT reconnect a gap where the outgoing
        and incoming directions are very different — even if the gap is small.
        This protects genuine pen lifts between strokes going in different
        directions (e.g. the 'y' descender junction).
        """
        # Seg A going right (+x), seg B going up (+y) — 90° apart, dot=0 < min_dot
        seg_a = [StrokePoint(x=float(i)*0.05, y=0.0, pen_up=(1 if i==9 else 0))
                 for i in range(10)]
        seg_b = [StrokePoint(x=0.5, y=float(i)*0.05, pen_up=(1 if i==9 else 0))
                 for i in range(10)]
        strokes = seg_a + seg_b

        result = _reconnect_compatible(strokes, max_gap_mm=1.0, min_dot=0.6)
        pen_ups_after = sum(1 for p in result if p.pen_up)
        self.assertEqual(pen_ups_after, 2,
            "_reconnect_compatible must not reconnect a gap where the "
            "outgoing direction is perpendicular to the incoming direction "
            "(dot=0, below min_dot=0.6)")


class TestHybridExtraction(unittest.TestCase):
    """
    extract_strokes_hybrid (contour-clean + medial axis) and
    extract_strokes_medial_axis_raw (medial axis on the raw mask,
    skipping contour cleaning) — added to compare whether the
    contour-cleaning preprocessing step is actually responsible for the
    extra branch points seen vs plain Zhang-Suen skeletonize, or
    whether that's inherent to medial_axis itself.
    """

    def _make_crop(self):
        crop = np.zeros((60, 100), dtype=np.uint8)
        cv2.ellipse(crop, (50, 30), (30, 15), 0, 0, 360, 255, 5)
        return crop

    def test_hybrid_runs_and_returns_strokepoints(self):
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            extract_strokes_hybrid,
        )
        config = make_config()
        strokes = extract_strokes_hybrid(self._make_crop(), baseline_y=45, strip_top=0, config=config)
        self.assertGreater(len(strokes), 0)
        self.assertTrue(all(isinstance(p, StrokePoint) for p in strokes))

    def test_medial_axis_raw_runs_and_returns_strokepoints(self):
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            extract_strokes_medial_axis_raw,
        )
        config = make_config()
        strokes = extract_strokes_medial_axis_raw(self._make_crop(), baseline_y=45, strip_top=0, config=config)
        self.assertGreater(len(strokes), 0)
        self.assertTrue(all(isinstance(p, StrokePoint) for p in strokes))

    def test_medial_axis_raw_handles_empty_crop(self):
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            extract_strokes_medial_axis_raw,
        )
        config = make_config()
        empty = np.zeros((60, 100), dtype=np.uint8)
        strokes = extract_strokes_medial_axis_raw(empty, baseline_y=45, strip_top=0, config=config)
        self.assertEqual(strokes, [])

    def test_medial_axis_pruned_runs_and_returns_strokepoints(self):
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            extract_strokes_medial_axis_pruned,
        )
        config = make_config()
        strokes = extract_strokes_medial_axis_pruned(self._make_crop(), baseline_y=45, strip_top=0, config=config)
        self.assertGreater(len(strokes), 0)
        self.assertTrue(all(isinstance(p, StrokePoint) for p in strokes))

    def test_medial_axis_pruned_handles_empty_crop(self):
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            extract_strokes_medial_axis_pruned,
        )
        config = make_config()
        empty = np.zeros((60, 100), dtype=np.uint8)
        strokes = extract_strokes_medial_axis_pruned(empty, baseline_y=45, strip_top=0, config=config)
        self.assertEqual(strokes, [])

    def test_pruning_removes_short_spur_off_a_junction(self):
        """
        Direct unit test of _prune_skeleton_spurs: builds a skeleton with
        a T-junction where one arm is a short spur (4px, well under the
        local-radius threshold) and the other is a long real branch
        (15px). Pruning should remove the spur but leave the long branch
        and the main stem intact.
        """
        from handwriting_system.p1_scan_pipeline.steps.stroke_extractor_hybrid import (
            _prune_skeleton_spurs,
        )
        h, w = 30, 40
        skel = np.zeros((h, w), dtype=np.uint8)

        stem      = [(x, 10) for x in range(0, 11)]   # main stem, horizontal
        long_arm  = [(11, y) for y in range(10, 25)]  # real branch, 15px down
        short_spur = [(11 - i, 10 - i) for i in range(1, 5)]  # 4px spur, diagonal up-left

        for (x, y) in stem + long_arm + short_spur:
            skel[y, x] = 255

        # Distance transform: uniform stroke radius of 3px (typical for a
        # real word-crop stroke width of ~6px). threshold = 1.5*3 = 4.5,
        # so the 4px spur falls under it and gets pruned, while the 15px
        # branch (and the 10px stem) stay well above it.
        distance = np.full((h, w), 3.0, dtype=np.float64)

        pruned = _prune_skeleton_spurs(skel, distance, min_branch_factor=1.5)

        # Junction point + long arm should survive
        self.assertEqual(pruned[10, 10], 255, "Junction point should survive pruning")
        self.assertEqual(pruned[24, 11], 255, "Tip of the long real branch should survive")
        # Spur tip should be gone
        self.assertEqual(pruned[6, 6], 0, "Tip of the short spur should be pruned away")


class TestBBoxComputation(unittest.TestCase):

    def test_empty_strokes(self):
        bbox = compute_bbox_mm([])
        self.assertEqual(bbox, {"width": 0.0, "ascent": 0.0, "descent": 0.0})

    def test_ascent_positive(self):
        strokes = [
            StrokePoint(x=0.0, y=5.0, pen_up=0),
            StrokePoint(x=5.0, y=0.0, pen_up=1),
        ]
        bbox = compute_bbox_mm(strokes)
        self.assertGreater(bbox["ascent"], 0)
        self.assertEqual(bbox["descent"], 0.0)

    def test_descent_positive_for_below_baseline(self):
        strokes = [
            StrokePoint(x=0.0, y=0.0,  pen_up=0),
            StrokePoint(x=5.0, y=-2.0, pen_up=1),
        ]
        bbox = compute_bbox_mm(strokes)
        self.assertGreater(bbox["descent"], 0)

    def test_width_correct(self):
        strokes = [
            StrokePoint(x=0.0, y=0.0, pen_up=0),
            StrokePoint(x=10.0, y=0.0, pen_up=1),
        ]
        bbox = compute_bbox_mm(strokes)
        self.assertAlmostEqual(bbox["width"], 10.0, places=1)


class TestPageStats(unittest.TestCase):

    def test_percentile_dist_no_averages(self):
        values = [1.0, 2.0, 3.0, 100.0]
        dist = _percentile_dist(values)
        self.assertAlmostEqual(dist["p50"], 2.5, places=0)
        self.assertLess(dist["p50"], 10.0)

    def test_iqr_present(self):
        dist = _percentile_dist([1, 2, 3, 4, 5])
        self.assertIn("iqr", dist)
        self.assertGreater(dist["iqr"], 0)

    def test_empty_values(self):
        dist = _percentile_dist([])
        self.assertEqual(dist, {})

    def test_page_stats_returns_counts(self):
        cfg   = make_config()
        page  = synthetic_page(cfg)
        binary = binarize(page, cfg)
        pairs  = detect_line_pairs(binary, cfg)
        words  = segment_words(binary, pairs, cfg)
        stats  = measure_page_stats(pairs, words, cfg)
        self.assertIn("total_words",   stats)
        self.assertIn("clean_words",   stats)
        self.assertIn("flagged_words", stats)


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end test using ScanPipeline on a synthetic image written to disk."""

    def test_pipeline_runs_end_to_end(self):
        import tempfile
        from handwriting_system.p1_scan_pipeline import ScanPipeline

        cfg = make_config()
        page = synthetic_page(cfg)

        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "test_page.png")
            cv2.imwrite(img_path, page)

            pipeline = ScanPipeline(cfg)
            # 10 pairs x 3 words = 30 words on the synthetic page
            word_list = [f"word{i}" for i in range(30)]
            result = pipeline.process(img_path, word_list, output_dir=d)

            self.assertTrue(result.get("success"), result.get("error"))
            self.assertEqual(result["pairs_found"], 10)
            self.assertGreater(result["total_words"], 0)

    def test_pipeline_handles_missing_file(self):
        from handwriting_system.p1_scan_pipeline import ScanPipeline
        pipeline = ScanPipeline(make_config())
        result = pipeline.process("/nonexistent/path.png", None, "/tmp")
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
