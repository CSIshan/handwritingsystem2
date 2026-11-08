"""
Test script: recenter_skeleton() on a real merged component -- checks
whether shifting points toward "more red, less blue" actually improves
centering, using the hybrid+pruned skeleton as the baseline.
"""

import cv2
import numpy as np
from p1_scan_pipeline.steps.binarize import binarize
from p1_scan_pipeline.steps.line_detector import detect_line_pairs
from p1_scan_pipeline.steps.segmenter import segment_words
from p1_scan_pipeline.steps.dpi_estimator import estimate_dpi, DPITooLowError
from p1_scan_pipeline.config import ScanConfig
from p1_scan_pipeline.component_extractor import ComponentExtractor
from p1_scan_pipeline.curve_fitter import (
    HybridExtractor, prune_spurs, skeleton_fidelity, render_fidelity_diff,
    recenter_skeleton,
)

PAGE_PATH = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"
WORD_INDEX_TO_TEST = 8

# --- run the already-working pipeline up through component extraction ---
config = ScanConfig()
grey = cv2.imread(PAGE_PATH, cv2.IMREAD_GRAYSCALE)

measured_dpi = estimate_dpi(grey, config, min_dpi=300.0)
config.dpi = measured_dpi

binary = binarize(grey, config)
pairs = detect_line_pairs(binary, config)
first_pair = [p for p in pairs if p.index == 0]
all_words = segment_words(binary, first_pair, config)

w = all_words[WORD_INDEX_TO_TEST]

extractor = ComponentExtractor(
    min_component_area=5,
    max_merge_height_px=config.mm_to_px(config.pair_gap_mm),
)
components = extractor.extract(w.image, page_offset=(w.x_start, w.top_y))
kept = extractor.filter(components)
merged = extractor.merge_nearby(kept)

target = max(merged, key=lambda c: c.area)
print(f"Testing on component: bbox={target.bbox} area={target.area}\n")

# --- baseline skeleton: hybrid + pruned ---
hybrid_ext = HybridExtractor()  # uses the promoted tuned defaults
skeleton_before = hybrid_ext.extract(target.mask)
skeleton_before = prune_spurs(skeleton_before)

before_fid = skeleton_fidelity(target.mask, skeleton_before)
print(f"Before recentering: IoU={before_fid['iou']:.4f} "
      f"missed={before_fid['missed_pixels']} extra={before_fid['extra_pixels']}")

# --- recenter, single pass ---
skeleton_after = recenter_skeleton(target.mask, skeleton_before)
skeleton_after = prune_spurs(skeleton_after)

after_fid = skeleton_fidelity(target.mask, skeleton_after)
print(f"After recentering:  IoU={after_fid['iou']:.4f} "
      f"missed={after_fid['missed_pixels']} extra={after_fid['extra_pixels']}")

# --- direct skeleton LINE overlay, before vs after ---
def _skeleton_overlay(mask, skeleton, color=(0, 255, 255)):
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas = (canvas * 0.3).astype(np.uint8)
    canvas[skeleton > 0] = color
    return canvas

h, w_ = target.mask.shape
gap = 10

overlay_before = _skeleton_overlay(target.mask, skeleton_before)
overlay_after = _skeleton_overlay(target.mask, skeleton_after)

overlay_comparison = np.zeros((h, w_ * 2 + gap, 3), dtype=np.uint8)
overlay_comparison[:, :w_] = overlay_before
overlay_comparison[:, w_ + gap:] = overlay_after
cv2.imwrite("debug_recenter_overlay.png", overlay_comparison)
print("\nSaved debug_recenter_overlay.png -- left = before, right = after")

# --- fidelity diff, before vs after ---
diff_before = render_fidelity_diff(target.mask, skeleton_before)
diff_after = render_fidelity_diff(target.mask, skeleton_after)

diff_comparison = np.zeros((h, w_ * 2 + gap, 3), dtype=np.uint8)
diff_comparison[:, :w_] = diff_before
diff_comparison[:, w_ + gap:] = diff_after
cv2.imwrite("debug_recenter_fidelity_diff.png", diff_comparison)
print("Saved debug_recenter_fidelity_diff.png -- left = before, right = after "
      "(green=correct, red=missed, blue=extra)")
