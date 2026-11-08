"""
Test script: page 1, first line (pair index 0) only.

Pipeline: raw scan -> estimate DPI -> binarize -> detect_line_pairs
          -> segment_words -> (filter to pair 0) -> component_extractor
          -> merge_nearby -> visualize
"""

import cv2
from tools.visualizer import visualize
from p1_scan_pipeline.steps.binarize import binarize
from p1_scan_pipeline.steps.line_detector import detect_line_pairs
from p1_scan_pipeline.steps.segmenter import segment_words
from p1_scan_pipeline.steps.dpi_estimator import estimate_dpi, DPITooLowError
from p1_scan_pipeline.config import ScanConfig
from p1_scan_pipeline.component_extractor import ComponentExtractor

# --- Fill this in ---
PAGE_PATH = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"

config = ScanConfig()

# --- Step 1: load raw page (grayscale, since binarize expects grey) ---
grey = cv2.imread(PAGE_PATH, cv2.IMREAD_GRAYSCALE)
if grey is None:
    raise FileNotFoundError(f"Could not load image at {PAGE_PATH}")

# --- Step 2: auto-detect real DPI FIRST (binarize needs correct config.dpi) ---
try:
    measured_dpi = estimate_dpi(grey, config, min_dpi=300.0)
    print(f"Measured DPI: {measured_dpi:.1f} (config default was {config.dpi})")
    config.dpi = measured_dpi
except DPITooLowError as e:
    raise SystemExit(f"Aborting: {e}")

# --- Step 2b: binarize (uses corrected config.dpi for line-row band) ---
binary = binarize(grey, config)

# --- Step 3: detect line pairs, keep only pair 0 (first line) ---
pairs = detect_line_pairs(binary, config)
first_pair = [p for p in pairs if p.index == 0]
print(f"Line pair 0: top_y={first_pair[0].top_y} baseline_y={first_pair[0].baseline_y}")

# --- Step 4: segment words, keep only words from pair 0 ---
all_words = segment_words(binary, first_pair, config)  # only pair 0 passed in
print(f"Words found in line 1: {len(all_words)}")
for w in all_words:
    flag = f" [FLAGGED: {w.flag_reason}]" if w.flagged else ""
    print(f"  word {w.word_index}: x=({w.x_start},{w.x_end}){flag}")

# --- Step 5: run component extraction + merge on each word crop in line 1 ---
extractor = ComponentExtractor(
    min_component_area=5,
    max_merge_height_px=config.mm_to_px(config.pair_gap_mm),
)

for w in all_words:
    page_offset = (w.x_start, w.top_y)  # word's position on the page
    components = extractor.extract(w.image, page_offset=page_offset)
    kept = extractor.filter(components)
    merged = extractor.merge_nearby(components)

    print(f"\nWord {w.word_index}: {len(components)} raw components, "
          f"{len(merged)} after merge")
    for c in components:
        print(f"    raw bbox={c.bbox} area={c.area}")
    for m in merged:
        print(f"    merged bbox={m.bbox} area={m.area}")

    visualize(components, background=w.image,
              save_path=f"debug_line1_word{w.word_index}_all.png", show=False)
    visualize(merged, background=w.image,
              save_path=f"debug_line1_word{w.word_index}_merged.png", show=False)

print("\nDone. Check debug_line1_word*_all.png and debug_line1_word*_merged.png")
