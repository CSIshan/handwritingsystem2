"""
Diagnostic: replays binarize()'s internal stages on just word 0's region,
so we can see which stage (ink_mask / line_mask_dilated / clean_ink)
introduces the gap -- without modifying binarize.py itself yet.
"""
import cv2
import numpy as np
from p1_scan_pipeline.config import ScanConfig

PAGE_PATH = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"
config = ScanConfig(dpi=350)  # set to whatever your last measured_dpi was

# word 0's page region, from your printed output: x=(883, 996)
# add a little vertical padding around the line strip so we see full letters
X_START, X_END = 883, 996
Y_START, Y_END = 750, 1250  # rough guess -- widen if letters get cut off

grey = cv2.imread(PAGE_PATH, cv2.IMREAD_GRAYSCALE)
crop = grey[Y_START:Y_END, X_START:X_END]

# --- replay binarize.py's stages manually ---
_, ink_mask = cv2.threshold(crop, config.ink_threshold, 255, cv2.THRESH_BINARY_INV)
_, combined_mask = cv2.threshold(crop, config.line_threshold, 255, cv2.THRESH_BINARY_INV)
line_mask = cv2.subtract(combined_mask, ink_mask)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
line_mask_dilated = cv2.dilate(line_mask, kernel, iterations=1)

clean_ink = cv2.subtract(ink_mask, line_mask_dilated)

denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
clean_ink_denoised = cv2.morphologyEx(clean_ink, cv2.MORPH_OPEN, denoise_kernel)

# --- save each stage so we can compare side by side ---
cv2.imwrite("diag_0_ink_mask.png", ink_mask)
cv2.imwrite("diag_1_line_mask_dilated.png", line_mask_dilated)
cv2.imwrite("diag_2_clean_ink_before_denoise.png", clean_ink)
cv2.imwrite("diag_3_clean_ink_after_denoise.png", clean_ink_denoised)

print("Saved diag_0 through diag_3. Compare them:")
print("  diag_0 -> raw ink before any line removal (should be UNBROKEN)")
print("  diag_1 -> what got dilated/removed as 'line'")
print("  diag_2 -> ink after line subtraction (gap should appear HERE if line-removal is the cause)")
print("  diag_3 -> after denoise open (gap could ALSO appear/worsen here)")
