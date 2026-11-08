"""
TUNE THE CORRECTION PIPELINE
=============================
This is a "turn one knob, look at the picture" tool. You don't need to
read or understand the code in curve_fitter.py to use it.

HOW TO USE IT:
  1. Pick which pipeline to test in Section 2 (PIPELINE_TO_TEST).
  2. Change ONE number in that pipeline's settings block.
  3. Run this file:  python3 tune_correction.py
  4. Open the saved image: tuning_result.png
  5. Look at the numbers printed in the terminal (IoU, red pixels, blue pixels).
  6. Decide: did that change help or hurt? Keep it or put the number back.
  7. Repeat with the next number.

That loop -- change one number, run, look, keep-or-revert -- is the whole
job. There's no shortcut to "the perfect values" other than doing this
patiently on real handwriting.

WHAT THE PICTURE MEANS:
  Top row:    the raw ink, and the skeleton line BEFORE correction.
  Bottom row: the skeleton line AFTER correction (with your settings),
              and a colour-coded "report card" of how well it fits:
                GREEN = correctly traced ink
                RED   = ink that the line MISSED (bad -- want less of this)
                BLUE  = line claims there's ink but there isn't (bad -- want less of this)

WHAT THE NUMBERS MEAN:
  IoU (Intersection over Union): 1.0 = perfect fit, lower = worse fit.
    This is the single number to watch. Higher after your change = good.
  missed_pixels: how much real ink the line failed to cover (= red area).
  extra_pixels:  how much the line claims that isn't real ink (= blue area).

TWO PIPELINES IN THIS FILE -- WHICH ONE AM I TESTING?
  "recenter"   -> the skew_left_red_first / add_centerlines_in_missing_red /
                  recenter_skeleton family. This one already has values
                  found by an earlier numeric search (the odd long-decimal
                  numbers like 73.23055871712805). You're not starting from
                  scratch here -- you're double-checking a search result by
                  eye on your own real handwriting, which the search may
                  never have seen.
  "correction" -> the correct_skeleton_from_fidelity family. This one is
                  genuinely NOT tuned yet -- plain round numbers, nobody's
                  searched over these. This is the one from your original
                  to-do list (section 20, steps 1-2).
"""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import curve_fitter as cf


# =============================================================================
# SECTION 1 -- WHAT TO TEST ON
# =============================================================================
# Point this at a cropped image of ONE word or ONE letter from your own
# pipeline (e.g. something you saved out of component_extractor.py or
# curve_fitter.py while debugging). Ink should be the dark/foreground part.
#
# If this file doesn't exist, a made-up squiggle is used instead, purely so
# you can try the tool immediately. Swap in a real crop as soon as you have
# one -- synthetic shapes don't tell you anything reliable about real
# handwriting.
INPUT_IMAGE = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"


# =============================================================================
# SECTION 2 -- WHICH PIPELINE TO TEST RIGHT NOW
# =============================================================================
PIPELINE_TO_TEST = "recenter"   # "recenter"  or  "correction"

# =============================================================================
# SECTION 3a -- "correction" PIPELINE KNOBS  (only matters if PIPELINE_TO_TEST
#               = "correction" above)
# =============================================================================
# Change ONE of these per run. Every value below is the CURRENT default --
# i.e. running the tool with no changes reproduces today's behaviour exactly.
# Comments explain what each number physically does, in plain English.

# --- correct_skeleton_from_fidelity: how many correction rounds to run, and
#     when to stop ---
MAX_ITERATIONS       = 3      # how many correction passes to attempt
MIN_IOU_IMPROVEMENT  = 0.001  # stop early if a pass improves fit by less than this
MAX_SPUR_LENGTH       = 6      # tiny dangling branches shorter than this (pixels) get trimmed off
SHIFT_MAX_SHIFT_FACTOR = 0.5   # how far a point can move in one pass, as a fraction of pen half-width
                                # (0.5 = "half a pen-width per pass"). Higher = faster but riskier moves.

# --- _shift_segments_from_normal_rays: the "measure the ink on both sides
#     and re-centre the line" step ---
RAY_LIMIT_FLOOR       = 2.0   # minimum distance (pixels) a measuring ray will travel
RAY_LIMIT_FACTOR      = 3.0   # ray travels up to (pen half-width x this) pixels
TANGENT_WINDOW        = 3     # how many pixels ahead/behind used to sense the line's direction
                                # bigger = smoother direction estimate, less locally sensitive
MIN_SEGMENT_LENGTH    = 3     # segments shorter than this (pixels) are left alone, not re-centred
PROFILE_SMOOTHING_RADIUS = 3  # smooths the measurements along the line before applying them
RAY_STEP              = 0.5   # how finely each ray is sampled (smaller = more precise, slower)

# --- _insert_missing_centerlines: the "there's a whole chunk of ink with no
#     line through it at all -- add one" step ---
AREA_FLOOR            = 4.0   # smallest red patch (pixels) worth adding a new line for
AREA_DIVISOR          = 2.0   # used with AREA_FLOOR to size that minimum patch
ALIGNMENT_FACTOR      = 1.0   # red closer than (pen half-width x this) to an existing line
                                # is treated as "just needs shifting", not "needs a new line"
INTERIOR_MARGIN_FLOOR = 1.0   # new lines must stay at least this many pixels inside the ink
INTERIOR_MARGIN_FACTOR = 0.5  # ...or (pen half-width x this), whichever is bigger
INSERT_DILATION_ITERATIONS = 1  # how much a red patch is "grown" before searching it for a line

# --- bridge_missing_strokes: the "these two loose ends look like they
#     should connect -- draw a bridge" step (NOT yet validated on real data --
#     tune this one carefully and check the picture closely) ---
BRIDGE_MAX_GAP_FACTOR   = 4.0   # two loose ends farther apart than (pen half-width x this)
                                  # are never considered for bridging at all
BRIDGE_TANGENT_LOOKBACK = 5     # how far back along each loose end we look to sense its direction
BRIDGE_NUM_POINTS       = 24    # how smooth the drawn bridge curve is (cosmetic only)
BRIDGE_MIN_INK_FRACTION = 0.8   # the bridge must land on real ink at least this fraction of the time
BRIDGE_MIN_RED_REDUCTION = 3    # a bridge must remove at least this many "missed ink" pixels to count
BRIDGE_MAX_TURN_DEG     = 70.0  # reject a bridge if it would require a sharper turn than this (degrees)
BRIDGE_MIN_SPUR_SEGMENT_LENGTH = 6  # loose ends belonging to a segment shorter than this are ignored
                                      # (too short to trust as a real stroke end, not just noise)

RUN_BRIDGING = False   # set True to also test bridge_missing_strokes on top of the correction pass


# =============================================================================
# SECTION 3b -- "recenter" PIPELINE KNOBS  (only matters if PIPELINE_TO_TEST
#               = "recenter" above)
# =============================================================================
# These start at the values an earlier numeric search landed on. Change ONE,
# run, look at the picture and IoU, keep-or-revert. You're checking whether a
# search result generalizes to YOUR real handwriting, not starting blind.

# --- HybridExtractor: how the starting centerline is built before recentering ---
HYBRID_WIDTH_PERCENTILE   = 73.23055871712805  # ink-width percentile above which
                                                 # medial-axis (not Zhang-Suen) is trusted --
                                                 # i.e. "how wide does a spot have to be
                                                 # before we treat it as a junction/loop"
HYBRID_DILATION_ITERATIONS = 1                  # how much that "wide spot" region is grown
                                                 # before blending the two skeleton types

# --- shared spur trimming, used before AND after recentering ---
RECENTER_MAX_SPUR_LENGTH = 2   # tiny dangling branches shorter than this (pixels) get trimmed off
                                # (this pipeline defaults to 2, stricter than the other pipeline's 6)

# --- skew_left_red_first: the "nudge the line toward nearby missed ink" step ---
MAX_SHIFT_RATIO       = 0.31080310858995536  # how far a point can move, as a fraction
                                               # of pen half-width. Higher = bolder nudges.
TANGENT_LOOKBACK      = 3      # how many pixels back are used to sense the line's local direction
RAY_LIMIT_RATIO       = 1.9301960287900635  # how far (x pen half-width) the "is there
                                              # missed ink nearby" probe looks
RED_PROFILE_RADIUS    = 2      # smooths the raw "is there missed ink here" signal along the line
DISPLACEMENT_PROFILE_RADIUS = 1  # smooths the actual movement amount along the line

# --- add_centerlines_in_missing_red: the "there's a whole patch with no line
#     at all -- add one" step ---
MIN_RED_REGION_AREA_FACTOR = 1.270458780815832  # smallest missed-ink patch (as a
                                                  # multiple of pen half-width) worth
                                                  # adding a brand new line for
OUTLINE_RED_DILATION_EXTRA = 0   # extra growth applied when tracing a patch too thin
                                  # for a full new line (uses the ink's own outline instead)
EXISTING_SKELETON_MOAT     = 1   # minimum gap (pixels) kept between a new line and the
                                  # existing one, so they don't accidentally touch/merge
ALLOWED_NEW_RED_PIXELS     = 0   # a newly added line must actually help -- this is how much
                                  # "no real improvement" is tolerated before removing it again


# =============================================================================
# SECTION 3 -- everything below this line just runs the test. No need to edit.
# =============================================================================

def _load_or_make_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Assume dark ink on light background, like a real scan crop.
        _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        print(f"Loaded real image: {path}  ({mask.shape[1]}x{mask.shape[0]}px)")
        return mask

    print(f"Could not find '{path}' -- using a made-up squiggle instead.")
    print("(Swap INPUT_IMAGE at the top of this file for a real cropped word/letter.)")
    mask = np.zeros((140, 220), dtype=np.uint8)
    pts = np.array([[20, 100], [50, 30], [80, 100], [110, 30], [140, 100],
                     [165, 70], [190, 100]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, 9)
    cv2.circle(mask, (65, 65), 20, 255, 6)   # a loop, like the top of a cursive 'l' or 'e'
    return mask


def main():
    mask = _load_or_make_mask(INPUT_IMAGE)

    if PIPELINE_TO_TEST == "recenter":
        params = cf.RecenterParams(
            hybrid_width_percentile=HYBRID_WIDTH_PERCENTILE,
            hybrid_dilation_iterations=HYBRID_DILATION_ITERATIONS,
            max_spur_length=RECENTER_MAX_SPUR_LENGTH,
            max_shift_ratio=MAX_SHIFT_RATIO,
            tangent_lookback=TANGENT_LOOKBACK,
            ray_limit_ratio=RAY_LIMIT_RATIO,
            red_profile_radius=RED_PROFILE_RADIUS,
            displacement_profile_radius=DISPLACEMENT_PROFILE_RADIUS,
            min_red_region_area_factor=MIN_RED_REGION_AREA_FACTOR,
            outline_red_dilation_extra=OUTLINE_RED_DILATION_EXTRA,
            existing_skeleton_moat=EXISTING_SKELETON_MOAT,
            allowed_new_red_pixels=ALLOWED_NEW_RED_PIXELS,
        )

        # Baseline: HybridExtractor + prune, same recipe your real test2_line.py uses.
        hybrid_ext = cf.HybridExtractor(
            width_percentile=params.hybrid_width_percentile,
            dilation_iterations=params.hybrid_dilation_iterations,
        )
        skeleton_before = hybrid_ext.extract(mask)
        skeleton_before = cf.prune_spurs(skeleton_before, max_spur_length=params.max_spur_length)

        skeleton_after = cf.recenter_skeleton(mask, skeleton_before, params=params)
        skeleton_after = cf.prune_spurs(skeleton_after, max_spur_length=params.max_spur_length)

    elif PIPELINE_TO_TEST == "correction":
        # Baseline: plain Zhang-Suen, the other pipeline's usual starting point.
        skeleton_before = cf.ZhangSuenExtractor().extract(mask)
        skeleton_before = cf.prune_spurs(skeleton_before)

        skeleton_after = cf.correct_skeleton_from_fidelity(
            mask, skeleton_before,
            max_iterations=MAX_ITERATIONS,
            min_iou_improvement=MIN_IOU_IMPROVEMENT,
            max_spur_length=MAX_SPUR_LENGTH,
            shift_max_shift_factor=SHIFT_MAX_SHIFT_FACTOR,
        )
    else:
        raise ValueError(f"PIPELINE_TO_TEST must be 'recenter' or 'correction', got {PIPELINE_TO_TEST!r}")

    fid_before = cf.skeleton_fidelity(mask, skeleton_before)

    if RUN_BRIDGING:
        skeleton_after = cf.bridge_missing_strokes(
            mask, skeleton_after,
            max_gap_factor=BRIDGE_MAX_GAP_FACTOR,
            tangent_lookback=BRIDGE_TANGENT_LOOKBACK,
            bridge_num_points=BRIDGE_NUM_POINTS,
            min_ink_fraction=BRIDGE_MIN_INK_FRACTION,
            min_red_reduction=BRIDGE_MIN_RED_REDUCTION,
            max_turn_deg=BRIDGE_MAX_TURN_DEG,
            min_spur_segment_length=BRIDGE_MIN_SPUR_SEGMENT_LENGTH,
        )
        skeleton_after = cf.finalize_topology_smoothing(skeleton_after)

    fid_after = cf.skeleton_fidelity(mask, skeleton_after)

    if PIPELINE_TO_TEST == "correction":
        print()
        print("NOTE: RAY_*, TANGENT_WINDOW, PROFILE_SMOOTHING_RADIUS, AREA_*, etc.")
        print("      are one level deeper than correct_skeleton_from_fidelity's own")
        print("      arguments -- ask me and I'll wire a direct path to them too.")
        print()

    print(f"Pipeline tested: {PIPELINE_TO_TEST}")

    # --- report ---
    print("=" * 60)
    print(f"{'':20s} {'BEFORE':>15s} {'AFTER':>15s}")
    print(f"{'IoU (higher=better)':20s} {fid_before['iou']:>15.4f} {fid_after['iou']:>15.4f}")
    print(f"{'missed (red) px':20s} {fid_before['missed_pixels']:>15d} {fid_after['missed_pixels']:>15d}")
    print(f"{'extra (blue) px':20s} {fid_before['extra_pixels']:>15d} {fid_after['extra_pixels']:>15d}")
    print("=" * 60)

    delta = fid_after["iou"] - fid_before["iou"]
    if delta > 0.0005:
        print(f"RESULT: This setting IMPROVED fit (IoU +{delta:.4f}).")
    elif delta < -0.0005:
        print(f"RESULT: This setting HURT fit (IoU {delta:.4f}). Consider reverting.")
    else:
        print("RESULT: No meaningful change in fit from this setting.")

    # --- picture ---
    diff_before = cf.render_fidelity_diff(mask, skeleton_before)
    diff_after = cf.render_fidelity_diff(mask, skeleton_after)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(mask, cmap="gray")
    axes[0, 0].set_title("Original ink")
    axes[0, 0].axis("off")

    overlay_before = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay_before = (overlay_before * 0.3).astype(np.uint8)
    overlay_before[skeleton_before > 0] = (255, 255, 0)
    axes[0, 1].imshow(overlay_before)
    axes[0, 1].set_title(f"Skeleton BEFORE correction\nIoU={fid_before['iou']:.4f}")
    axes[0, 1].axis("off")

    overlay_after = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay_after = (overlay_after * 0.3).astype(np.uint8)
    overlay_after[skeleton_after > 0] = (255, 255, 0)
    axes[1, 0].imshow(overlay_after)
    axes[1, 0].set_title(f"Skeleton AFTER correction\nIoU={fid_after['iou']:.4f}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(diff_after, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Report card AFTER\ngreen=good, red=missed, blue=extra")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("tuning_result.png", dpi=130)
    print("\nSaved picture: tuning_result.png")


if __name__ == "__main__":
    main()
