"""
Page Statistics Measurement
============================
Measures page-level handwriting statistics from detected line pairs
and word crops. Uses percentiles and IQR throughout — no averages.
All statistics are stored as distributions, not single values.
"""
import logging
import numpy as np
from typing import List, Dict, Any
from .steps.line_detector import LinePair
from .steps.segmenter import WordCrop

log = logging.getLogger(__name__)


def measure_page_stats(
    pairs:     List[LinePair],
    words:     List[WordCrop],
    config,
    deskew_angle: float = 0.0,
) -> Dict[str, Any]:
    """
    Measure all page-level handwriting statistics.
    Returns dict of distributions (percentiles) not averages.
    """
    stats = {}

    # ── Baseline positions per pair ───────────────────────────────────────────
    baseline_ys = [p.baseline_y for p in pairs]

    # ── Baseline drift per line ───────────────────────────────────────────────
    # Drift = how much the actual baseline deviates from the expected
    # linear spacing. Measures your natural line drift tendency.
    if len(pairs) >= 2:
        expected_spacing = np.mean(np.diff(baseline_ys))
        drifts = []
        for i, pair in enumerate(pairs):
            expected_y = pairs[0].baseline_y + i * expected_spacing
            drift_px   = pair.baseline_y - expected_y
            drift_mm   = config.px_to_mm(abs(drift_px))
            drifts.append(drift_mm)
        stats["baseline_drift_mm"] = _percentile_dist(drifts)
    else:
        stats["baseline_drift_mm"] = None

    # ── Inter-word gaps ───────────────────────────────────────────────────────
    gaps_mm = []
    for pair_idx in set(w.pair_index for w in words):
        pair_words = sorted(
            [w for w in words if w.pair_index == pair_idx and not w.flagged],
            key=lambda w: w.x_start
        )
        for i in range(len(pair_words) - 1):
            gap_px = pair_words[i + 1].x_start - pair_words[i].x_end
            gap_mm = config.px_to_mm(max(0, gap_px))
            if 0.5 < gap_mm < 20.0:   # sanity range
                gaps_mm.append(gap_mm)

    stats["inter_word_gap_mm"] = _percentile_dist(gaps_mm) if gaps_mm else None

    # ── Word heights per line (size scale across page) ────────────────────────
    # Groups by pair, computes median word height per pair
    # Reveals fatigue-related size reduction across the page
    heights_by_pair = {}
    for w in words:
        if w.flagged:
            continue
        h_px = w.baseline_y - w.top_y
        h_mm = config.px_to_mm(h_px)
        if 2.0 < h_mm < 20.0:
            heights_by_pair.setdefault(w.pair_index, []).append(h_mm)

    if heights_by_pair:
        median_heights = {
            pair_idx: float(np.median(hs))
            for pair_idx, hs in heights_by_pair.items()
        }
        # Size scale relative to first line
        first_height = median_heights.get(0, None)
        if first_height and first_height > 0:
            scales = [h / first_height for h in median_heights.values()]
            stats["size_scale_by_line"] = {
                "values": scales,
                "distribution": _percentile_dist(scales),
            }
        else:
            stats["size_scale_by_line"] = None
    else:
        stats["size_scale_by_line"] = None

    # ── Writing slant ─────────────────────────────────────────────────────────
    # Deskew angle gives overall page tilt; slant is per-stroke.
    # We measure the dominant angle of stroke segments from the skeleton.
    # Stored here as the deskew correction applied.
    stats["deskew_angle_degrees"] = deskew_angle

    # ── Word widths distribution ───────────────────────────────────────────────
    widths_mm = [
        config.px_to_mm(w.x_end - w.x_start)
        for w in words if not w.flagged
    ]
    stats["word_width_mm"] = _percentile_dist(widths_mm) if widths_mm else None

    # ── Summary counts ────────────────────────────────────────────────────────
    stats["total_words"]   = len(words)
    stats["flagged_words"] = sum(1 for w in words if w.flagged)
    stats["clean_words"]   = sum(1 for w in words if not w.flagged)

    log.info(
        f"Page stats: {stats['clean_words']} clean words, "
        f"{stats['flagged_words']} flagged"
    )
    return stats


def _percentile_dist(values: List[float]) -> Dict[str, float]:
    """
    Return percentile distribution instead of mean/stddev.
    Robust to outliers — IQR-based spread measure included.
    """
    if not values:
        return {}
    a = np.array(values, dtype=float)
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    return {
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(p25),
        "p50":    float(p50),      # median — use this as central value
        "p75":    float(p75),
        "p90":    float(np.percentile(a, 90)),
        "iqr":    float(p75 - p25),  # spread measure, not stddev
        "n":      len(values),
    }


def is_page_outlier(
    page_stats:   Dict[str, Any],
    corpus_stats: Dict[str, Any],
    config,
) -> tuple[bool, str]:
    """
    Compare this page's stats against the corpus-level stats.
    Uses IQR rule: outlier if > iqr_factor × corpus IQR from corpus median.
    Returns (is_outlier, reason).
    """
    factor = config.outlier_iqr_factor

    checks = [
        ("word_width_mm",   "median word width"),
        ("inter_word_gap_mm", "inter-word gap"),
    ]

    for key, label in checks:
        page_val   = page_stats.get(key, {})
        corpus_val = corpus_stats.get(key, {})

        if not page_val or not corpus_val:
            continue

        page_median   = page_val.get("p50", 0)
        corpus_median = corpus_val.get("p50", 0)
        corpus_iqr    = corpus_val.get("iqr", 0)

        if corpus_iqr > 0:
            deviation = abs(page_median - corpus_median)
            if deviation > factor * corpus_iqr:
                return True, (
                    f"{label}: page median {page_median:.1f} deviates "
                    f"{deviation:.1f} from corpus median {corpus_median:.1f} "
                    f"(>{factor}×IQR={corpus_iqr:.1f})"
                )

    return False, ""
