"""
Output — JSON Serialization
============================
Saves extracted word strokes and page statistics to JSON.
One output file per scanned page.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .steps.segmenter import WordCrop
from .steps.stroke_extractor import StrokePoint

log = logging.getLogger(__name__)


def build_word_record(
    word_crop:  WordCrop,
    label:      str,                    # the written word (from word list)
    strokes:    List[StrokePoint],
    bbox_mm:    Dict[str, float],       # width, ascent, descent
    config,
) -> Dict[str, Any]:
    """Build a single word record for the dataset."""
    return {
        "word":        label,
        "strokes":     [[round(s.x, 4), round(s.y, 4), s.pen_up]
                        for s in strokes],
        "bbox_mm": {
            "width":   round(bbox_mm.get("width",   0.0), 3),
            "ascent":  round(bbox_mm.get("ascent",  0.0), 3),
            "descent": round(bbox_mm.get("descent", 0.0), 3),
        },
        "source": {
            "pair_index":  word_crop.pair_index,
            "word_index":  word_crop.word_index,
        },
        "flagged":     word_crop.flagged,
        "flag_reason": word_crop.flag_reason,
        "stroke_count": len([s for s in strokes if s.pen_up == 1]),
        "point_count":  len(strokes),
    }


def save_page_output(
    source_file:  str,
    word_records: List[Dict[str, Any]],
    page_stats:   Dict[str, Any],
    output_dir:   str,
    dpi:          int,
) -> Path:
    """
    Save all extracted words and page stats from one scan to JSON.
    Returns path of saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_file).stem
    out_path = output_dir / f"{stem}_extracted.json"

    payload = {
        "source_file": str(source_file),
        "dpi":         dpi,
        "page_stats":  page_stats,
        "words":       word_records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    clean   = sum(1 for w in word_records if not w["flagged"])
    flagged = sum(1 for w in word_records if w["flagged"])
    log.info(f"Saved {out_path}: {clean} clean + {flagged} flagged words")

    return out_path


def compute_bbox_mm(strokes: List[StrokePoint]) -> Dict[str, float]:
    """
    Compute bounding box from normalized stroke points.
    x range → width, positive y range → ascent, negative y range → descent.
    """
    if not strokes:
        return {"width": 0.0, "ascent": 0.0, "descent": 0.0}

    xs = [s.x for s in strokes]
    ys = [s.y for s in strokes]

    width   = max(xs) - min(xs) if xs else 0.0
    ascent  = max(0.0, max(ys)) if ys else 0.0   # above baseline
    descent = max(0.0, -min(ys)) if ys else 0.0  # below baseline (positive)

    return {
        "width":   round(width,   3),
        "ascent":  round(ascent,  3),
        "descent": round(descent, 3),
    }
