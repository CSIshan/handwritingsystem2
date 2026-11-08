"""
P1 Scan Pipeline — Main Orchestrator
======================================
ScanPipeline.process(image_path, word_list, output_dir)

word_list: the words you wrote on that page, in order left-to-right,
           top-to-bottom. Used to label extracted crops automatically.
           If None, all records are flagged for manual labelling in P2.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import ScanConfig
from .steps.loader import load_image, LoadError
from .steps.deskew import deskew
from .steps.binarize import binarize
from .steps.line_detector import detect_line_pairs
from .steps.segmenter import segment_words
from .steps.stroke_extractor import extract_strokes
from .stats import measure_page_stats
from .output import build_word_record, save_page_output, compute_bbox_mm

log = logging.getLogger(__name__)


def _load_wordlist(wordlist_path: str) -> List[str]:
    """
    Read a .txt word list. One word per line.
    Strips whitespace, skips blank lines and # comments.
    """
    path = Path(wordlist_path)
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {wordlist_path}")
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


class ScanPipeline:

    def __init__(self, config: ScanConfig = None):
        self.config = config or ScanConfig()

    def process(
        self,
        image_path: str,
        word_list:  Optional[List[str]] = None,
        output_dir: str = "dataset/raw",
    ) -> Dict[str, Any]:
        """
        Process one scanned page.

        image_path: path to PNG/JPG scan
        word_list:  ordered list of words you wrote on the page.
                    If provided, crops are labelled automatically.
                    If None, all crops flagged for manual labelling.
        output_dir: where to save the JSON output

        Returns summary dict with counts and output path.
        """
        cfg = self.config
        log.info(f"Processing: {image_path}")

        # ── Step 1: Load ──────────────────────────────────────────────────────
        try:
            grey = load_image(image_path, cfg)
        except LoadError as e:
            log.error(f"Load failed: {e}")
            return {"success": False, "error": str(e)}

        # ── Step 2: Deskew ────────────────────────────────────────────────────
        grey, deskew_angle = deskew(grey)

        # ── Step 3: Binarize + remove template lines ──────────────────────────
        binary = binarize(grey, cfg)

        # ── Step 4: Detect line pairs ─────────────────────────────────────────
        pairs = detect_line_pairs(binary, cfg)
        if not pairs:
            log.error("No line pairs detected — check scan quality")
            return {"success": False, "error": "no line pairs detected"}

        # ── Step 5: Segment words ─────────────────────────────────────────────
        word_crops = segment_words(binary, pairs, cfg)
        if not word_crops:
            log.error("No words segmented — check scan quality")
            return {"success": False, "error": "no words segmented"}

        # ── Steps 6–9: Extract strokes for each crop ──────────────────────────
        word_records = []
        label_idx    = 0

        for crop in word_crops:
            strokes = extract_strokes(
                crop       = crop.image,
                baseline_y = crop.baseline_y,
                strip_top  = crop.top_y,
                config     = cfg,
            )

            # Assign label from word_list if provided
            if word_list and label_idx < len(word_list):
                label = word_list[label_idx]
                label_idx += 1
            else:
                label = ""
                crop.flagged     = True
                crop.flag_reason = (crop.flag_reason or "") + " | no label provided"

            bbox = compute_bbox_mm(strokes)

            record = build_word_record(
                word_crop = crop,
                label     = label,
                strokes   = strokes,
                bbox_mm   = bbox,
                config    = cfg,
            )
            word_records.append(record)

        # ── Page statistics ───────────────────────────────────────────────────
        page_stats = measure_page_stats(pairs, word_crops, cfg, deskew_angle)

        # ── Save output ───────────────────────────────────────────────────────
        out_path = save_page_output(
            source_file  = image_path,
            word_records = word_records,
            page_stats   = page_stats,
            output_dir   = output_dir,
            dpi          = cfg.dpi,
        )

        summary = {
            "success":       True,
            "output_path":   str(out_path),
            "total_words":   len(word_records),
            "clean_words":   sum(1 for w in word_records if not w["flagged"]),
            "flagged_words": sum(1 for w in word_records if w["flagged"]),
            "pairs_found":   len(pairs),
            "deskew_angle":  deskew_angle,
        }

        log.info(
            f"Done: {summary['clean_words']} clean, "
            f"{summary['flagged_words']} flagged → {out_path}"
        )
        return summary

    def process_with_wordlist(
        self,
        image_path:    str,
        wordlist_path: str = None,
        output_dir:    str = "dataset/raw",
    ) -> dict:
        """
        Convenience method — reads word list from a .txt file automatically.

        If wordlist_path is None, looks for a .txt file with the same name
        as the image in the same directory:
            dataset/scans/session1/page1.png
            → dataset/scans/session1/page1.txt  (auto-discovered)

        .txt format: one word per line, blank lines and # comments ignored.
        """
        image_path = Path(image_path)

        # Auto-discover word list if not provided
        if wordlist_path is None:
            wordlist_path = image_path.with_suffix(".txt")
            if not wordlist_path.exists():
                log.warning(
                    f"No word list found at {wordlist_path}. "
                    f"All crops will be flagged for manual labelling in P2."
                )
                wordlist_path = None

        word_list = None
        if wordlist_path is not None:
            try:
                word_list = _load_wordlist(str(wordlist_path))
                log.info(f"Loaded {len(word_list)} words from {wordlist_path}")
            except FileNotFoundError as e:
                log.warning(f"{e} — proceeding without word list")

        return self.process(str(image_path), word_list, output_dir)

    def process_session(
        self,
        session_dir: str,
        output_dir:  str = "dataset/raw",
    ) -> list:
        """
        Batch process all PNG/JPG scans in a session folder.
        Automatically pairs each image with its .txt word list.

        session_dir structure:
            session1/
                page1.png
                page1.txt
                page2.png
                page2.txt
                ...

        Returns list of summary dicts, one per image.
        """
        session_dir = Path(session_dir)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        images = sorted(
            p for p in session_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not images:
            log.warning(f"No images found in {session_dir}")
            return []

        log.info(f"Processing session: {session_dir} ({len(images)} images)")

        summaries = []
        for image_path in images:
            log.info(f"--- {image_path.name} ---")
            summary = self.process_with_wordlist(
                image_path = str(image_path),
                output_dir = output_dir,
            )
            summary["image"] = image_path.name
            summaries.append(summary)

        total_clean   = sum(s.get("clean_words",   0) for s in summaries)
        total_flagged = sum(s.get("flagged_words",  0) for s in summaries)
        log.info(
            f"Session complete: {len(images)} pages, "
            f"{total_clean} clean words, {total_flagged} flagged"
        )
        return summaries
