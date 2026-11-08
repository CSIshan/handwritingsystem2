"""
P2 Saver
========
Writes verified word records to dataset/verified/.
Groups records back by source file — one verified JSON per scan page.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from .progress import Progress

log = logging.getLogger(__name__)


def save_verified(
    all_records: List[Dict[str, Any]],
    progress:    Progress,
    verified_dir: str = "dataset/verified",
):
    """
    Write all accepted/edited records to verified_dir.
    Groups by source file. Discarded records are not written.
    """
    verified_dir = Path(verified_dir)
    verified_dir.mkdir(parents=True, exist_ok=True)

    # Group records by source file
    by_source: Dict[str, List[Dict]] = {}
    for record in all_records:
        source = record.get("_source_file", "unknown")
        by_source.setdefault(source, []).append(record)

    total_saved    = 0
    total_discarded = 0

    for source, records in by_source.items():
        verified_words = []

        for record in records:
            rid      = record["_record_id"]
            decision = progress.decided.get(rid)

            if decision is None:
                # Not yet reviewed — skip (not verified)
                continue

            if decision == "discard":
                total_discarded += 1
                continue

            # Build verified record
            word_record = {
                "word":           record["word"],
                "strokes":        record["strokes"],
                "bbox_mm":        record["bbox_mm"],
                "verified":       True,
                "edited":         False,
                "original_label": record["word"],
                "source":         record.get("source", {}),
            }

            if isinstance(decision, dict) and decision.get("action") == "edit":
                word_record["word"]           = decision["new_label"]
                word_record["original_label"] = decision["original_label"]
                word_record["edited"]         = True

            verified_words.append(word_record)

        if not verified_words:
            continue

        # Save to verified/
        out_name = source.replace("_extracted", "_verified") + ".json"
        out_path = verified_dir / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source_file": source, "words": verified_words},
                f, indent=2, ensure_ascii=False
            )

        total_saved += len(verified_words)
        log.info(f"Saved {len(verified_words)} verified words → {out_path}")

    log.info(
        f"Save complete: {total_saved} verified, "
        f"{total_discarded} discarded"
    )
    return {"saved": total_saved, "discarded": total_discarded}
