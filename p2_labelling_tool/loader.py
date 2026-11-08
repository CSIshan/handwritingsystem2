"""
P2 Loader
=========
Reads all extracted JSONs from dataset/raw/,
returns a flat ordered list of word records.
Flagged words come first, then clean words (randomly sampled
at spot_check_ratio for verification).
"""
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

log = logging.getLogger(__name__)


def load_raw_dataset(
    raw_dir:          str,
    spot_check_ratio: float = 0.15,
    seed:             int   = 42,
) -> List[Dict[str, Any]]:
    """
    Load all *_extracted.json files from raw_dir.
    Returns flat list of word records, flagged first.

    Each record gets extra fields added:
        _source_file:  which JSON file it came from
        _record_id:    unique stable ID for progress tracking
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    json_files = sorted(raw_dir.glob("*_extracted.json"))
    if not json_files:
        raise FileNotFoundError(f"No extracted JSON files found in {raw_dir}")

    log.info(f"Loading {len(json_files)} extracted files from {raw_dir}")

    flagged = []
    clean   = []

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                page_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping {json_path.name}: {e}")
            continue

        source = json_path.stem  # e.g. "session1_page1_extracted"

        for word in page_data.get("words", []):
            # Stable unique ID: source + pair + word index
            record_id = (
                f"{source}"
                f"_p{word.get('source', {}).get('pair_index', 0)}"
                f"_w{word.get('source', {}).get('word_index', 0)}"
            )
            word["_source_file"] = source
            word["_record_id"]   = record_id

            if word.get("flagged"):
                flagged.append(word)
            else:
                clean.append(word)

    # Spot-check sample of clean words
    rng = random.Random(seed)
    n_spot = max(1, int(len(clean) * spot_check_ratio))
    spot_sample = rng.sample(clean, min(n_spot, len(clean)))
    # Mark spot-checked words so UI can indicate they're optional
    for w in spot_sample:
        w["_spot_check"] = True

    # Flagged first, then spot-check sample
    ordered = flagged + spot_sample

    log.info(
        f"Loaded {len(flagged)} flagged + "
        f"{len(spot_sample)} spot-check ({spot_check_ratio:.0%} of "
        f"{len(clean)} clean) = {len(ordered)} total to review"
    )

    return ordered
