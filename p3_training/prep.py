"""
P3 Data Preparation
====================
Merges all verified JSONs → dataset/training/dataset.json
Builds character vocabulary, computes corpus stats.

Run once before training:
    from handwriting_system.p3_training.prep import prepare_dataset
    prepare_dataset()
"""
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

log = logging.getLogger(__name__)

# Full character vocabulary
# Lowercase, uppercase, digits, punctuation you'll write
VOCAB_CHARS = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + list(".,!?;:'\"-()[]&@%$#/+= ")
)

PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN   = "<END>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]


def build_vocab(words: List[str]) -> Dict[str, int]:
    """
    Build character → index mapping.
    Special tokens first, then all chars seen in dataset + VOCAB_CHARS.
    """
    chars_in_data = set()
    for word in words:
        chars_in_data.update(word)

    all_chars = set(VOCAB_CHARS) | chars_in_data
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for i, ch in enumerate(sorted(all_chars)):
        vocab[ch] = len(SPECIAL_TOKENS) + i

    log.info(f"Vocabulary: {len(vocab)} tokens ({len(chars_in_data)} unique chars in data)")
    return vocab


def load_verified(verified_dir: str) -> List[Dict[str, Any]]:
    """Load all verified JSON files, return flat list of word records."""
    verified_dir = Path(verified_dir)
    files = sorted(verified_dir.glob("*_verified.json"))

    if not files:
        raise FileNotFoundError(f"No verified JSONs in {verified_dir}")

    records = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for word in data.get("words", []):
            if word.get("verified") and word.get("strokes"):
                records.append(word)

    log.info(f"Loaded {len(records)} verified records from {len(files)} files")
    return records


def deduplicate(records: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicates — same word AND identical stroke sequence.
    Keeps all natural variation (same word written differently = keep both).
    """
    seen   = set()
    unique = []
    for r in records:
        key = (r["word"], str(r["strokes"][:5]))  # word + first 5 points fingerprint
        if key not in seen:
            seen.add(key)
            unique.append(r)
    removed = len(records) - len(unique)
    if removed:
        log.info(f"Deduplication: removed {removed} exact duplicates")
    return unique


def compute_corpus_stats(records: List[Dict]) -> Dict[str, Any]:
    """
    Compute corpus-level percentile distributions.
    These become the real humaniser parameters.
    Uses percentiles + IQR, never averages.
    """
    import numpy as np

    def pct(values):
        if not values:
            return {}
        a = np.array(values)
        p25, p50, p75 = np.percentile(a, [25, 50, 75])
        return {
            "p10": float(np.percentile(a, 10)),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p90": float(np.percentile(a, 90)),
            "iqr": float(p75 - p25),
            "n":   len(values),
        }

    widths   = [r["bbox_mm"]["width"]   for r in records if r.get("bbox_mm")]
    ascents  = [r["bbox_mm"]["ascent"]  for r in records if r.get("bbox_mm")]
    descents = [r["bbox_mm"]["descent"] for r in records if r.get("bbox_mm")]
    lengths  = [len(r["strokes"])       for r in records]

    stats = {
        "word_width_mm":    pct(widths),
        "word_ascent_mm":   pct(ascents),
        "word_descent_mm":  pct(descents),
        "stroke_seq_length":pct(lengths),
        "total_words":      len(records),
        "unique_words":     len(set(r["word"] for r in records)),
    }

    log.info(
        f"Corpus stats: {stats['total_words']} words, "
        f"{stats['unique_words']} unique, "
        f"median width={stats['word_width_mm'].get('p50',0):.1f}mm"
    )
    return stats


def prepare_dataset(
    verified_dir: str = "dataset/verified",
    training_dir: str = "dataset/training",
    seed:         int = 42,
    train_split:  float = 0.9,
) -> Dict[str, Any]:
    """
    Full prep pipeline:
    1. Load verified JSONs
    2. Deduplicate
    3. Build vocabulary
    4. Compute corpus stats
    5. Train/val split
    6. Save to training_dir

    Returns summary dict.
    """
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Load + clean
    records = load_verified(verified_dir)
    records = deduplicate(records)

    # Vocabulary
    words = [r["word"] for r in records]
    vocab = build_vocab(words)

    # Corpus stats
    stats = compute_corpus_stats(records)

    # Shuffle + split
    rng = random.Random(seed)
    rng.shuffle(records)
    n_train = int(len(records) * train_split)
    train_records = records[:n_train]
    val_records   = records[n_train:]

    # Save everything
    (training_dir / "train.json").write_text(
        json.dumps(train_records, indent=2, ensure_ascii=False)
    )
    (training_dir / "val.json").write_text(
        json.dumps(val_records, indent=2, ensure_ascii=False)
    )
    (training_dir / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False)
    )
    (training_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    summary = {
        "total":    len(records),
        "train":    len(train_records),
        "val":      len(val_records),
        "vocab_size": len(vocab),
    }

    log.info(
        f"Dataset ready: {summary['train']} train, "
        f"{summary['val']} val, vocab={summary['vocab_size']}"
    )
    return summary
