"""
P3 Dataset
===========
PyTorch Dataset wrapping the training JSON.
Handles tokenization, stroke encoding, padding, and batching.
"""
import json
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any

from .prep import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN


class HandwritingDataset(Dataset):
    """
    Each item: one word with its stroke sequence.

    Returns:
        char_ids:    LongTensor [seq_len]        character indices
        strokes:     FloatTensor [stroke_len, 3] (Δx, Δy, pen_up)
        char_len:    int                          actual character count
        stroke_len:  int                          actual stroke point count
    """

    def __init__(
        self,
        json_path:   str,
        vocab:       Dict[str, int],
        max_seq_len: int = 500,
    ):
        self.vocab       = vocab
        self.max_seq_len = max_seq_len
        self.unk_id      = vocab[UNK_TOKEN]
        self.pad_id      = vocab[PAD_TOKEN]

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        # Filter records with valid strokes
        self.records = [
            r for r in data
            if r.get("word") and r.get("strokes") and len(r["strokes"]) >= 3
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        word   = record["word"]
        strokes= record["strokes"]   # [[Δx, Δy, pen_up], ...]

        # ── Encode characters ─────────────────────────────────────────────────
        char_ids = [self.vocab.get(c, self.unk_id) for c in word]

        # ── Encode strokes ────────────────────────────────────────────────────
        # Truncate if needed
        strokes = strokes[:self.max_seq_len]

        # Convert to float tensors
        stroke_tensor = torch.tensor(strokes, dtype=torch.float32)

        # Normalise Δx, Δy to roughly unit scale
        # Median offset in mm is small (~0.1-0.5mm), we scale to help training
        stroke_tensor[:, 0] = stroke_tensor[:, 0] * 10.0   # Δx scale
        stroke_tensor[:, 1] = stroke_tensor[:, 1] * 10.0   # Δy scale
        # pen_up stays 0 or 1

        return {
            "word":       word,
            "char_ids":   torch.tensor(char_ids, dtype=torch.long),
            "strokes":    stroke_tensor,
            "char_len":   len(char_ids),
            "stroke_len": len(strokes),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Pad sequences to max length in batch.
    Called by DataLoader to assemble batches.
    """
    # Find max lengths in this batch
    max_char_len   = max(item["char_len"]   for item in batch)
    max_stroke_len = max(item["stroke_len"] for item in batch)

    char_ids_padded   = []
    strokes_padded    = []
    char_lens         = []
    stroke_lens       = []
    words             = []

    for item in batch:
        # Pad character IDs
        cids = item["char_ids"]
        pad_c = max_char_len - item["char_len"]
        char_ids_padded.append(
            torch.cat([cids, torch.zeros(pad_c, dtype=torch.long)])
        )

        # Pad strokes with zeros (pen_up=0 padding, layout engine handles end)
        s = item["strokes"]
        pad_s = max_stroke_len - item["stroke_len"]
        strokes_padded.append(
            torch.cat([s, torch.zeros(pad_s, 3)], dim=0)
        )

        char_lens.append(item["char_len"])
        stroke_lens.append(item["stroke_len"])
        words.append(item["word"])

    return {
        "words":       words,
        "char_ids":    torch.stack(char_ids_padded),     # [B, max_char_len]
        "strokes":     torch.stack(strokes_padded),      # [B, max_stroke_len, 3]
        "char_lens":   torch.tensor(char_lens),          # [B]
        "stroke_lens": torch.tensor(stroke_lens),        # [B]
    }


def make_dataloaders(
    training_dir: str,
    batch_size:   int,
    vocab:        Dict[str, int],
    max_seq_len:  int = 500,
    seed:         int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    training_dir = Path(training_dir)

    train_ds = HandwritingDataset(
        training_dir / "train.json", vocab, max_seq_len
    )
    val_ds = HandwritingDataset(
        training_dir / "val.json", vocab, max_seq_len
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        generator   = g,
        num_workers = 0,   # MPS doesn't support multiprocessing workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    return train_loader, val_loader
