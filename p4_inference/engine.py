"""
P4 Inference Engine
====================
Loads trained model weights and provides a clean interface
for generating handwriting strokes from word strings.

Usage:
    engine = InferenceEngine.load("dataset/model/checkpoint_best.pt")
    strokes = engine.generate("hello")
    width, ascent, descent = engine.get_bbox("hello")
"""
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

log = logging.getLogger(__name__)


@dataclass
class StrokePoint:
    """
    Single point in a stroke sequence.
    Coordinates in mm, baseline-normalized:
        y > 0  → above baseline (ascenders)
        y = 0  → on baseline
        y < 0  → below baseline (descenders)
    """
    x:      float
    y:      float
    pen_up: int    # 1 = lift pen after this point, 0 = keep drawing


class InferenceEngine:
    """
    Wraps the trained HandwritingModel for clean downstream use.

    Strokes:  NOT cached — every generate() call produces natural variation.
    BBoxes:   cached    — expected size is stable, no need to recompute.
    """

    def __init__(self, model, vocab: Dict[str, int], device):
        self._model  = model
        self._vocab  = vocab
        self._device = device
        self._unk_id = vocab.get("<UNK>", 1)
        self._bbox_cache: Dict[str, Tuple[float, float, float]] = {}

        # Known chars for unknown-char detection
        self._known_chars = set(vocab.keys())

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device_str:      str = "auto",
        strict:          bool = True,
    ) -> "InferenceEngine":
        """
        Load engine from a training checkpoint.

        checkpoint_path: path to checkpoint_best.pt or any checkpoint
        device_str:      "auto" | "mps" | "cpu"
        strict:          whether to enforce exact weight matching
        """
        import torch
        from handwriting_system.p3_training.model  import HandwritingModel
        from handwriting_system.p3_training.config import TrainConfig

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        # Map to CPU first — safe across all devices
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")

        # Reconstruct config from saved values
        saved_cfg = ckpt.get("config", {})
        cfg = TrainConfig(**{
            k: v for k, v in saved_cfg.items()
            if k in TrainConfig.__dataclass_fields__
        })

        # Load vocab — prefer alongside checkpoint, fall back to training dir
        vocab_path = checkpoint_path.parent / "vocab.json"
        if not vocab_path.exists():
            vocab_path = Path(cfg.training_dir) / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found near {checkpoint_path}. "
                f"Expected at {checkpoint_path.parent / 'vocab.json'}"
            )
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

        # Resolve device
        device = cls._resolve_device(device_str)
        log.info(f"Loading model on {device}")

        # Build model + load weights
        vocab_size = ckpt.get("vocab_size", len(vocab))
        model = HandwritingModel(cfg, vocab_size)
        model.load_state_dict(ckpt["model_state"], strict=strict)
        model.to(device)
        model.eval()

        epoch = ckpt.get("epoch", "?")
        log.info(
            f"Loaded checkpoint (epoch {epoch}) "
            f"from {checkpoint_path.name}"
        )

        return cls(model, vocab, device)

    @staticmethod
    def _resolve_device(device_str: str):
        import torch
        if device_str == "auto":
            if torch.backends.mps.is_available():
                try:
                    torch.zeros(1).to("mps")
                    log.info("Using MPS (Apple Silicon)")
                    return torch.device("mps")
                except Exception:
                    pass
            log.info("Using CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        word:        str,
        temperature: float = 1.0,
        max_steps:   int   = 500,
    ) -> List[StrokePoint]:
        """
        Generate stroke sequence for a word.
        Returns different result each call (natural variation).

        temperature: 0.5=neat/tight, 1.0=natural, 1.5=loose/casual
        """
        self._warn_unknown_chars(word)

        raw = self._model.generate(
            word        = word,
            vocab       = self._vocab,
            temperature = temperature,
            max_steps   = max_steps,
            device      = self._device,
        )

        # Convert from training scale (×10) back to mm
        # Accumulate deltas → absolute positions
        strokes = []
        x, y = 0.0, 0.0
        for dx_scaled, dy_scaled, pen_up in raw:
            x += dx_scaled / 10.0
            y += dy_scaled / 10.0
            strokes.append(StrokePoint(
                x      = round(x, 4),
                y      = round(y, 4),
                pen_up = int(pen_up),
            ))

        # Shift so x starts at 0 (word left edge)
        if strokes:
            x_min = min(s.x for s in strokes)
            for s in strokes:
                s.x = round(s.x - x_min, 4)

        return strokes

    def generate_batch(
        self,
        words:       List[str],
        temperature: float = 1.0,
        max_steps:   int   = 500,
    ) -> List[List[StrokePoint]]:
        """
        Generate strokes for multiple words.
        Currently sequential — batched LSTM inference is a future optimisation.
        Returns list of stroke lists, one per word.
        """
        return [
            self.generate(w, temperature, max_steps)
            for w in words
        ]

    # ── Bounding box ──────────────────────────────────────────────────────────

    def get_bbox(
        self,
        word:        str,
        n_samples:   int = 5,
    ) -> Tuple[float, float, float]:
        """
        Return (width_mm, ascent_mm, descent_mm) for a word.
        Cached — same result returned on subsequent calls.

        Averages over n_samples generations to get stable bbox
        (single generation varies too much for layout purposes).
        Uses median of samples — robust to outlier generations.
        """
        if word in self._bbox_cache:
            return self._bbox_cache[word]

        import numpy as np

        widths, ascents, descents = [], [], []
        for _ in range(n_samples):
            strokes = self.generate(word, temperature=0.8)
            if not strokes:
                continue
            xs = [s.x for s in strokes]
            ys = [s.y for s in strokes]
            widths.append(max(xs) - min(xs))
            ascents.append(max(0.0,  max(ys)))
            descents.append(max(0.0, -min(ys)))

        if not widths:
            # Fallback if generation failed
            result = (10.0, 5.0, 1.0)
        else:
            result = (
                float(np.median(widths)),
                float(np.median(ascents)),
                float(np.median(descents)),
            )

        self._bbox_cache[word] = result
        return result

    def clear_bbox_cache(self):
        """Clear bbox cache — call if model weights are updated."""
        self._bbox_cache.clear()

    # ── Unknown character warning ─────────────────────────────────────────────

    def _warn_unknown_chars(self, word: str):
        unknown = [c for c in word if c not in self._known_chars]
        if unknown:
            log.warning(
                f"Word {word!r} contains unknown chars {unknown} "
                f"→ substituting <UNK>. Quality may degrade."
            )

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def known_chars(self) -> set:
        return self._known_chars.copy()

    def __repr__(self):
        return (
            f"InferenceEngine("
            f"vocab_size={self.vocab_size}, "
            f"device={self._device}, "
            f"bbox_cached={len(self._bbox_cache)} words)"
        )
