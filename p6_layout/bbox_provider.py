"""
P6 BBox Provider
================
Provides bounding box estimates for tokens.

MockBBoxProvider  — used during development, estimates from char count.
ModelBBoxProvider — slots in once model is trained, returns real stroke extents.

The layout engine calls only get_bbox(token) → BBox.
It never knows which implementation is running.
"""

import logging
from abc import ABC, abstractmethod
from ..p5_tokenizer.token_types import Token, TokenType
from .data_structures import BBox, LayoutConfig

log = logging.getLogger(__name__)

# Average character widths in mm for cursive at normal writing size.
# Cursive is narrower than print because letters connect and compress.
# These are tuned approximations — real model will replace them.
CURSIVE_CHAR_WIDTH = {
    # Narrow letters
    'i': 2.0, 'l': 2.0, 'j': 2.2, 'r': 2.5, 't': 2.5, 'f': 2.8,
    # Medium letters
    'a': 3.2, 'b': 3.2, 'c': 3.0, 'd': 3.2, 'e': 3.0, 'g': 3.2,
    'h': 3.5, 'k': 3.2, 'n': 3.5, 'o': 3.2, 'p': 3.2, 'q': 3.2,
    's': 2.8, 'u': 3.5, 'v': 3.2, 'x': 3.2, 'y': 3.2, 'z': 3.0,
    # Wide letters
    'm': 5.0, 'w': 4.5,
    # Uppercase (generally wider)
    'M': 6.0, 'W': 6.0, 'A': 4.5, 'B': 4.0, 'C': 4.0, 'D': 4.5,
    'E': 3.8, 'F': 3.8, 'G': 4.5, 'H': 4.5, 'I': 2.5, 'J': 3.0,
    'K': 4.2, 'L': 3.8, 'N': 4.5, 'O': 4.8, 'P': 4.0, 'Q': 4.8,
    'R': 4.2, 'S': 3.5, 'T': 4.0, 'U': 4.5, 'V': 4.2, 'X': 4.2,
    'Y': 4.0, 'Z': 4.0,
    # Digits
    '0': 3.5, '1': 2.5, '2': 3.2, '3': 3.2, '4': 3.5, '5': 3.2,
    '6': 3.5, '7': 3.0, '8': 3.5, '9': 3.5,
    # Punctuation (attached, contributes small width)
    ',': 1.2, '.': 1.0, '!': 1.5, '?': 2.5, ';': 1.5, ':': 1.5,
    "'": 0.8, '"': 1.5, '-': 2.0, '(': 1.5, ')': 1.5,
}

DEFAULT_CHAR_WIDTH = 3.2   # fallback for unknown chars
DIGIT_CHAR_WIDTH   = 3.2


# Letters with ascenders (go above mid-height)
ASCENDER_CHARS  = frozenset('bdfhijkltABCDEFGHIJKLMNOPQRSTUVWXYZ')
# Letters with descenders (go below baseline)
DESCENDER_CHARS = frozenset('gjpqy')


def _estimate_width(text: str) -> float:
    """Estimate cursive word width in mm from character composition."""
    if not text:
        return 2.0
    total = sum(CURSIVE_CHAR_WIDTH.get(c, DEFAULT_CHAR_WIDTH) for c in text)
    # Cursive compression: connected letters save ~8% vs isolated
    return total * 0.92


def _estimate_ascent(text: str, default_ascent: float) -> float:
    """
    Ascent = cap height if word has uppercase or ascenders, else x-height.
    In cursive at normal size:
        cap height ≈ 6mm
        x-height   ≈ 3.5mm
    """
    if any(c in ASCENDER_CHARS for c in text):
        return default_ascent         # full cap/ascender height
    return default_ascent * 0.58     # x-height only


def _estimate_descent(text: str, default_descent: float) -> float:
    """Descent only if word contains descender letters."""
    if any(c in DESCENDER_CHARS for c in text):
        return default_descent
    return 0.0


class BBoxProvider(ABC):
    @abstractmethod
    def get_bbox(self, token: Token) -> BBox:
        """Return bounding box for a token. Always in mm, baseline-normalized."""
        ...


class MockBBoxProvider(BBoxProvider):
    """
    Estimates bounding boxes from character composition.
    Used during development before the ML model is trained.
    Produces plausible layout without any model dependency.
    """

    def __init__(self, config: LayoutConfig):
        self.config = config

    def get_bbox(self, token: Token) -> BBox:
        text = token.value or ""

        if token.type.name == "WORD":
            return BBox(
                width   = _estimate_width(text),
                ascent  = _estimate_ascent(text,  self.config.default_ascent),
                descent = _estimate_descent(text, self.config.default_descent),
            )

        elif token.type.name == "DIGIT":
            w = sum(CURSIVE_CHAR_WIDTH.get(c, DIGIT_CHAR_WIDTH) for c in text)
            return BBox(width=w, ascent=self.config.default_ascent * 0.85,
                        descent=0.0)

        elif token.type.name == "SYMBOL":
            return BBox(width=3.0, ascent=self.config.default_ascent * 0.7,
                        descent=0.0)

        elif token.type.name == "GHOST":
            # Sum bboxes of subwords + inter-word gaps
            subwords = token.metadata.get("subwords", [text])
            total_w = 0.0
            max_asc = 0.0
            max_des = 0.0
            for i, sw in enumerate(subwords):
                w = _estimate_width(sw)
                a = _estimate_ascent(sw,  self.config.default_ascent)
                d = _estimate_descent(sw, self.config.default_descent)
                total_w += w + (self.config.inter_word_gap if i > 0 else 0)
                max_asc  = max(max_asc, a)
                max_des  = max(max_des, d)
            return BBox(width=total_w, ascent=max_asc, descent=max_des)

        elif token.type.name == "CORRECTION":
            struck  = _estimate_width(token.value or "")
            rewrite = _estimate_width(token.metadata.get("rewrite") or "")
            gap     = 2.0 if rewrite else 0.0
            return BBox(
                width   = struck + gap + rewrite,
                ascent  = self.config.default_ascent,
                descent = self.config.default_descent,
            )

        # State tokens and others take no space
        return BBox(width=0.0, ascent=0.0, descent=0.0)


class ModelBBoxProvider(BBoxProvider):
    """
    Real bbox provider — uses trained inference engine.
    Slots in when model is trained. Zero layout code changes.

    inference_engine must implement:
        get_bbox(word: str) -> (width_mm, ascent_mm, descent_mm)
    """

    def __init__(self, inference_engine, config: LayoutConfig):
        self.engine = inference_engine
        self.config = config
        self._cache: dict = {}   # cache bboxes — same word same bbox

    def get_bbox(self, token: Token) -> BBox:
        text = token.value or ""
        if not text:
            return BBox(0.0, 0.0, 0.0)

        if text not in self._cache:
            try:
                width, ascent, descent = self.engine.get_bbox(text)
                self._cache[text] = BBox(width, ascent, descent)
            except Exception as e:
                log.warning(f"ModelBBoxProvider failed for {text!r}: {e} — falling back to mock")
                mock = MockBBoxProvider(self.config)
                self._cache[text] = mock.get_bbox(token)

        return self._cache[text]
