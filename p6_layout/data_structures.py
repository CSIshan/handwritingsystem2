"""
P6 Layout Engine — Data Structures
===================================
All spatial units are millimetres.
Baseline convention: Y=0 is the baseline. Ascenders are positive Y,
descenders are negative Y. This matches the ML model's normalized
output coordinate space exactly.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from ..p5_tokenizer.token_types import Token, TokenType


@dataclass
class LayoutConfig:
    # Page dimensions
    page_width:       float = 210.0   # mm, A4
    page_height:      float = 297.0   # mm, A4

    # Margins
    margin_top:       float = 20.0
    margin_bottom:    float = 20.0
    margin_left:      float = 20.0
    margin_right:     float = 20.0

    # Spacing — overridden by real model bbox stats once trained
    default_ascent:   float = 6.0     # mm above baseline (cap height)
    default_descent:  float = 2.0     # mm below baseline (descender)
    leading:          float = 3.0     # mm extra gap between lines

    # Word spacing
    inter_word_gap:   float = 3.0     # mm between words
    inter_digit_gap:  float = 1.5     # mm between digit tokens

    # Paragraph
    paragraph_gap:    float = 1.5     # multiplier on line_height for para break
    indent_enabled:   bool  = True
    indent:           float = 10.0    # mm, first line of paragraph

    # Margin notes column
    margin_note_x:    float = 5.0     # mm gap from right margin edge

    # Image
    default_img_width: float = 80.0   # mm, fallback if not specified

    @property
    def usable_width(self) -> float:
        return self.page_width - self.margin_left - self.margin_right

    @property
    def usable_height(self) -> float:
        return self.page_height - self.margin_top - self.margin_bottom

    @property
    def default_line_height(self) -> float:
        return self.default_ascent + self.default_descent + self.leading


@dataclass
class BBox:
    """Bounding box in baseline-normalized space (mm)."""
    width:   float   # horizontal extent
    ascent:  float   # above baseline (positive)
    descent: float   # below baseline (positive number, strokes go negative)

    @property
    def height(self) -> float:
        return self.ascent + self.descent


@dataclass
class PlacedToken:
    """
    A token with its exact position on a page.

    x          — mm from page left edge to word's left edge
    baseline_y — mm from page top to the baseline
                 All stroke Y coords are relative to this:
                 abs_stroke_y = baseline_y + stroke.y  (stroke.y is ±)

    The layout engine places by baseline_y exclusively.
    Ascent/descent tell humaniser and SVG composer the vertical extents.
    """
    token:      Token
    x:          float
    baseline_y: float
    bbox:       BBox
    page:       int          # 0-indexed
    line:       int          # 0-indexed within page
    is_ghost:   bool = False
    style:      str  = "casual"   # current style at time of placement
    pen:        str  = "black"    # current pen at time of placement

    @property
    def width(self)  -> float: return self.bbox.width
    @property
    def ascent(self) -> float: return self.bbox.ascent
    @property
    def descent(self)-> float: return self.bbox.descent

    @property
    def top_y(self) -> float:
        """Absolute Y of topmost stroke extent."""
        return self.baseline_y - self.ascent   # page Y increases downward

    @property
    def bottom_y(self) -> float:
        """Absolute Y of bottommost stroke extent."""
        return self.baseline_y + self.descent


@dataclass
class Line:
    """A single horizontal line of placed tokens on a page."""
    index:      int
    baseline_y: float          # shared baseline for all tokens on this line
    ascent:     float          # max ascent across all tokens on line
    descent:    float          # max descent across all tokens on line
    tokens:     List[PlacedToken] = field(default_factory=list)

    @property
    def line_height(self) -> float:
        return self.ascent + self.descent

    def add(self, pt: PlacedToken):
        self.tokens.append(pt)
        # Update line extents to encompass new token
        self.ascent  = max(self.ascent,  pt.ascent)
        self.descent = max(self.descent, pt.descent)


@dataclass
class ImageBlock:
    """A reserved rectangular region for an image."""
    x:          float
    y:          float    # top of image block (page coords)
    width:      float
    height:     float
    src:        str
    align:      str      # "center" | "left" | "right"
    page:       int


@dataclass
class MarginNote:
    """A note placed in the right margin."""
    token:      Token
    x:          float
    baseline_y: float
    page:       int


@dataclass
class Page:
    index:        int
    lines:        List[Line]        = field(default_factory=list)
    image_blocks: List[ImageBlock]  = field(default_factory=list)
    margin_notes: List[MarginNote]  = field(default_factory=list)
    width:        float             = 210.0
    height:       float             = 297.0

    def add_line(self, line: Line):
        self.lines.append(line)

    @property
    def all_placed_tokens(self) -> List[PlacedToken]:
        return [pt for line in self.lines for pt in line.tokens]


@dataclass
class Document:
    pages:  List[Page]
    config: LayoutConfig

    @property
    def total_words(self) -> int:
        return sum(
            1 for page in self.pages
            for pt in page.all_placed_tokens
            if pt.token.type.name in ("WORD", "DIGIT", "SYMBOL")
        )

    @property
    def total_pages(self) -> int:
        return len(self.pages)
