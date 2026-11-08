"""
P6 Layout Engine
================
Takes a flat List[Token] from P5 and places every token onto
virtual A4 pages with exact mm positions.

Baseline convention (critical):
    Every word is placed by baseline_y (mm from page top).
    ML model strokes are in baseline=0 space.
    SVG composer does: abs_y = placed_token.baseline_y + stroke.y
    This means ascenders naturally go UP and descenders go DOWN
    without any per-word adjustment.

State tokens (STYLE, PEN, PEN_END) modify renderer state but
take no space and don't advance the cursor.
"""

import logging
from typing import List, Optional, Tuple

from ..p5_tokenizer.token_types import Token, TokenType
from .data_structures import (
    LayoutConfig, BBox, PlacedToken, Line, Page, Document,
    ImageBlock, MarginNote,
)
from .bbox_provider import BBoxProvider, MockBBoxProvider

log = logging.getLogger(__name__)

# Token types that take no space and only update state
STATE_TOKENS = frozenset({
    TokenType.STYLE, TokenType.PEN, TokenType.PEN_END,
})

# Token types that produce placed tokens with actual positions
CONTENT_TOKENS = frozenset({
    TokenType.WORD, TokenType.DIGIT, TokenType.SYMBOL,
    TokenType.GHOST, TokenType.CORRECTION,
})


class LayoutEngine:

    def __init__(self, config: LayoutConfig, bbox_provider: BBoxProvider):
        self.config = config
        self.bbox   = bbox_provider

    def layout(self, tokens: List[Token]) -> Document:
        """
        Main entry point. Returns a fully laid-out Document.
        """
        cfg = self.config
        pages: List[Page] = []

        # ── Cursor state ──────────────────────────────────────────────────────
        cur_x       = cfg.margin_left
        cur_baseline= cfg.margin_top + cfg.default_ascent  # first baseline
        page_idx    = 0
        line_idx    = 0
        indent_next = False   # indent the next line (after paragraph)

        # ── Renderer state (updated by state tokens) ──────────────────────────
        current_style = "casual"
        current_pen   = "black"

        # ── Working page / line ───────────────────────────────────────────────
        current_page = Page(index=0)
        current_line = Line(
            index      = 0,
            baseline_y = cur_baseline,
            ascent     = cfg.default_ascent,
            descent    = cfg.default_descent,
        )
        pages.append(current_page)

        def finish_line():
            """Commit current line to page, prepare next line."""
            nonlocal current_line, cur_x, cur_baseline, line_idx, indent_next

            if current_line.tokens:
                current_page.add_line(current_line)

            line_idx   += 1
            cur_baseline += current_line.line_height + cfg.leading
            cur_x        = cfg.margin_left
            if indent_next:
                cur_x      += cfg.indent
                indent_next = False

            current_line = Line(
                index      = line_idx,
                baseline_y = cur_baseline,
                ascent     = cfg.default_ascent,
                descent    = cfg.default_descent,
            )

        def finish_page():
            """Commit current page, start a new one."""
            nonlocal current_page, current_line, cur_x, cur_baseline
            nonlocal page_idx, line_idx, indent_next

            # Commit whatever line is in progress
            if current_line.tokens:
                current_page.add_line(current_line)

            page_idx    += 1
            line_idx     = 0
            cur_baseline = cfg.margin_top + cfg.default_ascent
            cur_x        = cfg.margin_left
            indent_next  = False

            current_page = Page(index=page_idx)
            current_line = Line(
                index      = 0,
                baseline_y = cur_baseline,
                ascent     = cfg.default_ascent,
                descent    = cfg.default_descent,
            )
            pages.append(current_page)

        def page_overflows() -> bool:
            """True if adding another line would exceed bottom margin."""
            next_baseline = cur_baseline + current_line.line_height + cfg.leading
            return next_baseline + cfg.default_descent > (
                cfg.page_height - cfg.margin_bottom
            )

        def place_content_token(token: Token) -> Optional[PlacedToken]:
            """
            Place a single content token at the current cursor.
            Returns PlacedToken, or None if token takes no space.
            Handles line wrapping and page overflow internally.
            """
            nonlocal cur_x, cur_baseline, indent_next

            bb = self.bbox.get_bbox(token)

            if bb.width == 0.0:
                return None

            # ── Word wrap ─────────────────────────────────────────────────────
            right_limit = cfg.page_width - cfg.margin_right
            if cur_x + bb.width > right_limit and current_line.tokens:
                # Doesn't fit — wrap to next line
                if page_overflows():
                    finish_line()
                    finish_page()
                else:
                    finish_line()
                    cur_baseline = current_line.baseline_y

            # ── Long word wider than full line — place anyway with warning ────
            if bb.width > cfg.usable_width:
                log.warning(
                    f"Token {token.value!r} width {bb.width:.1f}mm exceeds "
                    f"usable width {cfg.usable_width:.1f}mm — placing anyway"
                )

            # ── Build PlacedToken ─────────────────────────────────────────────
            pt = PlacedToken(
                token      = token,
                x          = cur_x,
                baseline_y = current_line.baseline_y,
                bbox       = bb,
                page       = page_idx,
                line       = line_idx,
                is_ghost   = (token.type == TokenType.GHOST),
                style      = current_style,
                pen        = current_pen,
            )

            current_line.add(pt)

            # ── Advance cursor ────────────────────────────────────────────────
            gap = (cfg.inter_digit_gap
                   if token.type == TokenType.DIGIT
                   else cfg.inter_word_gap)
            cur_x += bb.width + gap

            return pt

        # ── Main token loop ───────────────────────────────────────────────────

        for token in tokens:

            # ── State tokens — update renderer state, no space ────────────────
            if token.type in STATE_TOKENS:
                if token.type == TokenType.STYLE:
                    current_style = token.value or "casual"
                elif token.type == TokenType.PEN:
                    current_pen = token.value or "black"
                elif token.type == TokenType.PEN_END:
                    current_pen = "black"
                continue

            # ── Paragraph break ───────────────────────────────────────────────
            elif token.type == TokenType.PARAGRAPH:
                indent_next = cfg.indent_enabled   # set BEFORE finish_line consumes it
                if page_overflows():
                    finish_page()
                else:
                    finish_line()
                    # Extra paragraph gap
                    para_gap = current_line.line_height * (cfg.paragraph_gap - 1.0)
                    cur_baseline += para_gap
                    current_line.baseline_y = cur_baseline
                continue

            # ── Newline ───────────────────────────────────────────────────────
            elif token.type == TokenType.NEWLINE:
                if page_overflows():
                    finish_page()
                else:
                    finish_line()
                    cur_baseline = current_line.baseline_y
                continue

            # ── Force new page ────────────────────────────────────────────────
            elif token.type == TokenType.NEW_PAGE:
                finish_page()
                continue

            # ── Image block ───────────────────────────────────────────────────
            elif token.type == TokenType.IMAGE:
                block = self._place_image(
                    token, cur_x, cur_baseline, page_idx,
                    current_line
                )
                if block:
                    current_page.image_blocks.append(block)
                    # Image always acts as a block element — break line after
                    if page_overflows():
                        finish_page()
                    else:
                        finish_line()
                        cur_baseline += block.height + cfg.leading
                        current_line.baseline_y = cur_baseline
                continue

            # ── Margin note ───────────────────────────────────────────────────
            elif token.type == TokenType.MARGIN:
                note = MarginNote(
                    token      = token,
                    x          = cfg.page_width - cfg.margin_right + cfg.margin_note_x,
                    baseline_y = current_line.baseline_y,
                    page       = page_idx,
                )
                current_page.margin_notes.append(note)
                continue

            # ── Signature ─────────────────────────────────────────────────────
            elif token.type == TokenType.SIGNATURE:
                # Place at bottom right of current page, then close page
                sig_x = cfg.page_width - cfg.margin_right - 40.0  # ~40mm wide
                sig_y = cfg.page_height - cfg.margin_bottom
                pt = PlacedToken(
                    token      = token,
                    x          = sig_x,
                    baseline_y = sig_y,
                    bbox       = BBox(width=40.0, ascent=6.0, descent=2.0),
                    page       = page_idx,
                    line       = line_idx,
                    style      = current_style,
                    pen        = current_pen,
                )
                current_page.add_line(Line(
                    index      = line_idx,
                    baseline_y = sig_y,
                    ascent     = 6.0,
                    descent    = 2.0,
                    tokens     = [pt],
                ))
                finish_page()
                continue

            # ── Content tokens ────────────────────────────────────────────────
            elif token.type in CONTENT_TOKENS:
                place_content_token(token)
                continue

            else:
                log.warning(f"Layout engine received unhandled token type {token.type} — skipping")

        # ── Finalise last line and page ───────────────────────────────────────
        if current_line.tokens:
            current_page.add_line(current_line)

        # Remove trailing empty pages
        pages = [p for p in pages if p.lines or p.image_blocks]

        return Document(pages=pages, config=cfg)

    # ── Image placement ───────────────────────────────────────────────────────

    def _place_image(
        self,
        token:        Token,
        cur_x:        float,
        cur_baseline: float,
        page_idx:     int,
        current_line: Line,
    ) -> Optional[ImageBlock]:

        cfg   = self.config
        src   = token.value
        meta  = token.metadata

        # Parse width
        width_str = meta.get("width", f"{cfg.default_img_width}mm")
        try:
            width_mm = float(width_str.replace("mm", "").strip())
        except ValueError:
            log.warning(f"Invalid image width {width_str!r} — using default {cfg.default_img_width}mm")
            width_mm = cfg.default_img_width

        width_mm = min(width_mm, cfg.usable_width)

        # Aspect ratio — we don't know the image yet, assume 4:3 portrait
        # SVG composer will correct this when it loads the actual image
        height_mm = width_mm * 1.33

        align = meta.get("align", "center")

        if align == "center":
            x = cfg.margin_left + (cfg.usable_width - width_mm) / 2
        elif align == "right":
            x = cfg.page_width - cfg.margin_right - width_mm
        else:  # left
            x = cfg.margin_left

        # Image top sits at current baseline - ascent (top of current line)
        y_top = cur_baseline - current_line.ascent

        return ImageBlock(
            x      = x,
            y      = y_top,
            width  = width_mm,
            height = height_mm,
            src    = src,
            align  = align,
            page   = page_idx,
        )
