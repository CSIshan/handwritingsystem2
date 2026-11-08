"""
P5 Tokenizer
============
Converts a raw input string into an ordered list of typed Token objects.
Every downstream module (layout engine, inference engine, g-code emitter)
consumes this token list — never the raw string.

Slash markers:
    /p   — paragraph break
    /o   — force new page

Bracket tags:
    [GHOST: word or phrase]
    [IMAGE src="x.jpg" width="80mm" align="center|left|right"]
    [STYLE: neat|casual|tired]
    [PEN:red] ... [/PEN]
    [MARGIN: note text]
    [SIGNATURE]
    ~~word~~          — strikethrough correction, no rewrite
    ~~word~~rewrite   — strikethrough + replacement word
"""

import re
import uuid
import logging
from typing import List

from .token_types import Token, TokenType
from .rules import (
    RE_PARAGRAPH, RE_NEW_PAGE,
    RE_GHOST, RE_IMAGE, RE_STYLE, RE_PEN_OPEN, RE_PEN_CLOSE,
    RE_CORRECTION, RE_MARGIN, RE_SIGNATURE,
    RE_IMG_SRC, RE_IMG_WIDTH, RE_IMG_ALIGN,
    RE_DIGIT, RE_DIGIT_PREFIX,
    TRAILING_ATTACH, LEADING_ATTACH, CLOSING_ATTACH, STANDALONE_SYMBOLS,
    is_standalone_symbol,
)

log = logging.getLogger(__name__)


class TokenizerError(ValueError):
    pass


class Tokenizer:
    """
    Three-pass tokenizer:
      Pass 1 — extract bracket tags, replace with placeholders
      Pass 2 — split on whitespace, detect /p /o and paragraph breaks
      Pass 3 — classify each chunk into its final TokenType
    """

    def parse(self, text: str) -> List[Token]:
        if not isinstance(text, str):
            raise TokenizerError(f"Input must be a string, got {type(text)}")

        if not text.strip():
            return []

        # Pass 1: extract bracket tags → placeholders
        text, tag_map = self._extract_tags(text)

        # Pass 2: split into raw chunks
        chunks = self._split(text)

        # Pass 3: classify each chunk
        tokens = self._classify(chunks, tag_map)

        # Post-process: auto-close any unclosed PEN tokens
        tokens = self._close_open_pens(tokens)

        return tokens

    # ── Pass 1: Tag Extraction ────────────────────────────────────────────────

    def _extract_tags(self, text: str):
        """
        Replace all bracket tags with unique placeholders like §TAG_uuid§.
        Returns modified text and a dict mapping placeholder → Token.
        """
        tag_map = {}

        def placeholder(token: Token, raw: str) -> str:
            key = f"§{uuid.uuid4().hex}§"
            token.raw = raw
            tag_map[key] = token
            return f" {key} "

        # Order matters — more specific patterns first

        # [SIGNATURE]
        text = RE_SIGNATURE.sub(
            lambda m: placeholder(Token(TokenType.SIGNATURE, None, {}, m.group(0)), m.group(0)),
            text
        )

        # [GHOST: phrase]
        def _ghost(m):
            phrase = m.group(1).strip()
            subwords = phrase.split()
            return placeholder(
                Token(TokenType.GHOST, phrase, {"subwords": subwords}),
                m.group(0)
            )
        text = RE_GHOST.sub(_ghost, text)

        # [IMAGE src="..." width="..." align="..."]
        def _image(m):
            attrs_str = m.group(1)
            src   = (RE_IMG_SRC.search(attrs_str)   or type('', (), {'group': lambda s,n: None})()).group(1)
            width = (RE_IMG_WIDTH.search(attrs_str)  or type('', (), {'group': lambda s,n: "80mm"})()).group(1)
            align = (RE_IMG_ALIGN.search(attrs_str)  or type('', (), {'group': lambda s,n: "center"})()).group(1)
            if not src:
                log.warning(f"IMAGE tag missing src attribute: {m.group(0)!r} — skipping")
                return " "
            return placeholder(
                Token(TokenType.IMAGE, src, {"width": width or "80mm", "align": align or "center"}),
                m.group(0)
            )
        text = RE_IMAGE.sub(_image, text)

        # [STYLE: value]
        text = RE_STYLE.sub(
            lambda m: placeholder(
                Token(TokenType.STYLE, m.group(1).strip().lower(), {}),
                m.group(0)
            ),
            text
        )

        # [/PEN] — must come before [PEN:x]
        text = RE_PEN_CLOSE.sub(
            lambda m: placeholder(Token(TokenType.PEN_END, None, {}), m.group(0)),
            text
        )

        # [PEN:color]
        text = RE_PEN_OPEN.sub(
            lambda m: placeholder(
                Token(TokenType.PEN, m.group(1).strip().lower(), {}),
                m.group(0)
            ),
            text
        )

        # [MARGIN: text]
        text = RE_MARGIN.sub(
            lambda m: placeholder(
                Token(TokenType.MARGIN, m.group(1).strip(), {}),
                m.group(0)
            ),
            text
        )

        # ~~word~~rewrite or ~~word~~
        def _correction(m):
            struck  = m.group(1).strip()
            rewrite = m.group(2).strip() if m.group(2) else None
            return placeholder(
                Token(TokenType.CORRECTION, struck, {"rewrite": rewrite}),
                m.group(0)
            )
        text = RE_CORRECTION.sub(_correction, text)

        # Warn about any remaining unrecognised bracket tags
        remaining = re.findall(r'\[[^\]]{1,80}\]', text)
        for r in remaining:
            log.warning(f"Unrecognised tag {r!r} — treating contents as plain text, removing brackets")
            text = text.replace(r, r[1:-1])  # strip the brackets, keep inner text

        return text, tag_map

    # ── Pass 2: Splitting ─────────────────────────────────────────────────────

    def _split(self, text: str) -> List[str]:
        """
        Split text into chunks, preserving paragraph breaks (double newline)
        and single newlines. Returns a flat list of string chunks.
        """
        chunks = []

        # Normalise line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Split on double newlines first to find paragraph breaks
        # Then split each paragraph's content on whitespace
        # We need to preserve the boundary information

        # Strategy: replace \n\n with a unique paragraph sentinel, then split
        PARA_SENTINEL  = '\x00PARA\x00'
        LINE_SENTINEL  = '\x00LINE\x00'

        text = re.sub(r'\n{2,}', PARA_SENTINEL, text)
        text = text.replace('\n', LINE_SENTINEL)

        # Now split on whitespace (spaces, tabs) — NOT on sentinels
        raw_chunks = re.split(r'[ \t]+', text)

        for chunk in raw_chunks:
            if not chunk:
                continue

            # A chunk might contain sentinels embedded e.g. "word\x00PARA\x00word2"
            # Split on sentinels and emit sentinel tokens between words
            parts = re.split(f'({re.escape(PARA_SENTINEL)}|{re.escape(LINE_SENTINEL)})', chunk)
            for part in parts:
                if part == PARA_SENTINEL:
                    chunks.append('\x00PARA\x00')
                elif part == LINE_SENTINEL:
                    chunks.append('\x00LINE\x00')
                elif part:
                    chunks.append(part)

        return chunks

    # ── Pass 3: Classification ────────────────────────────────────────────────

    def _classify(self, chunks: List[str], tag_map: dict) -> List[Token]:
        tokens = []

        for chunk in chunks:

            # Sentinel: paragraph break
            if chunk == '\x00PARA\x00':
                tokens.append(Token(TokenType.PARAGRAPH, None, {}, '\n\n'))
                continue

            # Sentinel: single newline
            if chunk == '\x00LINE\x00':
                tokens.append(Token(TokenType.NEWLINE, None, {}, '\n'))
                continue

            # Placeholder from tag extraction
            if chunk in tag_map:
                tokens.append(tag_map[chunk])
                continue

            # Slash markers — must be exact whole tokens
            if RE_PARAGRAPH.match(chunk):
                tokens.append(Token(TokenType.PARAGRAPH, None, {}, chunk))
                continue

            if RE_NEW_PAGE.match(chunk):
                tokens.append(Token(TokenType.NEW_PAGE, None, {}, chunk))
                continue

            # Digit prefix split: "42nd" → DIGIT("42") + WORD("nd")
            dp = RE_DIGIT_PREFIX.match(chunk)
            if dp:
                tokens.append(Token(TokenType.DIGIT,  dp.group(1), {}, dp.group(1)))
                tokens.append(Token(TokenType.WORD,   dp.group(2), {}, dp.group(2)))
                continue

            # Pure digit / number
            if RE_DIGIT.match(chunk):
                tokens.append(Token(TokenType.DIGIT, chunk, {}, chunk))
                continue

            # Single standalone symbol
            if is_standalone_symbol(chunk):
                tokens.append(Token(TokenType.SYMBOL, chunk, {}, chunk))
                continue

            # Everything else is a WORD (includes attached punctuation)
            tokens.append(Token(TokenType.WORD, chunk, {}, chunk))

        return tokens

    # ── Post-processing ───────────────────────────────────────────────────────

    def _close_open_pens(self, tokens: List[Token]) -> List[Token]:
        """
        If there's a PEN_OPEN without a matching PEN_END, auto-insert
        PEN_END at the end of the token list and log a warning.
        """
        open_count  = sum(1 for t in tokens if t.type == TokenType.PEN)
        close_count = sum(1 for t in tokens if t.type == TokenType.PEN_END)

        if open_count > close_count:
            diff = open_count - close_count
            log.warning(f"{diff} unclosed [PEN:...] tag(s) — auto-closing at end of input")
            for _ in range(diff):
                tokens.append(Token(TokenType.PEN_END, None, {}, "[/PEN auto-inserted]"))

        elif close_count > open_count:
            log.warning(f"{close_count - open_count} extra [/PEN] tag(s) with no matching open — ignoring extras")
            # Remove the excess PEN_END tokens from the front
            excess = close_count - open_count
            cleaned = []
            removed = 0
            for t in tokens:
                if t.type == TokenType.PEN_END and removed < excess:
                    removed += 1
                    continue
                cleaned.append(t)
            return cleaned

        return tokens
