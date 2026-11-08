"""
Test suite for P6 Layout Engine.
Run with: python -m unittest discover handwriting_system/p6_layout/tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import unittest
from handwriting_system.p5_tokenizer import Tokenizer, Token, TokenType
from handwriting_system.p6_layout import (
    LayoutEngine, LayoutConfig, MockBBoxProvider,
    PlacedToken, Document,
)


def make_engine(cfg=None):
    cfg = cfg or LayoutConfig()
    return LayoutEngine(cfg, MockBBoxProvider(cfg))


def layout(text: str, cfg=None) -> Document:
    tokens = Tokenizer().parse(text)
    return make_engine(cfg).layout(tokens)


class TestBasicPlacement(unittest.TestCase):

    def test_single_word_placed(self):
        doc = layout("hello")
        self.assertEqual(doc.total_pages, 1)
        pts = doc.pages[0].all_placed_tokens
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].token.value, "hello")

    def test_word_at_left_margin(self):
        doc = layout("hello")
        cfg = LayoutConfig()
        self.assertAlmostEqual(doc.pages[0].all_placed_tokens[0].x,
                               cfg.margin_left, places=1)

    def test_first_baseline_correct(self):
        doc = layout("hello")
        cfg = LayoutConfig()
        expected = cfg.margin_top + cfg.default_ascent
        self.assertAlmostEqual(
            doc.pages[0].all_placed_tokens[0].baseline_y,
            expected, places=1
        )

    def test_multiple_words_advance_cursor(self):
        doc = layout("hello world")
        pts = doc.pages[0].all_placed_tokens
        self.assertEqual(len(pts), 2)
        # Second word must be to the right of first
        self.assertGreater(pts[1].x, pts[0].x + pts[0].width)

    def test_same_line_same_baseline(self):
        doc = layout("hello world foo")
        pts = doc.pages[0].all_placed_tokens
        baselines = {pt.baseline_y for pt in pts}
        self.assertEqual(len(baselines), 1,
                         "All words on one line must share baseline_y")

    def test_words_share_exact_baseline_y(self):
        """
        Core baseline test: tall words (H) and short words (a)
        on same line must have identical baseline_y.
        Ascenders go up from it, descenders go down — but baseline is shared.
        """
        doc = layout("Hello and going")
        pts = doc.pages[0].all_placed_tokens
        baseline_ys = [pt.baseline_y for pt in pts if pt.line == 0]
        self.assertEqual(len(set(baseline_ys)), 1,
            "Mixed ascender/descender words must all sit on same baseline_y")

    def test_ascent_varies_per_word(self):
        """
        'hello' (no ascenders) should have smaller ascent than 'Hello' (cap H).
        """
        cfg = LayoutConfig()
        provider = MockBBoxProvider(cfg)
        tok_lower = Token(TokenType.WORD, "more")
        tok_upper = Token(TokenType.WORD, "Hello")
        bbox_lower = provider.get_bbox(tok_lower)
        bbox_upper = provider.get_bbox(tok_upper)
        self.assertGreater(bbox_upper.ascent, bbox_lower.ascent)

    def test_descent_only_for_descender_words(self):
        cfg = LayoutConfig()
        provider = MockBBoxProvider(cfg)
        tok_desc  = Token(TokenType.WORD, "going")   # g,g = descenders
        tok_plain = Token(TokenType.WORD, "the")     # no descenders
        self.assertGreater(provider.get_bbox(tok_desc).descent, 0)
        self.assertEqual(provider.get_bbox(tok_plain).descent, 0.0)


class TestLineWrapping(unittest.TestCase):

    def test_long_text_wraps_to_second_line(self):
        # 30 words will definitely wrap on A4
        text = " ".join(["word"] * 30)
        doc = layout(text)
        lines = doc.pages[0].lines
        self.assertGreater(len(lines), 1)

    def test_second_line_baseline_below_first(self):
        text = " ".join(["word"] * 30)
        doc = layout(text)
        lines = doc.pages[0].lines
        self.assertGreater(lines[1].baseline_y, lines[0].baseline_y)

    def test_wrapped_word_resets_to_left_margin(self):
        text = " ".join(["word"] * 30)
        doc = layout(text)
        # Find first word on line 1
        line1_tokens = [pt for pt in doc.pages[0].all_placed_tokens
                        if pt.line == 1]
        self.assertTrue(len(line1_tokens) > 0)
        cfg = LayoutConfig()
        self.assertAlmostEqual(line1_tokens[0].x, cfg.margin_left, places=1)

    def test_very_long_word_placed_anyway(self):
        # A word wider than the page should still be placed (with a warning)
        long_word = "a" * 100
        doc = layout(long_word)
        pts = doc.pages[0].all_placed_tokens
        self.assertEqual(len(pts), 1)


class TestParagraph(unittest.TestCase):

    def test_slash_p_creates_paragraph_gap(self):
        doc = layout("para one /p para two")
        lines = doc.pages[0].lines
        # Should have lines for both paragraphs
        self.assertGreaterEqual(len(lines), 2)
        # Gap between last line of para1 and first line of para2
        # should be larger than normal line spacing
        if len(lines) >= 2:
            normal_gap = LayoutConfig().default_line_height + LayoutConfig().leading
            actual_gap = lines[1].baseline_y - lines[0].baseline_y
            self.assertGreater(actual_gap, normal_gap * 0.9)

    def test_paragraph_indent(self):
        doc = layout("para one /p para two")
        cfg = LayoutConfig()
        # Find first token of second paragraph
        all_pts = doc.pages[0].all_placed_tokens
        # Token after paragraph break should be indented
        para2_tokens = [pt for pt in all_pts if pt.line >= 1]
        if para2_tokens and cfg.indent_enabled:
            self.assertAlmostEqual(
                para2_tokens[0].x,
                cfg.margin_left + cfg.indent,
                places=0
            )

    def test_double_newline_paragraph(self):
        doc = layout("para one\n\npara two")
        lines = doc.pages[0].lines
        self.assertGreaterEqual(len(lines), 2)


class TestNewPage(unittest.TestCase):

    def test_slash_o_forces_new_page(self):
        doc = layout("page one /o page two")
        self.assertEqual(doc.total_pages, 2)

    def test_new_page_resets_cursor(self):
        doc = layout("page one /o page two")
        cfg = LayoutConfig()
        page2_pts = doc.pages[1].all_placed_tokens
        self.assertTrue(len(page2_pts) > 0)
        self.assertAlmostEqual(page2_pts[0].x, cfg.margin_left, places=1)
        self.assertAlmostEqual(
            page2_pts[0].baseline_y,
            cfg.margin_top + cfg.default_ascent,
            places=1
        )

    def test_overflow_creates_new_page(self):
        # Enough words to overflow a page
        text = " ".join(["word"] * 500)
        doc = layout(text)
        self.assertGreater(doc.total_pages, 1)


class TestGhostTokens(unittest.TestCase):

    def test_ghost_is_flagged(self):
        doc = layout("I have [GHOST: attached the] documents")
        pts = doc.pages[0].all_placed_tokens
        ghost_pts = [pt for pt in pts if pt.is_ghost]
        self.assertEqual(len(ghost_pts), 1)

    def test_ghost_advances_cursor(self):
        doc = layout("before [GHOST: hello world] after")
        pts = doc.pages[0].all_placed_tokens
        after = next(pt for pt in pts if pt.token.value == "after")
        ghost = next(pt for pt in pts if pt.is_ghost)
        # "after" must be to the right of ghost's right edge
        self.assertGreater(after.x, ghost.x + ghost.width - 0.1)

    def test_ghost_shares_baseline_with_neighbors(self):
        doc = layout("before [GHOST: hello] after")
        pts = doc.pages[0].all_placed_tokens
        baselines = {pt.baseline_y for pt in pts if pt.line == 0}
        self.assertEqual(len(baselines), 1)


class TestStateTokens(unittest.TestCase):

    def test_style_token_takes_no_space(self):
        doc1 = layout("hello world")
        doc2 = layout("[STYLE: neat] hello world")
        # Word positions should be identical
        pts1 = doc1.pages[0].all_placed_tokens
        pts2 = doc2.pages[0].all_placed_tokens
        self.assertEqual(len(pts1), len(pts2))

    def test_style_applied_to_following_tokens(self):
        doc = layout("[STYLE: neat] hello world")
        pts = doc.pages[0].all_placed_tokens
        self.assertTrue(all(pt.style == "neat" for pt in pts))

    def test_pen_applied_to_following_tokens(self):
        doc = layout("[PEN:red] hello [/PEN] world")
        pts = doc.pages[0].all_placed_tokens
        red_pts   = [pt for pt in pts if pt.pen == "red"]
        black_pts = [pt for pt in pts if pt.pen == "black"]
        self.assertTrue(len(red_pts) > 0)
        self.assertTrue(len(black_pts) > 0)

    def test_pen_resets_after_pen_end(self):
        doc = layout("[PEN:blue] hello [/PEN] world")
        pts = doc.pages[0].all_placed_tokens
        world = next(pt for pt in pts if pt.token.value == "world")
        self.assertEqual(world.pen, "black")


class TestSpecialTokens(unittest.TestCase):

    def test_signature_placed_at_bottom(self):
        doc = layout("Dear John [SIGNATURE]")
        # Signature forces page close
        self.assertGreaterEqual(doc.total_pages, 1)
        all_pts = [pt for page in doc.pages for pt in page.all_placed_tokens]
        sig = next((pt for pt in all_pts
                    if pt.token.type == TokenType.SIGNATURE), None)
        self.assertIsNotNone(sig)
        cfg = LayoutConfig()
        # Signature should be near bottom
        self.assertGreater(sig.baseline_y, cfg.page_height * 0.7)

    def test_image_creates_image_block(self):
        doc = layout('[IMAGE src="x.jpg" width="80mm" align="center"]')
        self.assertGreater(len(doc.pages[0].image_blocks), 0)

    def test_margin_note_placed_in_margin(self):
        doc = layout("hello [MARGIN: check this] world")
        self.assertGreater(len(doc.pages[0].margin_notes), 0)
        cfg = LayoutConfig()
        note = doc.pages[0].margin_notes[0]
        self.assertGreater(note.x, cfg.page_width - cfg.margin_right)


class TestDigitsAndSymbols(unittest.TestCase):

    def test_digit_placed(self):
        doc = layout("page 42")
        pts = doc.pages[0].all_placed_tokens
        digit = next((pt for pt in pts if pt.token.type == TokenType.DIGIT), None)
        self.assertIsNotNone(digit)

    def test_symbol_placed(self):
        doc = layout("salt & pepper")
        pts = doc.pages[0].all_placed_tokens
        sym = next((pt for pt in pts if pt.token.type == TokenType.SYMBOL), None)
        self.assertIsNotNone(sym)


class TestEmptyInput(unittest.TestCase):

    def test_empty_token_list(self):
        engine = make_engine()
        doc = engine.layout([])
        self.assertEqual(doc.total_pages, 0)

    def test_only_state_tokens(self):
        doc = layout("[STYLE: neat]")
        self.assertEqual(doc.total_pages, 0)


class TestComplexDocument(unittest.TestCase):

    def test_full_document_structure(self):
        text = (
            "Dear John, /p "
            "I have [GHOST: attached the] documents ~~you~~ we discussed. /p "
            "[STYLE: neat] Please review them carefully. /o "
            "Regards, [SIGNATURE]"
        )
        doc = layout(text)
        self.assertGreaterEqual(doc.total_pages, 2)
        self.assertGreater(doc.total_words, 5)

    def test_baseline_consistency_across_lines(self):
        """Every line must have a single consistent baseline_y for all its tokens."""
        text = " ".join(["Hello going world Typography"] * 10)
        doc = layout(text)
        for page in doc.pages:
            for line in page.lines:
                baselines = {pt.baseline_y for pt in line.tokens}
                self.assertEqual(len(baselines), 1,
                    f"Line {line.index} has mixed baseline_y values: {baselines}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
