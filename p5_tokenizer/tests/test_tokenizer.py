"""
Test suite for P5 Tokenizer.
Run with: python -m pytest p5_tokenizer/tests/ -v
      or: python -m unittest discover p5_tokenizer/tests/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from p5_tokenizer import Tokenizer, Token, TokenType


class TestBasicWordSplitting(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_single_word(self):
        tokens = self.t.parse("hello")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.WORD)
        self.assertEqual(tokens[0].value, "hello")

    def test_multiple_words(self):
        tokens = self.t.parse("the quick brown fox")
        words = [t.value for t in tokens]
        self.assertEqual(words, ["the", "quick", "brown", "fox"])
        self.assertTrue(all(t.type == TokenType.WORD for t in tokens))

    def test_extra_spaces_collapsed(self):
        tokens = self.t.parse("hello   world")
        self.assertEqual(len(tokens), 2)

    def test_tabs_treated_as_spaces(self):
        tokens = self.t.parse("hello\tworld")
        self.assertEqual(len(tokens), 2)

    def test_empty_string(self):
        tokens = self.t.parse("")
        self.assertEqual(tokens, [])

    def test_whitespace_only(self):
        tokens = self.t.parse("   \t  ")
        self.assertEqual(tokens, [])

    def test_attached_trailing_comma(self):
        tokens = self.t.parse("hello, world")
        self.assertEqual(tokens[0].value, "hello,")
        self.assertEqual(tokens[0].type, TokenType.WORD)

    def test_attached_trailing_period(self):
        tokens = self.t.parse("end.")
        self.assertEqual(tokens[0].value, "end.")

    def test_attached_question_mark(self):
        tokens = self.t.parse("really?")
        self.assertEqual(tokens[0].value, "really?")

    def test_apostrophe_internal(self):
        tokens = self.t.parse("don't")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].value, "don't")

    def test_multiple_sentences(self):
        tokens = self.t.parse("Hello world. How are you?")
        values = [t.value for t in tokens]
        self.assertIn("world.", values)
        self.assertIn("you?", values)


class TestParagraphAndNewline(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_double_newline_paragraph(self):
        tokens = self.t.parse("hello\n\nworld")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.PARAGRAPH, types)

    def test_slash_p_paragraph(self):
        tokens = self.t.parse("hello /p world")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.PARAGRAPH, types)

    def test_slash_p_case_insensitive(self):
        tokens = self.t.parse("hello /P world")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.PARAGRAPH, types)

    def test_single_newline(self):
        tokens = self.t.parse("hello\nworld")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.NEWLINE, types)

    def test_new_page_slash_o(self):
        tokens = self.t.parse("page one /o page two")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.NEW_PAGE, types)

    def test_slash_o_case_insensitive(self):
        tokens = self.t.parse("page one /O page two")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.NEW_PAGE, types)

    def test_slash_p_not_inside_word(self):
        # "/p" only triggers when it's a standalone token
        tokens = self.t.parse("stop/p")
        # "stop/p" is one chunk, should be treated as WORD not PARAGRAPH
        types = [t.type for t in tokens]
        self.assertNotIn(TokenType.PARAGRAPH, types)


class TestGhostTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_basic_ghost(self):
        tokens = self.t.parse("[GHOST: quick brown]")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.GHOST)
        self.assertEqual(tokens[0].value, "quick brown")

    def test_ghost_subwords(self):
        tokens = self.t.parse("[GHOST: hello world]")
        self.assertEqual(tokens[0].metadata["subwords"], ["hello", "world"])

    def test_ghost_single_word(self):
        tokens = self.t.parse("[GHOST: hello]")
        self.assertEqual(tokens[0].metadata["subwords"], ["hello"])

    def test_ghost_case_insensitive(self):
        tokens = self.t.parse("[ghost: hello]")
        self.assertEqual(tokens[0].type, TokenType.GHOST)

    def test_ghost_inline(self):
        tokens = self.t.parse("I have [GHOST: attached the] documents")
        self.assertEqual(tokens[0].value, "I")
        self.assertEqual(tokens[1].value, "have")
        self.assertEqual(tokens[2].type, TokenType.GHOST)
        self.assertEqual(tokens[3].value, "documents")


class TestImageTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_basic_image(self):
        tokens = self.t.parse('[IMAGE src="chart.jpg" width="80mm" align="center"]')
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.IMAGE)
        self.assertEqual(tokens[0].value, "chart.jpg")

    def test_image_metadata(self):
        tokens = self.t.parse('[IMAGE src="chart.jpg" width="80mm" align="center"]')
        meta = tokens[0].metadata
        self.assertEqual(meta["width"], "80mm")
        self.assertEqual(meta["align"], "center")

    def test_image_default_align(self):
        tokens = self.t.parse('[IMAGE src="x.jpg" width="60mm"]')
        self.assertEqual(tokens[0].metadata["align"], "center")

    def test_image_default_width(self):
        tokens = self.t.parse('[IMAGE src="x.jpg"]')
        self.assertEqual(tokens[0].metadata["width"], "80mm")

    def test_image_missing_src_skipped(self):
        # Should log warning and skip, not crash
        tokens = self.t.parse('[IMAGE width="80mm"]')
        image_tokens = [t for t in tokens if t.type == TokenType.IMAGE]
        self.assertEqual(len(image_tokens), 0)


class TestStyleToken(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_style_neat(self):
        tokens = self.t.parse("[STYLE: neat]")
        self.assertEqual(tokens[0].type, TokenType.STYLE)
        self.assertEqual(tokens[0].value, "neat")

    def test_style_casual(self):
        tokens = self.t.parse("[STYLE: casual]")
        self.assertEqual(tokens[0].value, "casual")

    def test_style_tired(self):
        tokens = self.t.parse("[STYLE: tired]")
        self.assertEqual(tokens[0].value, "tired")

    def test_style_lowercased(self):
        tokens = self.t.parse("[STYLE: NEAT]")
        self.assertEqual(tokens[0].value, "neat")


class TestPenTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_pen_open(self):
        tokens = self.t.parse("[PEN:red] hello [/PEN]")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.PEN, types)
        self.assertIn(TokenType.PEN_END, types)

    def test_pen_color(self):
        tokens = self.t.parse("[PEN:blue] word [/PEN]")
        pen_token = next(t for t in tokens if t.type == TokenType.PEN)
        self.assertEqual(pen_token.value, "blue")

    def test_unclosed_pen_auto_closed(self):
        tokens = self.t.parse("[PEN:red] hello")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.PEN_END, types)

    def test_extra_pen_close_ignored(self):
        # More closes than opens — extras silently dropped
        tokens = self.t.parse("hello [/PEN] world")
        pen_ends = [t for t in tokens if t.type == TokenType.PEN_END]
        pen_opens = [t for t in tokens if t.type == TokenType.PEN]
        self.assertEqual(len(pen_ends), len(pen_opens))


class TestCorrectionTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_correction_no_rewrite(self):
        tokens = self.t.parse("~~word~~")
        self.assertEqual(tokens[0].type, TokenType.CORRECTION)
        self.assertEqual(tokens[0].value, "word")
        self.assertIsNone(tokens[0].metadata["rewrite"])

    def test_correction_with_rewrite(self):
        tokens = self.t.parse("~~you~~we")
        self.assertEqual(tokens[0].type, TokenType.CORRECTION)
        self.assertEqual(tokens[0].value, "you")
        self.assertEqual(tokens[0].metadata["rewrite"], "we")

    def test_correction_inline(self):
        tokens = self.t.parse("documents ~~you~~ we discussed")
        corr = next(t for t in tokens if t.type == TokenType.CORRECTION)
        self.assertEqual(corr.value, "you")


class TestDigitTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_pure_integer(self):
        tokens = self.t.parse("42")
        self.assertEqual(tokens[0].type, TokenType.DIGIT)
        self.assertEqual(tokens[0].value, "42")

    def test_decimal(self):
        tokens = self.t.parse("3.14")
        self.assertEqual(tokens[0].type, TokenType.DIGIT)

    def test_currency(self):
        tokens = self.t.parse("$99")
        self.assertEqual(tokens[0].type, TokenType.DIGIT)

    def test_digit_word_split(self):
        # "42nd" should become DIGIT("42") + WORD("nd")
        tokens = self.t.parse("42nd")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].type, TokenType.DIGIT)
        self.assertEqual(tokens[0].value, "42")
        self.assertEqual(tokens[1].type, TokenType.WORD)
        self.assertEqual(tokens[1].value, "nd")

    def test_ordinals(self):
        for word, digit, suffix in [("1st","1","st"),("2nd","2","nd"),("3rd","3","rd")]:
            tokens = self.t.parse(word)
            self.assertEqual(tokens[0].value, digit)
            self.assertEqual(tokens[1].value, suffix)


class TestSymbolTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_at_symbol(self):
        tokens = self.t.parse("@")
        self.assertEqual(tokens[0].type, TokenType.SYMBOL)

    def test_ampersand(self):
        tokens = self.t.parse("&")
        self.assertEqual(tokens[0].type, TokenType.SYMBOL)

    def test_symbol_between_words(self):
        tokens = self.t.parse("salt & pepper")
        types = [t.type for t in tokens]
        self.assertIn(TokenType.SYMBOL, types)


class TestSpecialTokens(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_signature(self):
        tokens = self.t.parse("[SIGNATURE]")
        self.assertEqual(tokens[0].type, TokenType.SIGNATURE)

    def test_margin(self):
        tokens = self.t.parse("[MARGIN: check this]")
        self.assertEqual(tokens[0].type, TokenType.MARGIN)
        self.assertEqual(tokens[0].value, "check this")


class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.t = Tokenizer()

    def test_unknown_tag_treated_as_text(self):
        # Unknown tags: brackets stripped, content kept as words
        tokens = self.t.parse("[UNKNOWN: something]")
        # Should not crash, should produce word tokens from inner text
        self.assertGreater(len(tokens), 0)
        self.assertFalse(any(t.type == TokenType.IMAGE for t in tokens))

    def test_unclosed_tilde_not_correction(self):
        tokens = self.t.parse("~~unclosed")
        # No correction token — just a word
        self.assertFalse(any(t.type == TokenType.CORRECTION for t in tokens))

    def test_very_long_word(self):
        long = "a" * 200
        tokens = self.t.parse(long)
        self.assertEqual(tokens[0].value, long)

    def test_mixed_complex_input(self):
        text = (
            "Dear John, [STYLE: neat] I have [GHOST: attached the]\n"
            "documents ~~you~~ we discussed. /p "
            '[IMAGE src="chart.jpg" width="80mm" align="center"] /o '
            "Regards, [SIGNATURE]"
        )
        tokens = self.t.parse(text)
        types = [t.type for t in tokens]

        self.assertIn(TokenType.STYLE,     types)
        self.assertIn(TokenType.GHOST,     types)
        self.assertIn(TokenType.NEWLINE,   types)
        self.assertIn(TokenType.CORRECTION,types)
        self.assertIn(TokenType.PARAGRAPH, types)
        self.assertIn(TokenType.IMAGE,     types)
        self.assertIn(TokenType.NEW_PAGE,  types)
        self.assertIn(TokenType.SIGNATURE, types)
        self.assertIn(TokenType.WORD,      types)

    def test_non_string_input_raises(self):
        with self.assertRaises(Exception):
            self.t.parse(12345)

    def test_multiple_paragraphs(self):
        text = "para one\n\npara two\n\npara three"
        tokens = self.t.parse(text)
        para_count = sum(1 for t in tokens if t.type == TokenType.PARAGRAPH)
        self.assertEqual(para_count, 2)

    def test_slash_p_and_double_newline_both_work(self):
        t1 = self.t.parse("a /p b")
        t2 = self.t.parse("a\n\nb")
        types1 = [t.type for t in t1]
        types2 = [t.type for t in t2]
        self.assertIn(TokenType.PARAGRAPH, types1)
        self.assertIn(TokenType.PARAGRAPH, types2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
