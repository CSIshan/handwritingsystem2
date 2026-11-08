"""
P4 Test Suite
=============
Tests engine interface, bbox caching, unknown char handling,
stroke structure, and preview SVG output.
Torch-dependent tests skip gracefully if torch not installed.
"""
import sys, os, json, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from handwriting_system.p4_inference.engine  import StrokePoint
from handwriting_system.p4_inference.preview import _strokes_to_path, _build_svg

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SKIP_TORCH = unittest.skipUnless(HAS_TORCH, "PyTorch not installed")


# ── StrokePoint tests (no torch) ──────────────────────────────────────────────

class TestStrokePoint(unittest.TestCase):

    def test_attributes(self):
        sp = StrokePoint(x=1.5, y=0.3, pen_up=0)
        self.assertEqual(sp.x,      1.5)
        self.assertEqual(sp.y,      0.3)
        self.assertEqual(sp.pen_up, 0)

    def test_pen_up_values(self):
        sp0 = StrokePoint(x=0, y=0, pen_up=0)
        sp1 = StrokePoint(x=0, y=0, pen_up=1)
        self.assertIn(sp0.pen_up, [0, 1])
        self.assertIn(sp1.pen_up, [0, 1])

    def test_baseline_convention(self):
        # y > 0 = above baseline (ascender)
        # y < 0 = below baseline (descender)
        ascender  = StrokePoint(x=0, y=5.0,  pen_up=0)
        descender = StrokePoint(x=0, y=-2.0, pen_up=0)
        self.assertGreater(ascender.y,  0)
        self.assertLess(descender.y, 0)


# ── Preview tests (no torch) ─────────────────────────────────────────────────

class TestPreview(unittest.TestCase):

    def _make_strokes(self):
        return [
            StrokePoint(0.0, 0.0,  0),
            StrokePoint(2.0, 3.0,  0),
            StrokePoint(4.0, 0.0,  0),
            StrokePoint(4.0, 0.0,  1),
            StrokePoint(5.0, 2.0,  0),
            StrokePoint(7.0, 0.0,  1),
        ]

    def test_strokes_to_path_contains_M(self):
        strokes = self._make_strokes()
        path = _strokes_to_path(strokes, x_off=40, y_off=150)
        self.assertIn("M", path)

    def test_strokes_to_path_contains_L(self):
        strokes = self._make_strokes()
        path = _strokes_to_path(strokes, x_off=40, y_off=150)
        self.assertIn("L", path)

    def test_pen_up_creates_new_M(self):
        strokes = self._make_strokes()
        path = _strokes_to_path(strokes, x_off=40, y_off=150)
        # Two pen-up points → two M commands
        self.assertEqual(path.count("M"), 2)

    def test_empty_strokes_returns_empty(self):
        path = _strokes_to_path([], x_off=40, y_off=150)
        self.assertEqual(path, "")

    def test_build_svg_is_valid_xml_start(self):
        svg = _build_svg(800, 300, [], [], "test")
        self.assertTrue(svg.strip().startswith("<svg"))
        self.assertTrue(svg.strip().endswith("</svg>"))

    def test_build_svg_contains_title(self):
        svg = _build_svg(800, 300, [], [], "hello world")
        self.assertIn("hello world", svg)

    def test_preview_word_creates_file(self):
        # Mock engine for preview test
        class MockEngine:
            def generate(self, word, temperature=1.0):
                return [
                    StrokePoint(float(i)*0.5, float(i%3)*0.3, i%5==4)
                    for i in range(30)
                ]
            def get_bbox(self, word):
                return (15.0, 5.0, 1.5)

        from handwriting_system.p4_inference.preview import preview_word
        with tempfile.TemporaryDirectory() as d:
            out = preview_word(MockEngine(), "hello", f"{d}/test.svg",
                               n_variants=2)
            import os
            self.assertTrue(os.path.exists(out))
            content = open(out).read()
            self.assertIn("<svg", content)
            self.assertIn("</svg>", content)

    def test_preview_words_creates_file(self):
        class MockEngine:
            def generate(self, word, temperature=1.0):
                return [
                    StrokePoint(float(i)*0.5, 0.0, i%8==7)
                    for i in range(20)
                ]
            def get_bbox(self, word):
                return (10.0, 5.0, 1.0)

        from handwriting_system.p4_inference.preview import preview_words
        with tempfile.TemporaryDirectory() as d:
            out = preview_words(MockEngine(), ["hello", "world"],
                                f"{d}/test.svg")
            self.assertTrue(os.path.exists(out))


# ── Torch-dependent tests ─────────────────────────────────────────────────────

@SKIP_TORCH
class TestInferenceEngine(unittest.TestCase):
    """
    These tests require a real trained checkpoint.
    On M2 after training, point CKPT_PATH to checkpoint_best.pt.
    They're structured to run against a mock model for CI purposes.
    """

    def _make_mock_engine(self):
        """Create engine with tiny untrained model for structural tests."""
        from handwriting_system.p3_training.model  import HandwritingModel
        from handwriting_system.p3_training.config import TrainConfig
        from handwriting_system.p3_training.prep   import build_vocab
        from handwriting_system.p4_inference.engine import InferenceEngine

        cfg   = TrainConfig(
            embed_dim=16, encoder_hidden=32, decoder_hidden=64,
            encoder_layers=1, decoder_layers=1, mdn_components=3,
        )
        vocab = build_vocab(["hello", "world", "the", "quick", "brown"])
        model = HandwritingModel(cfg, len(vocab))
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        return InferenceEngine(model, vocab, device)

    def test_generate_returns_list(self):
        engine = self._make_mock_engine()
        result = engine.generate("hello")
        self.assertIsInstance(result, list)

    def test_generate_returns_stroke_points(self):
        engine = self._make_mock_engine()
        result = engine.generate("hello")
        self.assertGreater(len(result), 0)
        for pt in result:
            self.assertIsInstance(pt, StrokePoint)

    def test_pen_up_values_valid(self):
        engine = self._make_mock_engine()
        result = engine.generate("hello")
        for pt in result:
            self.assertIn(pt.pen_up, [0, 1])

    def test_x_starts_at_zero(self):
        engine = self._make_mock_engine()
        result = engine.generate("hello")
        if result:
            min_x = min(s.x for s in result)
            self.assertAlmostEqual(min_x, 0.0, places=3)

    def test_generate_produces_variation(self):
        engine = self._make_mock_engine()
        r1 = engine.generate("hello")
        r2 = engine.generate("hello")
        # Different results each call (natural variation)
        if r1 and r2:
            coords1 = [(s.x, s.y) for s in r1[:5]]
            coords2 = [(s.x, s.y) for s in r2[:5]]
            self.assertNotEqual(coords1, coords2)

    def test_get_bbox_returns_triple(self):
        engine = self._make_mock_engine()
        w, a, d = engine.get_bbox("hello")
        self.assertIsInstance(w, float)
        self.assertIsInstance(a, float)
        self.assertIsInstance(d, float)

    def test_get_bbox_non_negative(self):
        engine = self._make_mock_engine()
        w, a, d = engine.get_bbox("hello")
        self.assertGreaterEqual(w, 0)
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(d, 0)

    def test_get_bbox_cached(self):
        engine = self._make_mock_engine()
        b1 = engine.get_bbox("hello")
        b2 = engine.get_bbox("hello")
        # Same object from cache
        self.assertEqual(b1, b2)
        self.assertIn("hello", engine._bbox_cache)

    def test_clear_bbox_cache(self):
        engine = self._make_mock_engine()
        engine.get_bbox("hello")
        self.assertIn("hello", engine._bbox_cache)
        engine.clear_bbox_cache()
        self.assertEqual(len(engine._bbox_cache), 0)

    def test_unknown_char_doesnt_crash(self):
        engine = self._make_mock_engine()
        # 'ñ' almost certainly not in tiny vocab
        try:
            result = engine.generate("cañon")
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"Unknown char caused crash: {e}")

    def test_generate_batch_length_matches(self):
        engine = self._make_mock_engine()
        words  = ["hello", "world", "the"]
        result = engine.generate_batch(words)
        self.assertEqual(len(result), len(words))

    def test_generate_batch_all_stroke_lists(self):
        engine = self._make_mock_engine()
        result = engine.generate_batch(["hello", "world"])
        for strokes in result:
            self.assertIsInstance(strokes, list)

    def test_temperature_low_less_variance(self):
        """Lower temperature should produce less spread in coordinates."""
        import numpy as np
        engine = self._make_mock_engine()

        def variance(word, temp, n=5):
            all_lens = [len(engine.generate(word, temperature=temp)) for _ in range(n)]
            return np.std(all_lens)

        var_low  = variance("hello", 0.5)
        var_high = variance("hello", 1.5)
        # High temp should have more variance in sequence length
        # This is a soft test — direction should be right
        self.assertGreaterEqual(var_high + 0.01, var_low)

    def test_vocab_size_property(self):
        engine = self._make_mock_engine()
        self.assertGreater(engine.vocab_size, 0)

    def test_known_chars_property(self):
        engine = self._make_mock_engine()
        chars = engine.known_chars
        self.assertIsInstance(chars, set)
        self.assertGreater(len(chars), 0)

    def test_repr(self):
        engine = self._make_mock_engine()
        r = repr(engine)
        self.assertIn("InferenceEngine", r)
        self.assertIn("vocab_size", r)


import os  # needed for preview file existence check above

if __name__ == "__main__":
    unittest.main(verbosity=2)
