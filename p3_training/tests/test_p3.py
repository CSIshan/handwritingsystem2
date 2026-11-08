"""
P3 Test Suite
=============
Tests model architecture, MDN, dataset, and prep
without requiring a real dataset or GPU.
Uses torch if available, otherwise skips torch-dependent tests.
"""
import sys, os, json, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from handwriting_system.p3_training.config import TrainConfig
from handwriting_system.p3_training.prep   import (
    build_vocab, deduplicate, compute_corpus_stats,
    SPECIAL_TOKENS, PAD_TOKEN, UNK_TOKEN,
)
# ── Prep tests (no torch needed) ──────────────────────────────────────────────

class TestVocab(unittest.TestCase):

    def test_special_tokens_first(self):
        vocab = build_vocab(["hello"])
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.assertEqual(vocab[tok], i)

    def test_all_chars_present(self):
        vocab = build_vocab(["hello", "world", "123"])
        for ch in "helo wrd123":
            self.assertIn(ch, vocab)

    def test_unknown_chars_handled(self):
        vocab = build_vocab(["hello"])
        self.assertIn(UNK_TOKEN, vocab)

    def test_vocab_indices_unique(self):
        vocab = build_vocab(["abcdefg"])
        indices = list(vocab.values())
        self.assertEqual(len(indices), len(set(indices)))

    def test_pad_is_zero(self):
        vocab = build_vocab(["hello"])
        self.assertEqual(vocab[PAD_TOKEN], 0)


class TestDeduplicate(unittest.TestCase):

    def _make_record(self, word, strokes=None):
        strokes = strokes or [[0.1, 0.0, 0]] * 5 + [[0, 0, 1]]
        return {"word": word, "strokes": strokes,
                "bbox_mm": {"width": 10.0, "ascent": 5.0, "descent": 1.0}}

    def test_exact_duplicates_removed(self):
        r = self._make_record("hello")
        records = [r, r.copy(), r.copy()]
        result = deduplicate(records)
        self.assertEqual(len(result), 1)

    def test_same_word_different_strokes_kept(self):
        r1 = self._make_record("hello", [[0.1, 0.0, 0]] * 5 + [[0,0,1]])
        r2 = self._make_record("hello", [[0.2, 0.1, 0]] * 5 + [[0,0,1]])
        result = deduplicate([r1, r2])
        self.assertEqual(len(result), 2)

    def test_different_words_kept(self):
        r1 = self._make_record("hello")
        r2 = self._make_record("world")
        result = deduplicate([r1, r2])
        self.assertEqual(len(result), 2)


class TestCorpusStats(unittest.TestCase):

    def _make_records(self, n=10):
        import random
        rng = random.Random(42)
        return [
            {
                "word":    f"word{i}",
                "strokes": [[rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.5), 0]
                             for _ in range(rng.randint(20, 80))] + [[0,0,1]],
                "bbox_mm": {
                    "width":   rng.uniform(8, 25),
                    "ascent":  rng.uniform(4, 7),
                    "descent": rng.uniform(0, 2),
                },
            }
            for i in range(n)
        ]

    def test_stats_have_required_keys(self):
        records = self._make_records(10)
        stats   = compute_corpus_stats(records)
        self.assertIn("word_width_mm",     stats)
        self.assertIn("word_ascent_mm",    stats)
        self.assertIn("stroke_seq_length", stats)
        self.assertIn("total_words",       stats)

    def test_percentile_keys_present(self):
        records = self._make_records(10)
        stats   = compute_corpus_stats(records)
        dist    = stats["word_width_mm"]
        for key in ["p10", "p25", "p50", "p75", "p90", "iqr"]:
            self.assertIn(key, dist)

    def test_median_not_mean(self):
        # With an outlier, median should be robust
        import numpy as np
        records = self._make_records(9)
        # Add outlier with huge width
        records.append({
            "word": "outlier", "strokes": [[0,0,1]],
            "bbox_mm": {"width": 500.0, "ascent": 5.0, "descent": 1.0}
        })
        stats  = compute_corpus_stats(records)
        median = stats["word_width_mm"]["p50"]
        # Median should be close to normal range, not pulled to 500
        self.assertLess(median, 100.0)

    def test_iqr_used_not_stddev(self):
        records = self._make_records(20)
        stats   = compute_corpus_stats(records)
        self.assertIn("iqr", stats["word_width_mm"])
        # IQR should be positive
        self.assertGreater(stats["word_width_mm"]["iqr"], 0)


class TestTrainConfig(unittest.TestCase):

    def test_encoder_output_dim(self):
        cfg = TrainConfig(encoder_hidden=256)
        self.assertEqual(cfg.encoder_output_dim, 512)

    def test_default_values_sane(self):
        cfg = TrainConfig()
        self.assertGreater(cfg.batch_size, 0)
        self.assertGreater(cfg.epochs,     0)
        self.assertGreater(cfg.grad_clip,  0)
        self.assertLessEqual(cfg.train_split, 1.0)
        self.assertGreaterEqual(cfg.train_split, 0.5)


# ── Torch-dependent tests ─────────────────────────────────────────────────────
class TestMDN(unittest.TestCase):

    def setUp(self):
        from handwriting_system.p3_training.mdn import mdn_loss, mdn_sample
        self.mdn_loss   = mdn_loss
        self.mdn_sample = mdn_sample

    def test_mdn_loss_returns_scalar(self):
        B, T, K = 2, 10, 5
        pi      = torch.softmax(torch.randn(B, T, K), dim=-1)
        mu_x    = torch.randn(B, T, K)
        mu_y    = torch.randn(B, T, K)
        sigma_x = torch.exp(torch.randn(B, T, K)).clamp(min=1e-4)
        sigma_y = torch.exp(torch.randn(B, T, K)).clamp(min=1e-4)
        rho     = torch.tanh(torch.randn(B, T, K))
        tx      = torch.randn(B, T)
        ty      = torch.randn(B, T)
        mask    = torch.ones(B, T)

        loss = self.mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho, tx, ty, mask)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_mdn_loss_positive(self):
        B, T, K = 2, 5, 3
        pi      = torch.softmax(torch.randn(B, T, K), dim=-1)
        mu_x    = torch.zeros(B, T, K)
        mu_y    = torch.zeros(B, T, K)
        sigma_x = torch.ones(B, T, K)
        sigma_y = torch.ones(B, T, K)
        rho     = torch.zeros(B, T, K)
        tx      = torch.randn(B, T)
        ty      = torch.randn(B, T)
        mask    = torch.ones(B, T)
        loss = self.mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho, tx, ty, mask)
        self.assertGreater(loss.item(), 0)

    def test_mdn_sample_returns_two_floats(self):
        K  = 5
        pi = torch.softmax(torch.randn(K), dim=0)
        dx, dy = self.mdn_sample(
            pi,
            torch.randn(K), torch.randn(K),
            torch.exp(torch.randn(K)), torch.exp(torch.randn(K)),
            torch.tanh(torch.randn(K)),
        )
        self.assertIsInstance(dx, float)
        self.assertIsInstance(dy, float)

    def test_mdn_mask_zeros_out_padding(self):
        B, T, K = 2, 5, 3
        pi      = torch.softmax(torch.randn(B, T, K), dim=-1)
        mu_x    = torch.zeros(B, T, K)
        mu_y    = torch.zeros(B, T, K)
        sigma_x = torch.ones(B, T, K)
        sigma_y = torch.ones(B, T, K)
        rho     = torch.zeros(B, T, K)
        tx      = torch.randn(B, T)
        ty      = torch.randn(B, T)

        mask_full = torch.ones(B, T)
        mask_half = torch.zeros(B, T)
        mask_half[:, :3] = 1.0

        loss_full = self.mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho,
                                  tx, ty, mask_full)
        loss_half = self.mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho,
                                  tx, ty, mask_half)
        # Different masks → different losses
        self.assertNotAlmostEqual(loss_full.item(), loss_half.item(), places=3)

class TestModel(unittest.TestCase):

    def setUp(self):
        from handwriting_system.p3_training.model import HandwritingModel
        from handwriting_system.p3_training.prep  import build_vocab
        self.cfg   = TrainConfig(
            embed_dim=16, encoder_hidden=32, decoder_hidden=64,
            encoder_layers=1, decoder_layers=1, mdn_components=3,
            batch_size=2, epochs=1,
        )
        self.vocab = build_vocab(["hello", "world"])
        self.model = HandwritingModel(self.cfg, len(self.vocab))

    def test_forward_output_keys(self):
        B, T, L = 2, 20, 5
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L-1])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T-2])

        out = self.model(char_ids, char_lens, strokes, stroke_lens)
        for key in ["pi", "mu_x", "mu_y", "sigma_x", "sigma_y", "rho",
                    "pen_up", "stop"]:
            self.assertIn(key, out)

    def test_forward_output_shapes(self):
        B, T, L, K = 2, 10, 4, self.cfg.mdn_components
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T])

        out = self.model(char_ids, char_lens, strokes, stroke_lens)
        self.assertEqual(out["pi"].shape,     (B, T, K))
        self.assertEqual(out["pen_up"].shape, (B, T))
        self.assertEqual(out["stop"].shape,   (B, T))

    def test_loss_is_scalar(self):
        B, T, L = 2, 10, 4
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T])

        out    = self.model(char_ids, char_lens, strokes, stroke_lens)
        losses = self.model.compute_loss(out, strokes, stroke_lens)
        self.assertEqual(losses["total"].shape, torch.Size([]))
        self.assertFalse(torch.isnan(losses["total"]))

    def test_loss_backward(self):
        B, T, L = 2, 8, 4
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T])

        out    = self.model(char_ids, char_lens, strokes, stroke_lens)
        losses = self.model.compute_loss(out, strokes, stroke_lens)
        # Should not crash
        losses["total"].backward()

    def test_generate_returns_list(self):
        result = self.model.generate("hello", self.vocab, max_steps=20)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for point in result:
            self.assertEqual(len(point), 3)

    def test_get_bbox_returns_triple(self):
        w, a, d = self.model.get_bbox("hello", self.vocab)
        self.assertIsInstance(w, float)
        self.assertIsInstance(a, float)
        self.assertIsInstance(d, float)
        self.assertGreaterEqual(w, 0)
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(d, 0)

    def test_sigma_positive(self):
        B, T, L = 2, 5, 3
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T])

        out = self.model(char_ids, char_lens, strokes, stroke_lens)
        self.assertTrue((out["sigma_x"] > 0).all())
        self.assertTrue((out["sigma_y"] > 0).all())

    def test_pi_sums_to_one(self):
        B, T, L = 2, 5, 3
        char_ids    = torch.randint(0, len(self.vocab), (B, L))
        char_lens   = torch.tensor([L, L])
        strokes     = torch.randn(B, T, 3)
        stroke_lens = torch.tensor([T, T])

        out = self.model(char_ids, char_lens, strokes, stroke_lens)
        pi_sum = out["pi"].sum(dim=-1)
        self.assertTrue(torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
