"""
P2 Test Suite
=============
Tests loader, progress, and saver without needing real scans or curses.
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import unittest
from pathlib import Path
from handwriting_system.p2_labelling_tool.loader   import load_raw_dataset
from handwriting_system.p2_labelling_tool.progress import Progress
from handwriting_system.p2_labelling_tool.saver    import save_verified
from handwriting_system.p2_labelling_tool.renderer import (
    render_strokes_frame, render_image_ascii
)


def _make_raw_json(tmp_dir, name, words, flagged_indices=None):
    """Create a synthetic extracted JSON file."""
    flagged_indices = flagged_indices or []
    records = []
    for i, word in enumerate(words):
        records.append({
            "word":        word,
            "strokes":     [[float(j)*0.1, 0.0, 0] for j in range(10)] + [[0,0,1]],
            "bbox_mm":     {"width": 15.0, "ascent": 5.0, "descent": 1.0},
            "flagged":     i in flagged_indices,
            "flag_reason": "too wide" if i in flagged_indices else None,
            "source":      {"pair_index": i // 5, "word_index": i % 5},
            "stroke_count": 1,
            "point_count":  11,
        })
    data = {"source_file": name, "dpi": 600, "page_stats": {}, "words": records}
    path = Path(tmp_dir) / f"{name}_extracted.json"
    path.write_text(json.dumps(data, indent=2))
    return path


class TestLoader(unittest.TestCase):

    def test_loads_records(self):
        with tempfile.TemporaryDirectory() as d:
            _make_raw_json(d, "session1_page1", ["hello", "world", "the"])
            records = load_raw_dataset(d, spot_check_ratio=1.0)
            self.assertGreater(len(records), 0)

    def test_flagged_come_first(self):
        with tempfile.TemporaryDirectory() as d:
            _make_raw_json(d, "session1_page1",
                           ["hello", "world", "the", "quick"],
                           flagged_indices=[2])
            records = load_raw_dataset(d, spot_check_ratio=1.0)
            # First record should be flagged
            self.assertTrue(records[0]["flagged"])

    def test_record_ids_unique(self):
        with tempfile.TemporaryDirectory() as d:
            _make_raw_json(d, "session1_page1", ["hello", "world", "the"])
            records = load_raw_dataset(d, spot_check_ratio=1.0)
            ids = [r["_record_id"] for r in records]
            self.assertEqual(len(ids), len(set(ids)))

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_raw_dataset("/nonexistent/path")

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_raw_dataset(d)

    def test_spot_check_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            words = [f"word{i}" for i in range(100)]
            _make_raw_json(d, "session1_page1", words)
            records = load_raw_dataset(d, spot_check_ratio=0.1, seed=42)
            spot = [r for r in records if r.get("_spot_check")]
            self.assertAlmostEqual(len(spot), 10, delta=2)

    def test_multiple_files_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            _make_raw_json(d, "session1_page1", ["hello", "world"])
            _make_raw_json(d, "session1_page2", ["foo", "bar"])
            records = load_raw_dataset(d, spot_check_ratio=1.0)
            sources = {r["_source_file"] for r in records}
            self.assertIn("session1_page1_extracted", sources)
            self.assertIn("session1_page2_extracted", sources)


class TestProgress(unittest.TestCase):

    def test_accept(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(f"{d}/progress.json")
            p.accept("word_001")
            self.assertTrue(p.is_decided("word_001"))

    def test_discard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(f"{d}/progress.json")
            p.discard("word_002")
            self.assertEqual(p.decided["word_002"], "discard")

    def test_edit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(f"{d}/progress.json")
            p.edit("word_003", "world", "hello")
            d_val = p.decided["word_003"]
            self.assertEqual(d_val["new_label"],      "world")
            self.assertEqual(d_val["original_label"], "hello")

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/progress.json"
            p1 = Progress(path)
            p1.accept("word_001")
            p1.discard("word_002")
            # Reload
            p2 = Progress(path)
            self.assertTrue(p2.is_decided("word_001"))
            self.assertTrue(p2.is_decided("word_002"))

    def test_skip_not_decided(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(f"{d}/progress.json")
            p.skip("word_004")
            self.assertFalse(p.is_decided("word_004"))

    def test_summary_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(f"{d}/progress.json")
            p.accept("w1")
            p.accept("w2")
            p.discard("w3")
            p.edit("w4", "world", "hello")
            s = p.summary()
            self.assertEqual(s["accepted"],  2)
            self.assertEqual(s["discarded"], 1)
            self.assertEqual(s["edited"],    1)
            self.assertEqual(s["total"],     4)

    def test_corrupt_progress_file_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "progress.json"
            path.write_text("not valid json")
            p = Progress(str(path))   # should not crash
            self.assertEqual(p.decided, {})


class TestSaver(unittest.TestCase):

    def _make_records(self, words, flagged_idx=None):
        flagged_idx = flagged_idx or []
        records = []
        for i, w in enumerate(words):
            records.append({
                "word":          w,
                "strokes":       [[0.1*j, 0.0, 0] for j in range(5)] + [[0,0,1]],
                "bbox_mm":       {"width": 10.0, "ascent": 5.0, "descent": 1.0},
                "flagged":       i in flagged_idx,
                "flag_reason":   None,
                "source":        {"pair_index": 0, "word_index": i},
                "_source_file":  "session1_page1_extracted",
                "_record_id":    f"session1_page1_extracted_p0_w{i}",
                "_spot_check":   False,
            })
        return records

    def test_accepted_words_saved(self):
        with tempfile.TemporaryDirectory() as d:
            records = self._make_records(["hello", "world"])
            p = Progress(f"{d}/progress.json")
            p.accept(records[0]["_record_id"])
            p.accept(records[1]["_record_id"])
            result = save_verified(records, p, verified_dir=d)
            self.assertEqual(result["saved"], 2)

    def test_discarded_words_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            records = self._make_records(["hello", "world"])
            p = Progress(f"{d}/progress.json")
            p.accept(records[0]["_record_id"])
            p.discard(records[1]["_record_id"])
            result = save_verified(records, p, verified_dir=d)
            self.assertEqual(result["saved"],    1)
            self.assertEqual(result["discarded"],1)

    def test_edited_label_saved_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            verified = Path(d) / "verified"
            records = self._make_records(["hello"])
            p = Progress(f"{d}/progress.json")
            p.edit(records[0]["_record_id"], "world", "hello")
            save_verified(records, p, verified_dir=str(verified))
            # Read saved file
            saved = list(verified.glob("*.json"))
            self.assertEqual(len(saved), 1)
            data = json.loads(saved[0].read_text())
            word = data["words"][0]
            self.assertEqual(word["word"],           "world")
            self.assertEqual(word["original_label"], "hello")
            self.assertTrue(word["edited"])

    def test_verified_flag_set(self):
        with tempfile.TemporaryDirectory() as d:
            verified = Path(d) / "verified"
            records = self._make_records(["hello"])
            p = Progress(f"{d}/progress.json")
            p.accept(records[0]["_record_id"])
            save_verified(records, p, verified_dir=str(verified))
            saved = list(verified.glob("*.json"))
            self.assertEqual(len(saved), 1)
            data  = json.loads(saved[0].read_text())
            self.assertTrue(data["words"][0]["verified"])

    def test_unreviewed_words_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            records = self._make_records(["hello", "world"])
            p = Progress(f"{d}/progress.json")
            # Only accept first, leave second unreviewed
            p.accept(records[0]["_record_id"])
            result = save_verified(records, p, verified_dir=d)
            self.assertEqual(result["saved"], 1)


class TestRenderer(unittest.TestCase):

    def test_stroke_render_returns_lines(self):
        strokes = [[0.0, 0.0, 0], [1.0, 0.5, 0], [2.0, 0.0, 1]]
        lines = render_strokes_frame(strokes, width=20, height=5)
        self.assertEqual(len(lines), 5)
        self.assertTrue(all(isinstance(l, str) for l in lines))

    def test_empty_strokes_handled(self):
        lines = render_strokes_frame([], width=20, height=5)
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)

    def test_render_width_respected(self):
        strokes = [[float(i)*0.1, 0.0, 0] for i in range(20)] + [[0,0,1]]
        lines = render_strokes_frame(strokes, width=30, height=5)
        for line in lines:
            self.assertLessEqual(len(line), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
