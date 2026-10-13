"""Fast smoke tests for the no-model local search index."""
import tempfile
import unittest
from pathlib import Path

from local_file_search import LocalFileIndex


class LocalFileSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "Documents"
        self.root.mkdir()
        (self.root / "floating-canvas-calibration.txt").write_text(
            "Notes about hand tracking and calibration for the floating canvas.", encoding="utf-8")
        (self.root / "holiday-photo.jpg").write_bytes(b"image placeholder")
        (self.root / "meeting-minutes.md").write_text(
            "Project discussion, decisions and meeting minutes.", encoding="utf-8")
        self.index = LocalFileIndex(base / "index.sqlite3", base / "roots.json")
        self.index._save_roots([str(self.root)])
        self.index.index_roots([str(self.root)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_content_query_finds_relevant_document(self):
        results = self.index.search("floating canvas calibration")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "floating-canvas-calibration.txt")

    def test_simple_synonym_query_finds_filename(self):
        results = self.index.search("holiday photo")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "holiday-photo.jpg")

    def test_meeting_notes_query_finds_minutes(self):
        results = self.index.search("meeting notes")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "meeting-minutes.md")

    def test_reindex_skips_unchanged_files(self):
        seen, changed = self.index.index_roots([str(self.root)])
        self.assertEqual(seen, 3)
        self.assertEqual(changed, 0)


if __name__ == "__main__":
    unittest.main()
