"""Regression tests for grouped Explorer file copies in the floating clipboard."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


class GroupedFileClipboardDragTests(unittest.TestCase):
    def test_multi_file_copy_is_kept_as_one_entry_with_group_count(self):
        self.assertIn('"group_count": len(payload["urls"]) if payload["kind"] == "files" else 1', MAIN)
        self.assertIn('self.floating_clipboard_entries.append(entry)', MAIN)
        self.assertIn('"urls": payload["urls"]', MAIN)

    def test_grouped_files_keep_every_url_in_clipboard_and_drop_payload(self):
        start = MAIN.index("    def _clipboard_mime_for_entry(self, entry, for_drag=False):")
        end = MAIN.index("    def _set_system_clipboard_from_entry(self, entry):", start)
        mime_builder = MAIN[start:end]
        self.assertIn('urls = list(entry.get("urls") or [])', mime_builder)
        self.assertIn('if urls:\n            mime.setUrls(urls)', mime_builder)

    def test_multi_file_cards_show_group_count_and_names(self):
        self.assertIn('kind_label = f"FILES · {file_count}" if file_count > 1 else "FILES"', MAIN)
        self.assertIn('detail = " · ".join(names)', MAIN)
        self.assertIn('label = (f"{count} files · " + ", ".join(names[:3]))', MAIN)

    def test_files_use_non_modal_drag_path_not_native_qdrag(self):
        self.assertIn('entry_obj.get("kind") in ("text", "image", "files")', MAIN)
        self.assertIn('"files": "files_paste"', MAIN)
        self.assertIn('clipboard_state.get("mode") in ("text_paste", "image_paste", "files_paste")', MAIN)
        self.assertIn('self._paste_floating_clipboard_entry_at(item, pos)', MAIN)

    def test_file_group_is_pasted_with_urls_unchanged(self):
        start = MAIN.index("    def _set_system_clipboard_for_drop(self, entry):")
        end = MAIN.index("    def _clipboard_drag_pixmap(self, entry):", start)
        setup = MAIN[start:end]
        self.assertIn('mime = self._clipboard_mime_for_entry(entry, for_drag=True)', setup)
        self.assertIn('QApplication.clipboard().setMimeData(mime)', setup)
        self.assertNotIn('for url in entry.get("urls"', setup)

    def test_main_source_parses(self):
        ast.parse(MAIN)


if __name__ == "__main__":
    unittest.main()
