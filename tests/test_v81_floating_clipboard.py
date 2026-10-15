"""Static regression coverage for the system-backed floating clipboard feature."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")


class FloatingClipboardTests(unittest.TestCase):
    def test_clipboard_listener_is_connected_on_startup(self):
        self.assertIn("self._system_clipboard = QApplication.clipboard()", SOURCE)
        self.assertIn("self._system_clipboard.dataChanged.connect(self._capture_system_clipboard)", SOURCE)
        self.assertIn("QTimer.singleShot(350, self._capture_system_clipboard)", SOURCE)

    def test_capture_supports_text_images_and_urls(self):
        for token in ("mime.hasImage()", "mime.hasUrls()", "mime.hasText()", "mime.hasHtml()",
                      'kind = "image"', 'kind = "files"', 'else "links"', 'kind = "text"'):
            self.assertIn(token, SOURCE)

    def test_image_file_copy_gets_preview_while_preserving_file_url(self):
        self.assertIn("image_suffixes = {\".png\", \".jpg\", \".jpeg\", \".bmp\", \".gif\", \".webp\", \".tif\", \".tiff\"}", SOURCE)
        self.assertIn("candidate = QImage(local_path)", SOURCE)
        self.assertIn('QUrl.fromLocalFile(entry["image_path"])', SOURCE)

    def test_full_resolution_image_is_kept_on_disk_and_loaded_on_demand(self):
        self.assertIn('entry["image_width"] = int(original_image.width())', SOURCE)
        self.assertIn('original_image.save(str(path), "PNG")', SOURCE)
        self.assertIn('def _clipboard_full_image(self, entry):', SOURCE)
        self.assertIn('full = QImage(image_path)', SOURCE)
        self.assertIn('clipboard.setImage(self._clipboard_full_image(entry))', SOURCE)
        self.assertIn('self._clipboard_capture_suppression_until = time.monotonic()+0.8', SOURCE)

    def test_clipboard_history_is_bounded_and_cleans_evicted_previews(self):
        self.assertIn("self.floating_clipboard_max_entries = 36", SOURCE)
        self.assertIn("evicted = self.floating_clipboard_entries[:-self.floating_clipboard_max_entries]", SOURCE)
        self.assertIn("Path(stale_path).unlink(missing_ok=True)", SOURCE)

    def test_pinch_menu_has_icon_only_right_click_and_clipboard_toggle(self):
        self.assertIn('return ("right_click", "clipboard_toggle", "glass_menu", "dictation_toggle", "stt_toggle")', SOURCE)
        self.assertIn('"right_click": ""', SOURCE)
        self.assertIn('"clipboard_toggle": ""', SOURCE)
        self.assertIn('self._paint_right_click_icon(painter, rect, hovered)', SOURCE)
        self.assertIn('self._paint_clipboard_icon(painter, rect, hovered or self.floating_clipboard_visible)', SOURCE)

    def test_top_row_buttons_match_dictation_and_stt_width(self):
        self.assertIn('"right_click": QRectF(item_x, top, half_w, item_h)', SOURCE)
        self.assertIn('"clipboard_toggle": QRectF(item_x+half_w+gap, top, half_w, item_h)', SOURCE)
        self.assertIn('"dictation_toggle": QRectF(item_x, top+2*(item_h+gap), half_w, item_h)', SOURCE)
        self.assertIn('"stt_toggle": QRectF(item_x+half_w+gap, top+2*(item_h+gap), half_w, item_h)', SOURCE)

    def test_clipboard_toggle_action_shows_and_hides_the_panel(self):
        self.assertIn('elif action == "clipboard_toggle":', SOURCE)
        self.assertIn('self._set_floating_clipboard_visible(not self.floating_clipboard_visible)', SOURCE)
        self.assertIn("self._paint_floating_clipboard(painter, now)", SOURCE)

    def test_native_drag_provides_text_image_file_url_mime_payloads(self):
        self.assertIn("mime = QMimeData()", SOURCE)
        self.assertIn("mime.setImageData(image)", SOURCE)
        self.assertIn("mime.setUrls(urls)", SOURCE)
        self.assertIn("mime.setText(text)", SOURCE)
        self.assertIn("mime.setHtml(html)", SOURCE)
        self.assertIn("drag.exec(Qt.CopyAction | Qt.LinkAction, Qt.CopyAction)", SOURCE)

    def test_drag_fallback_sets_clipboard_and_pastes_at_drop_location(self):
        self.assertIn("self._paste_floating_clipboard_entry_at(entry, drop_global)", SOURCE)
        self.assertIn('pyautogui.hotkey(*keys, interval=0.015, _pause=False)', SOURCE)
        self.assertIn('self._clipboard_capture_suppression_until = time.monotonic()+0.8', SOURCE)

    def test_existing_copy_this_routes_to_clipboard_capture(self):
        self.assertIn('QTimer.singleShot(180, self._capture_system_clipboard)', SOURCE)
        self.assertIn('QTimer.singleShot(120, self._capture_system_clipboard)', SOURCE)
        self.assertIn('copy this/copy text, paste here/paste this', SOURCE)

    def test_clipboard_drag_survives_modal_glass_menu_hit_test(self):
        self.assertIn("clipboard_drag_in_progress = self.floating_clipboard_drag_state.get(side) is not None", SOURCE)
        self.assertIn("not clipboard_drag_in_progress):", SOURCE)

    def test_main_source_parses(self):
        ast.parse(SOURCE)


if __name__ == "__main__":
    unittest.main()
