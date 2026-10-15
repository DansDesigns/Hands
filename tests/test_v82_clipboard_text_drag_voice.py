"""Regression checks for non-blocking clipboard text drops and voice visibility commands."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")
VOICE = (ROOT / "voice_pc_commands.py").read_text(encoding="utf-8")


class ClipboardTextDragAndVoiceTests(unittest.TestCase):
    def test_plain_text_uses_non_native_drag_mode(self):
        self.assertIn('entry_obj.get("kind") == "text"', MAIN)
        self.assertIn('"text_paste" if entry_obj.get("kind") == "text" else "image_paste"', MAIN)
        self.assertIn('clipboard_state.get("mode") in ("text_paste", "image_paste")', MAIN)
        self.assertIn('self._paste_floating_clipboard_entry_at(item, pos)', MAIN)

    def test_text_and_image_use_non_native_drag_modes(self):
        start = MAIN.index('if entry_obj is not None and entry_obj.get("kind") in ("text", "image"):')
        end = MAIN.index('drag_id = clipboard_state["id"]', start)
        branch = MAIN[start:end]
        self.assertIn('"text_paste" if entry_obj.get("kind") == "text" else "image_paste"', branch)
        self.assertNotIn("_run_floating_clipboard_native_drag", branch)

    def test_image_drag_uses_same_cursor_preview_and_nonblocking_paste_release(self):
        self.assertIn('clipboard_state.get("mode") in ("text_paste", "image_paste")', MAIN)
        start = MAIN.index("    def _paste_floating_clipboard_entry_at(self, entry, global_pos):")
        end = MAIN.index("    def _finish_floating_clipboard_paste(self, keys):", start)
        method = MAIN[start:end]
        self.assertIn("self._set_system_clipboard_for_drop(entry)", method)
        self.assertIn("QCursor.setPos(global_pos)", method)
        self.assertIn("QTimer.singleShot(140", method)

    def test_tracking_loss_completes_nonmodal_image_drag(self):
        tracking_loss_start = MAIN.index('if (clipboard_state.get("running") and')
        tracking_loss_end = MAIN.index('self.floating_clipboard_drag_state[side] = None', tracking_loss_start)
        tracking_loss_branch = MAIN[tracking_loss_start:tracking_loss_end]
        self.assertIn('"image_paste")', tracking_loss_branch)
        self.assertIn('clipboard_state.get("moved")', tracking_loss_branch)

    def test_drop_paste_wait_is_nonblocking_and_clicks_current_qt_cursor_position(self):
        start = MAIN.index("    def _paste_floating_clipboard_entry_at(self, entry, global_pos):")
        end = MAIN.index("    def _finish_floating_clipboard_paste(self, keys):", start)
        method = MAIN[start:end]
        self.assertIn("QTimer.singleShot(140", method)
        self.assertIn("QCursor.setPos(global_pos)", method)
        self.assertIn("pyautogui.click(_pause=False)", method)
        self.assertNotIn("pyautogui.click(int(global_pos.x())", method)
        self.assertNotIn("time.sleep", method)

    def test_text_drag_draws_a_moving_clipboard_preview(self):
        start = MAIN.index("    def _paint_floating_clipboard(self, painter, now=None):")
        end = MAIN.index("    def _clipboard_full_image(self, entry):", start)
        method = MAIN[start:end]
        self.assertIn('state.get("preview_pixmap")', method)
        self.assertIn('state.get("last_cursor")', method)
        self.assertIn("painter.drawPixmap(QRectF(px, py, preview_w, preview_h)", method)

    def test_text_drag_finishes_after_a_stable_release_and_uses_current_global_cursor(self):
        self.assertIn('clipboard_state["release_candidate_since"] = now', MAIN)
        self.assertIn("now-release_since >= 0.14", MAIN)
        self.assertIn("drop_global = QCursor.pos()", MAIN)
        self.assertIn('clipboard_state["mode"] = (', MAIN)
        self.assertIn('else "image_paste")', MAIN)

    def test_voice_aliases_open_and_close_clipboard(self):
        from voice_pc_commands import match_pc_command
        for phrase in ("open clipboard", "show clipboard", "open floating clipboard", "show floating clipboard"):
            self.assertEqual(match_pc_command(phrase), "show_clipboard")
        for phrase in ("close clipboard", "hide clipboard", "close floating clipboard", "hide floating clipboard"):
            self.assertEqual(match_pc_command(phrase), "hide_clipboard")

    def test_voice_actions_set_visibility_explicitly(self):
        start = MAIN.index("    def execute_pc_voice_command(self, action):")
        end = MAIN.index("    def _clear_heard_popup(self):", start)
        method = MAIN[start:end]
        self.assertIn('if action == "show_clipboard":', method)
        self.assertIn("self._set_floating_clipboard_visible(True)", method)
        self.assertIn('if action == "hide_clipboard":', method)
        self.assertIn("self._set_floating_clipboard_visible(False)", method)

    def test_main_and_voice_source_parse(self):
        ast.parse(MAIN)
        ast.parse(VOICE)


if __name__ == "__main__":
    unittest.main()
