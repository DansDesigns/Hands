from pathlib import Path
import unittest
from voice_pc_commands import match_pc_command

ROOT = Path(__file__).resolve().parent.parent


class VoicePCCommandTests(unittest.TestCase):
    def test_clipboard_and_selection_aliases(self):
        self.assertEqual(match_pc_command("copy this"), "copy")
        self.assertEqual(match_pc_command("PASTE HERE!"), "paste_here")
        self.assertEqual(match_pc_command("start text selection"), "start_text_selection")
        self.assertEqual(match_pc_command("copy image"), "copy_image_lasso")
        self.assertEqual(match_pc_command("cancel lasso"), "cancel_selection")

    def test_fullscreen_toggle_aliases(self):
        self.assertEqual(match_pc_command("fullscreen"), "fullscreen_toggle")
        self.assertEqual(match_pc_command("toggle full screen"), "fullscreen_toggle")

    def test_media_aliases(self):
        self.assertEqual(match_pc_command("play or pause"), "media_play_pause")
        self.assertEqual(match_pc_command("next song"), "media_next")
        self.assertEqual(match_pc_command("volume down"), "volume_down")
        self.assertEqual(match_pc_command("unmute audio"), "volume_unmute")

    def test_window_and_file_manager_aliases(self):
        self.assertEqual(match_pc_command("minimise this window"), "minimize_window")
        self.assertEqual(match_pc_command("snap window right"), "snap_right")
        self.assertEqual(match_pc_command("open downloads folder"), "open_downloads")
        self.assertEqual(match_pc_command("new folder"), "new_folder")
        self.assertEqual(match_pc_command("move to recycle bin"), "delete_selected")

    def test_unknown_phrase_is_not_interpreted_as_command(self):
        self.assertIsNone(match_pc_command("please find my old floating canvas notes"))

    def test_commands_are_dispatched_before_search_dictation(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("    def on_speech(self, text):")
        end = source.index("    def apply_desktop_mode", start)
        method = source[start:end]
        self.assertIn("pc_action = match_pc_command(low)", method)
        self.assertLess(method.index("pc_action = match_pc_command(low)"), method.index("dialog.set_dictation_text(text, final=True)"))

    def test_lasso_suppresses_conflicting_gestures_and_releases_mouse_on_exit(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('if self.voice_selection_mode is not None:', source)
        self.assertIn('pyautogui.mouseUp(button="left", _pause=False)', source)
        self.assertIn('QApplication.clipboard().setImage(image)', source)

    def test_commands_have_lasso_and_clipboard_integration(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("def _begin_voice_text_selection", source)
        self.assertIn("def _begin_voice_image_lasso", source)
        self.assertIn("def _paste_at_tli", source)
        self.assertIn('pyautogui.hotkey("ctrl", "v"', source)
        self.assertIn("setWindowOpacity(0.0)", source)
        self.assertIn("return False", source[source.index("# Never send keyboard shortcuts to our own overlay"):source.index("def _voice_hotkey")])

    def test_extended_command_set_is_documented(self):
        guide = (ROOT / "VOICE_SETUP.md").read_text(encoding="utf-8").lower()
        for phrase in ("play pause", "volume up", "open downloads", "snap window left",
                       "start text selection", "copy image", "paste here"):
            self.assertIn(phrase, guide)


if __name__ == "__main__":
    unittest.main()
