from pathlib import Path
import unittest
from voice_pc_commands import match_pc_command, parse_navigation_command, is_navigation_intent, has_explicit_create_prefix

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
        self.assertIsNone(match_pc_command("new folder"))
        self.assertEqual(match_pc_command("create new folder"), "new_folder")
        self.assertEqual(match_pc_command("create new text file"), "create_new_text_file")
        self.assertEqual(match_pc_command("create new powerpoint"), "create_new_powerpoint")
        self.assertEqual(match_pc_command("move to recycle bin"), "delete_selected")


    def test_my_computer_aliases(self):
        self.assertEqual(match_pc_command("open my computer"), "open_my_computer")
        self.assertEqual(match_pc_command("show this PC"), "open_my_computer")

    def test_parameterised_file_navigation_and_selection(self):
        self.assertEqual(parse_navigation_command("go to C drive"),
                         {"action": "navigate_drive", "drive": "C"})
        self.assertEqual(parse_navigation_command("select the 'Agents' folder"),
                         {"action": "select_items", "names": ["Agents"], "kind": "folder"})
        self.assertEqual(parse_navigation_command("select a folder called llama.cpp"),
                         {"action": "select_items", "names": ["llama.cpp"], "kind": "folder"})
        self.assertEqual(parse_navigation_command("could you select the folder called llama.cpp"),
                         {"action": "select_items", "names": ["llama.cpp"], "kind": "folder"})
        self.assertEqual(parse_navigation_command("open the folder named llama.cpp"),
                         {"action": "open_item", "name": "llama.cpp", "kind": "folder"})
        self.assertTrue(is_navigation_intent("could you select a folder called llama.cpp"))
        self.assertFalse(is_navigation_intent("what is llama.cpp used for"))
        self.assertEqual(parse_navigation_command('open "Annual report.pdf"'),
                         {"action": "open_item", "name": "Annual report.pdf", "kind": None})
        self.assertEqual(parse_navigation_command('select "report one.docx" and "image two.png"'),
                         {"action": "select_items", "names": ["report one.docx", "image two.png"], "kind": None})
        self.assertEqual(parse_navigation_command("select report one.docx and image two.png"),
                         {"action": "select_items", "names": ["report one.docx", "image two.png"], "kind": None})
        self.assertEqual(parse_navigation_command("select all files"),
                         {"action": "select_all_kind", "kind": "file"})
        self.assertEqual(parse_navigation_command("select all folders"),
                         {"action": "select_all_kind", "kind": "folder"})
        self.assertEqual(parse_navigation_command("select new folder 2"),
                         {"action": "select_items", "names": ["new folder 2"], "kind": None})
        self.assertEqual(parse_navigation_command("select ponder.mp4"),
                         {"action": "select_items", "names": ["ponder.mp4"], "kind": None})
        self.assertEqual(parse_navigation_command("select my_project"),
                         {"action": "select_items", "names": ["my_project"], "kind": None})
        self.assertTrue(is_navigation_intent("select new folder 2"))
        self.assertTrue(is_navigation_intent("select ponder.mp4"))
        self.assertTrue(is_navigation_intent("select ponder"))
        self.assertTrue(is_navigation_intent("create new text file"))
        self.assertTrue(is_navigation_intent("create new powerpoint"))
        self.assertFalse(is_navigation_intent("new folder"))
        self.assertFalse(has_explicit_create_prefix("new folder 2"))
        self.assertFalse(has_explicit_create_prefix("please new folder"))
        self.assertTrue(has_explicit_create_prefix("please create new folder"))
        self.assertEqual(parse_navigation_command("could you create a new text file"),
                         {"action": "create_item", "kind": "text"})
        self.assertEqual(parse_navigation_command("please create a new powerpoint"),
                         {"action": "create_item", "kind": "powerpoint"})

    def test_dynamic_navigation_runs_before_dictation_fallback(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("    def on_speech(self, text):")
        end = source.index("    def apply_desktop_mode", start)
        method = source[start:end]
        self.assertIn("navigation_command = parse_navigation_command(text)", method)
        self.assertLess(method.index("self.execute_navigation_voice_command(navigation_command)"),
                        method.index('dialog.set_dictation_text(text, final=True)'))
        self.assertIn("def _run_windows_explorer_navigation", source)
        self.assertIn("$document.SelectItem($item, $flags)", source)
        self.assertIn("Normalize-SpokenItemName", source)
        self.assertIn("I couldn't complete that file command. Check the item name", source)
        self.assertNotIn('error[-1] if error else "File Explorer could not complete that command."', source)
        self.assertIn("'create_item'", source)
        self.assertIn("has_explicit_create_prefix(text)", source)
        self.assertIn("unprefixed_create = re.fullmatch", source)

    def test_navigation_does_not_reach_generative_fallback_and_does_not_unmaximize(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("    def _focus_external_target(self, close_search=True):")
        end = source.index("    def _voice_hotkey", start)
        focus_method = source[start:end]
        self.assertIn("if user32.IsIconic(wintypes.HWND(target)):", focus_method)
        self.assertIn("user32.ShowWindow(wintypes.HWND(target), 9)", focus_method)
        start = source.index("    def on_speech(self, text):")
        end = source.index("    def apply_desktop_mode", start)
        speech_method = source[start:end]
        self.assertIn("if is_navigation_intent(text):", speech_method)
        self.assertLess(speech_method.index("if is_navigation_intent(text):"),
                        speech_method.index("pc_action = classify_voice_command(text, aliases_for_help())", speech_method.index("navigation_command = parse_navigation_command(text)")))

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
