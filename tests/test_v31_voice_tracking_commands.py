"""Regression checks for voice commands that pause and resume hand tracking."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class VoiceTrackingCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.voice_setup = (ROOT / "VOICE_SETUP.md").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_commands_are_global_before_search_dictation(self):
        method_start = self.main_source.index("    def on_speech(self, text):")
        method_end = self.main_source.index("    def apply_desktop_mode", method_start)
        method = self.main_source[method_start:method_end]
        self.assertLess(method.index('"pause tracking"'), method.index("dialog.set_dictation_text(text, final=True)"))
        self.assertLess(method.index('"resume tracking"'), method.index("dialog.set_dictation_text(text, final=True)"))
        self.assertIn('"pause hand tracking"', method)
        self.assertIn('"resume hand tracking"', method)

    def test_pause_keeps_vosk_running_but_turns_off_hand_inference(self):
        self.assertIn("self.tracker.set_hand_tracking_enabled(not paused)", self.main_source)
        self.assertIn("def set_hand_tracking_enabled(self, enabled: bool)", self.main_source)
        self.assertIn("if self.hand_tracking_enabled:", self.main_source)
        self.assertNotIn("self.speech.stop()", self.main_source[
            self.main_source.index("    def set_hand_tracking_paused"):self.main_source.index("    def on_speech(self, text):")
        ])

    def test_pausing_cancels_actions_and_clears_visuals(self):
        method_start = self.main_source.index("    def set_hand_tracking_paused")
        method_end = self.main_source.index("    def on_speech(self, text):", method_start)
        method = self.main_source[method_start:method_end]
        self.assertIn("self._release_fist_mouse(side)", method)
        self.assertIn("self.active_action[side] = None", method)
        self.assertIn('self.hand_positions[side] = []', method)
        self.assertIn("self.gesture_owner = None", method)

    def test_tracking_pause_banner_and_docs_exist(self):
        self.assertIn("HAND TRACKING PAUSED", self.main_source)
        self.assertIn("resume tracking", self.voice_setup.lower())
        self.assertIn("pause tracking", self.readme.lower())


if __name__ == "__main__":
    unittest.main()
