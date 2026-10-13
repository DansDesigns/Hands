"""v34 regression tests for overlay stacking, pinch detection, scrolling and dictation."""
from pathlib import Path
import unittest

from gesture_rules import progressive_scroll_gain, should_recognize_menu_pinch

ROOT = Path(__file__).resolve().parent.parent


class V34PinchAndScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")

    def test_menu_pinch_default_is_point_six_seconds(self):
        self.assertIn('"menu_pinch_hold_seconds": 0.6', self.main)
        self.assertIn("menu_pinch_hold_seconds=0.60", self.gestures)
        self.assertIn('("pinch_timer", -124, self.menu_pinch_hold_seconds, 0.5, 5.0', self.main)

    def test_natural_bent_index_pinch_is_recognised(self):
        self.assertTrue(should_recognize_menu_pinch(
            normalized_gap=0.35, finger_straightness=[0.48, 0.91, 0.77, 0.54]))

    def test_pinch_rejects_fist_and_non_contact(self):
        self.assertFalse(should_recognize_menu_pinch(
            normalized_gap=0.30, finger_straightness=[0.42, 0.48, 0.43, 0.39]))
        self.assertFalse(should_recognize_menu_pinch(
            normalized_gap=0.72, finger_straightness=[0.70, 0.95, 0.94, 0.92]))

    def test_pinch_detector_can_override_broad_grip_classifier(self):
        self.assertIn('if side == "left" and raw_pinch:', self.main)
        self.assertIn("if clear_left_pinch:\n                raw_fist = False", self.main)
        self.assertIn("elif raw_fist:\n                # A fist/grip keeps priority over pinch recognition", self.main)

    def test_short_pinch_recognition_dropout_preserves_timer(self):
        self.assertIn("self.menu_pinch_sequence_gap_seconds = 0.80", self.main)
        self.assertIn("if release_for <= self.menu_pinch_sequence_gap_seconds:\n                    continue", self.main)
        self.assertIn("held >= self.menu_pinch_hold_seconds", self.main)

    def test_progressive_scroll_gain_increases_with_travel(self):
        values = [progressive_scroll_gain(distance) for distance in (0, 30, 90, 180, 360, 720)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0] * 3)

    def test_overlay_reasserts_topmost_without_focus(self):
        self.assertIn("Qt.WindowDoesNotAcceptFocus", self.main)
        self.assertIn("Qt.WA_TransparentForMouseEvents", self.main)
        self.assertIn("now-self._last_raise > 0.12", self.main)
        self.assertIn("now-self._last_native_raise > 0.12", self.main)
        self.assertIn("wintypes.HWND(-1)", self.main)
        self.assertIn("self.search_hand_overlay.sync_to_launcher()", self.main)

    def test_overlay_paints_menu_before_final_tli(self):
        self.assertIn("owner._paint_pinch_menu(painter, side, now)", self.main)
        self.assertIn("owner._paint_gesture_timer_fade(painter, side, now)", self.main)
        self.assertLess(self.main.index("owner._paint_pinch_menu(painter, side, now)"), self.main.index("Repaint the TLI last so it remains visible over both launcher and menu buttons."))
        self.assertIn("Repaint the TLI last so it remains visible over both launcher and menu buttons.", self.main)
        self.assertIn('dialog is not None and dialog.isVisible()', self.main)

    def test_focus_dictation_streams_partials_and_end_phrase(self):
        self.assertIn('def begin_focused_dictation(self):', self.main)
        self.assertIn('def _stream_focused_dictation_partial(self, text):', self.main)
        self.assertIn('"end dictation"', self.main)
        self.assertIn('self._stream_focused_dictation_partial(text)', self.main)
        self.assertIn('self.end_focused_dictation(text)', self.main)

    def test_voice_commands_and_pinch_menu_switches(self):
        self.assertIn('"begin dictation"', self.main)
        self.assertIn('"dictation_toggle":', self.main)
        self.assertIn('"stt_toggle":', self.main)
        self.assertIn('elif action == "dictation_toggle":', self.main)
        self.assertIn('elif action == "stt_toggle":', self.main)

    def test_docs_explain_live_overlay_dictation_and_timer(self):
        self.assertIn("visible above the search launcher", self.main)
        self.assertIn("default 0.6", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("begin dictation", (ROOT / "VOICE_SETUP.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
