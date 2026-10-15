"""Regression checks for Freehand enable/disable voice commands and Gestures UI polish."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FreehandControlsVoiceUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_quit_button_uses_requested_label(self):
        self.assertIn('painter.drawText(exit_banner, Qt.AlignCenter, "Quit Freehand")', self.main)
        self.assertNotIn('painter.drawText(exit_banner, Qt.AlignCenter, "EXIT")', self.main)

    def test_enable_disable_freehand_commands_run_before_dictation(self):
        handler_start = self.main.index("    def on_speech(self, text):")
        handler_end = self.main.index("    def apply_desktop_mode", handler_start)
        handler = self.main[handler_start:handler_end]
        self.assertIn('"enable freehand"', handler)
        self.assertIn('"disable freehand"', handler)
        self.assertIn("self.set_freehand_gestures_enabled(True)", handler)
        self.assertIn("self.set_freehand_gestures_enabled(False)", handler)
        self.assertLess(handler.index('"disable freehand"'), handler.index("dialog.set_dictation_text(text, final=True)"))

    def test_freehand_voice_toggle_controls_both_gesture_sides_not_camera(self):
        start = self.main.index("    def set_freehand_gestures_enabled(")
        end = self.main.index("    def set_hand_tracking_paused(", start)
        helper = self.main[start:end]
        self.assertIn('sides = ("left", "right")', helper)
        self.assertIn("self.gesture_tracking_enabled[side] = enabled", helper)
        self.assertIn("self._cancel_side_interactions_for_tracking_pause(side)", helper)
        self.assertNotIn("self.tracker.set_hand_tracking_enabled", helper)
        self.assertIn("self.thumbs_up_toggle_latched[side] = bool(thumb_pose_recent)", helper)

    def test_command_prefixes_are_held_out_of_dictation(self):
        start = self.main.index("    def _partial_may_be_voice_command(")
        end = self.main.index("    def _type_focus_text(", start)
        method = self.main[start:end]
        self.assertIn('"enable freehand"', method)
        self.assertIn('"disable freehand"', method)

    def test_custom_gesture_delete_control_is_square(self):
        match = re.search(r'"delete": r\(delete_x, y\+3, ([0-9.]+), ([0-9.]+)\)', self.main)
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), ("32", "32"))

    def test_small_gesture_controls_have_hover_and_active_highlights(self):
        self.assertIn("def draw_control_glow(rect, rgb=(100, 255, 255))", self.main)
        self.assertIn('side_hovered = draw_control_glow(row["side"])', self.main)
        self.assertIn('timer_hovered = draw_control_glow(row["timer"])', self.main)
        self.assertIn('elif capture_enabled:', self.main)
        self.assertIn('draw_control_glow(row["capture"])', self.main)
        self.assertIn('capture_enabled = action == "custom_key"', self.main)
        self.assertIn('if gesture.get("action") == "custom_key":', self.main)
        self.assertIn('delete_hovered = draw_control_glow(row["delete"], (255, 75, 85))', self.main)
        self.assertIn("if capturing:", self.main)


if __name__ == "__main__":
    unittest.main()
