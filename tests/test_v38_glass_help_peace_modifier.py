"""Source-level regression checks for v38 glass-menu help and gesture updates."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V38GlassHelpPeaceModifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_help_tiles_navigate_to_independent_glass_pages(self):
        self.assertIn('self.settings_page = "gesture_map"', self.main)
        self.assertIn('self.settings_page = "speech_commands"', self.main)
        self.assertIn('"gesture_map": "GESTURE MAP"', self.main)
        self.assertIn('"speech_commands": "SPEECH COMMANDS"', self.main)
        self.assertIn('if self.settings_page == "gesture_map":', self.main)
        self.assertIn('elif self.settings_page == "speech_commands":', self.main)
        self.assertIn('if self.settings_page not in ("gesture_map", "speech_commands"):', self.main)

    def test_help_tiles_no_longer_launch_system_message_boxes(self):
        menu_start = self.main.index('if action == "desktop":')
        menu_end = self.main.index('return  # every press in the glass panel is consumed', menu_start)
        menu_route = self.main[menu_start:menu_end]
        self.assertNotIn('QMessageBox.information', menu_route)

    def test_peace_toggle_has_countdown_and_immediate_visual_state_without_tts(self):
        self.assertIn('self.peace_toggle_hold_seconds = 1.5', self.main)
        self.assertIn('elapsed/max(0.01, self.peace_toggle_hold_seconds)', self.main)
        self.assertIn('elapsed/max(0.01, owner.peace_toggle_hold_seconds)', self.main)
        self.assertIn('self._refresh_hand_tracking_notice(now, show_enabled=True)', self.main)
        self.assertNotIn('self.tts.speak(f"{side} hand tracking enabled")', self.main)
        self.assertIn('f"{remaining:.1f}s"', self.main)

    def test_modifier_pinch_right_click_and_grip_drags(self):
        self.assertIn('self.modifier_click_kind[side] = "pinch_right_click"', self.main)
        self.assertIn('self._activate_modifier_right_click(origin if origin is not None else cursor, now)', self.main)
        self.assertIn('self.modifier_click_kind[side] = "grip_drag"', self.main)
        self.assertIn('active["modifier_drag"] = True', self.main)
        self.assertIn('release_delay = 0.14 if active.get("modifier_drag")', self.main)

    def test_tli_is_midpoint_anchored_and_docs(self):
        self.assertIn('tx, ty = xy(landmarks[4])', self.main)
        self.assertIn('ix, iy = xy(landmarks[8])', self.main)
        self.assertIn('return (tx+ix)*0.5, (ty+iy)*0.5', self.main)
        self.assertIn('RIGHT pinch/release = right-click', self.guide)
        self.assertIn('PEACE-SIGN MOUSE-DETACH TOGGLE', self.guide)
        self.assertIn('Help pages are in the glass menu', self.readme)
        self.assertIn('Peace-sign toggle', self.readme)


if __name__ == "__main__":
    unittest.main()
