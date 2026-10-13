"""Regression checks for v39 3D modifier, paused-hand cursor and notification timers."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V39TrackingModifierTimersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.ui = (ROOT / "ui_settings.json").read_text(encoding="utf-8")

    def test_modifier_has_virtual_context_menu_and_canvas_drag(self):
        self.assertIn('"3D context menu opened at the TLI', self.main)
        self.assertIn('self.fist_action.get("right") == "canvas_drag"', self.main)
        self.assertIn('self.desktop_mode != "os"', self.main)
        self.assertIn('"3D canvas/app drag active', self.main)

    def test_peace_toggle_detaches_os_mouse_only_and_keeps_gestures_live(self):
        self.assertIn('self.mouse_detached_by_side = {"left": False, "right": False}', self.main)
        self.assertIn('self.mouse_detached_by_side[side] = not self.mouse_detached_by_side.get(side, False)', self.main)
        self.assertNotIn('cursor_points.pop(side, None)', self.main)
        self.assertNotIn('self._cancel_side_interactions_for_tracking_pause(side)', self.main[self.main.index('if peace:'):self.main.index('raw_fist = is_fist_or_grip_pose')])
        self.assertIn('self.mouse_detached_by_side.get(mouse_side, False)', self.main)
        self.assertIn('self.side_cursor_positions[side] = QPointF(cursor)', self.main)

    def test_disabled_announcement_is_visual_for_five_seconds_and_skips_tts(self):
        self.assertIn('self.hand_tracking_notice_until = now + 5.0', self.main)
        self.assertIn('self.hand_tracking_notice_text.upper()', self.main)
        self.assertIn('self._refresh_hand_tracking_notice(now, show_enabled=False)', self.main)
        self.assertIn('self._refresh_hand_tracking_notice(now, show_enabled=True)', self.main)
        self.assertNotIn('self.tts.speak(f"{side} hand tracking enabled")', self.main)
        self.assertIn('self.hand_tracking_notice_until = float("inf")', self.main)
        self.assertIn('mouse detached', self.main)

    def test_modifier_notice_uses_configured_anchor_and_blur_is_stronger(self):
        self.assertIn('modifier_badge = self._stacked_notification_rect(', self.main)
        self.assertIn('self._blur_glass_image(image, sigma=40.0)', self.main)
        self.assertIn('self._blur_glass_image(image, sigma=36.0)', self.main)
        self.assertIn('"paused_notice_position": "bottom-center"', self.ui)
        self.assertIn('"heard_bar_position": "bottom-center"', self.ui)

    def test_general_gesture_countdowns_show_numeric_seconds(self):
        self.assertGreaterEqual(self.main.count('f"{remaining:.1f}s"'), 3)
        self.assertIn('total_duration = self.gesture_timer_delay_seconds + duration', self.main)
        self.assertIn('total_duration = delay + duration', self.main)

    def test_documentation_updated(self):
        self.assertIn('virtual canvas/app', self.guide)
        self.assertIn('detached notice persists until reattached', self.readme)
        self.assertIn('Spatial Desktop v40', self.readme)


if __name__ == "__main__":
    unittest.main()
