"""Regression checks for the Hands-page disabled-gesture indicator toggle."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DisabledIndicatorVisibilityToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_preference_defaults_on_and_round_trips_as_boolean(self):
        self.assertIn('"show_gesture_disabled_indicator": True', self.main)
        self.assertIn('elif key == "show_gesture_disabled_indicator":', self.main)
        self.assertIn('if isinstance(value, bool):', self.main)
        self.assertIn('self.ui_preferences.get("show_gesture_disabled_indicator", True)', self.main)

    def test_toggle_is_on_hands_page_and_persists_immediately(self):
        layout = self.main[self.main.index('elif page == "hands":', self.main.index('def _settings_layout')):
                           self.main.index('elif page == "tli":', self.main.index('def _settings_layout'))]
        self.assertIn('stack("disabled_indicator_toggle")', layout)
        self.assertIn('if pressed("disabled_indicator_toggle"):', self.main)
        self.assertIn('save_ui_preferences(self.ui_preferences)', self.main)
        self.assertIn('"DISABLED GESTURE INDICATOR  ·  "+("ON" if indicator_enabled else "OFF")', self.main)

    def test_indicator_paint_obeys_visibility_preference_in_all_overlays(self):
        start = self.main.index('    def _paint_gesture_disabled_indicator(')
        end = self.main.index('    def _paint_hand_skeleton(', start)
        helper = self.main[start:end]
        self.assertIn('not getattr(self, "show_gesture_disabled_indicator", True)', helper)
        self.assertIn('owner._paint_gesture_disabled_indicator(painter, points, side)', self.main)
        self.assertIn('self._paint_gesture_disabled_indicator(painter, points, side)', self.main)


if __name__ == "__main__":
    unittest.main()
