"""Regression checks for scroll acceleration, main-menu Close, and timer default."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V60ScrollCloseTimerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_scroll_acceleration_slider_is_persisted_and_controls_distance_ramp(self):
        self.assertIn('("scroll_acceleration", 101, self.scroll_acceleration, 0.5, 10.0, "SCROLL ACCELERATION")', self.main)
        self.assertIn('"scroll_acceleration": (0.5, 10.0)', self.main)
        self.assertIn('"scroll_acceleration": self.scroll_acceleration', self.main)
        self.assertIn('"scroll_acceleration": ("scroll_acceleration", 0.01)', self.main)
        self.assertIn('ramp_distance=17.0/max(0.1, self.scroll_acceleration)', self.main)
        self.assertIn('scroll_acceleration=1.00', self.gestures)
        self.assertIn('default `1.0×`', self.readme)
        self.assertIn('\"interaction\": 700.0', self.main)

    def test_main_glass_menu_has_wide_close_button_and_click_action(self):
        self.assertIn('def _main_menu_close_rect(', self.main)
        self.assertIn('cy+(panel_height*0.5-54.0)*scale', self.main)
        self.assertIn('painter.drawText(close_banner, Qt.AlignCenter, "CLOSE")', self.main)
        self.assertIn('if self._main_menu_close_rect(cx, cy, scale).contains(point):', self.main)
        self.assertIn('self._set_menu_open(False, now, reset_settings=True)', self.main)

    def test_main_glass_menu_quit_freehand_button_is_above_tiles_and_closes_application(self):
        self.assertIn('def _main_menu_exit_rect(', self.main)
        self.assertIn('cy+(18.0-panel_height*0.5)*scale', self.main)
        self.assertIn('painter.drawText(exit_banner, Qt.AlignCenter, "Quit Freehand")', self.main)
        self.assertIn('if self._main_menu_exit_rect(cx, cy, scale).contains(point):', self.main)
        self.assertIn('self.close()', self.main)
        self.assertIn('self.status = "Exiting Freehand: Spatial Computing"', self.main)
        self.assertIn('press_rects.append(self._main_menu_exit_rect(pcx, pcy, pscale))', self.main)

    def test_timer_default_is_06_and_bundled_value_does_not_override_it(self):
        self.assertIn('"menu_pinch_hold_seconds": 0.6', self.main)
        self.assertIn('"menu_pinch_hold_seconds": 0.6,', self.main)
        values = re.findall(r'^menu_pinch_hold_seconds=([0-9.]+)$', self.gestures, re.MULTILINE)
        self.assertEqual(values, ["0.60"])
        self.assertIn('default 0.6', self.readme)

    def test_close_banner_only_appears_on_top_level_menu(self):
        paint_marker = 'if not self.settings_open:\n                close_banner = self._main_menu_close_rect'
        click_marker = 'if not self.settings_open:\n                if self._main_menu_close_rect(cx, cy, scale).contains(point):'
        self.assertIn(paint_marker, self.main)
        self.assertIn(click_marker, self.main)
        self.assertLess(self.main.index(paint_marker), self.main.index('labels = [', self.main.index(paint_marker)))


if __name__ == "__main__":
    unittest.main()
