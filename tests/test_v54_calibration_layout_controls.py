"""Regression coverage for v54 calibration visibility and new visual controls."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V54CalibrationLayoutControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")

    def test_calibration_target_interior_is_transparent_for_live_hand_overlay(self):
        self.assertIn("self.setAttribute(Qt.WA_TranslucentBackground, True)", self.main)
        self.assertIn("# stays transparent, allowing the live skeleton/skin hand overlay", self.main)
        self.assertIn("painter.setBrush(Qt.NoBrush)\n        painter.drawRoundedRect(area, 14, 14)", self.main)
        self.assertIn("def pinch_area_global_rect(self):", self.main)

    def test_calibration_reattaches_a_peace_detached_mouse(self):
        self.assertIn("if self.mouse_detached_by_side.get(side, False):", self.main)
        self.assertIn("self.mouse_detached_by_side[side] = False", self.main)
        self.assertIn("self.smoothed_side_cursor_positions[side] = None", self.main)
        self.assertIn("self._refresh_hand_tracking_notice(time.monotonic(), show_enabled=True)", self.main)

    def test_palm_width_default_is_1_26_and_persisted(self):
        self.assertIn('"palm_width_scale": 1.26', self.main)
        self.assertIn('settings.get("palm_width_scale", 1.26)', self.main)
        self.assertIn("palm_width_scale=1.26", self.gestures)
        self.assertIn("clamped('palm_width_scale', 0.65, 1.8, 1.26)", self.main)

    def test_tli_diameter_and_skeleton_thickness_are_loaded_saved_and_slidable(self):
        self.assertIn('"tli_diameter": 30.0', self.main)
        self.assertIn('("tli_diameter", 64, self.tli_diameter, 12.0, 96.0, "TLI DIAMETER")', self.main)
        self.assertIn('"skeleton_line_thickness": 1.5', self.main)
        self.assertIn('("skeleton_line_thickness", -60, self.skeleton_line_thickness, 0.5, 4.0, "SKELETON LINE THICKNESS")', self.main)
        self.assertIn('"tli_diameter": self.tli_diameter', self.main)
        self.assertIn('"skeleton_line_thickness": self.skeleton_line_thickness', self.main)
        self.assertIn('diameter = min(96.0, max(12.0, float(getattr(self, "tli_diameter", 30.0))))', self.main)
        self.assertIn('skeleton_width_scale = min(4.0, max(0.5, float(getattr(self, "skeleton_line_thickness", 1.5))))', self.main)

    def test_back_button_is_bottom_centered_and_pinched_menu_has_bottom_padding(self):
        self.assertIn("def _settings_back_rect(self, cx=None, cy=None, scale=None):", self.main)
        self.assertIn("cx-50.0*scale, cy+(panel_height*0.5-44.0)*scale", self.main)
        self.assertIn("panel_h = pad + 9.0*scale + item_h*3 + 2*gap + pad", self.main)

    def test_glass_menu_backgrounds_are_set_to_15_percent_transparency(self):
        self.assertIn("def _paint_frosted_backdrop(self, painter, rect, radius, opacity=0.15):", self.main)
        self.assertIn("self._paint_frosted_backdrop(painter, panel, 15, opacity=0.15)", self.main)
        self.assertIn("self._paint_frosted_backdrop(painter, panel, 28*scale, opacity=0.15)", self.main)
        self.assertIn("QColor(4, 12, 28, 217)", self.main)
        self.assertIn("QColor(5, 17, 35, 217)", self.main)

    def test_slider_spacing_added_below_tabs(self):
        self.assertIn('("sensitivity", -218, self.hand_tracking_sensitivity', self.main)
        self.assertIn('"interaction": 700.0', self.main)
        self.assertIn('"hands": 650.0', self.main)


if __name__ == "__main__":
    unittest.main()
