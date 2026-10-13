"""Regression checks for the v53 Hands tab, settings defaults and isolated TLI calibration."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V53HandsTabCalibrationWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.settings = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_requested_defaults_are_loaded_and_persisted(self):
        self.assertIn('"finger_thickness": 4.24', self.main)
        self.assertIn('"cursor_smoothing": 0.9', self.main)
        self.assertIn('"hand_tracking_sensitivity": 1.5', self.main)
        for setting in (
            "finger_thickness=4.24",
            "cursor_smoothing=0.90",
            "hand_tracking_sensitivity=1.50",
            "hand_visual_scale=0.30",
        ):
            self.assertIn(setting, self.settings)

    def test_hands_is_a_separate_tab_with_palm_width_and_style_toggle(self):
        for token in ('"hands": QRectF', '"hands": "HANDS"', '"hands": 760.0'):
            self.assertIn(token, self.main)
        self.assertIn('("palm_width_scale", -5, self.palm_width_scale, 0.65, 1.8, "PALM WIDTH")', self.main)
        self.assertIn('"hand_skin_effect": True', self.main)
        self.assertIn('self.hand_skin_effect_enabled = not self.hand_skin_effect_enabled', self.main)
        self.assertIn('"hand_skin_effect": self.hand_skin_effect_enabled', self.main)
        self.assertIn('elif key == "hand_skin_effect":', self.main)  # boolean parser recognizes the key

    def test_interaction_tab_exposes_tli_appearance_sliders(self):
        self.assertIn('("tli_glow_intensity", 52, self.tli_glow_intensity, 0.0, 2.5, "TLI GLOW INTENSITY")', self.main)
        self.assertIn('("tli_colour_shift_degrees", 109, self.tli_colour_shift_degrees, 0.0, 360.0, "TLI COLOUR / HUE")', self.main)
        self.assertIn('("tli_border_thickness", 166, self.tli_border_thickness, 1.5, 8.0, "TLI BORDER THICKNESS")', self.main)
        self.assertIn('self.tli_glow_intensity', self.main)
        self.assertIn('self.tli_colour_shift_degrees', self.main)
        self.assertIn('self.tli_border_thickness', self.main)

    def test_settings_panel_height_is_page_dependent_and_scaled_to_screen(self):
        self.assertIn('def _menu_panel_dimensions(self):', self.main)
        self.assertIn('"interaction": 760.0', self.main)
        self.assertIn('"hands": 760.0', self.main)
        self.assertIn('fit_h = max(0.4, (self.height()-36.0)/panel_height)', self.main)
        self.assertIn('panel_width, panel_height = self._menu_panel_dimensions()', self.main)

    def test_calibration_opens_a_distinct_window_with_a_bounded_pinch_target(self):
        self.assertIn('class TliCalibrationWindow(QWidget):', self.main)
        self.assertIn('Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool', self.main)
        self.assertIn('def pinch_area_global_rect(self):', self.main)
        self.assertIn('return window.pinch_area_global_rect().contains(QPointF(float(global_point.x()), float(global_point.y())))', self.main)
        self.assertIn('window.show()', self.main)
        self.assertIn('window.close()', self.main)
        self.assertIn('window.deleteLater()', self.main)

    def test_calibration_window_suppresses_other_actions_until_finished(self):
        self.assertIn('calibration_modal_frame = self.tli_calibration_active_side is not None', self.main)
        self.assertIn('if calibration_modal_frame or calibration_consumes_gesture:', self.main)
        self.assertIn('if calibration_modal_frame:', self.main)
        self.assertIn('Do not move the OS pointer or pass poses into any normal gesture handler', self.main)
        self.assertIn('self.action_stop_wait_open[tracked_side] = True', self.main)

    def test_calibration_only_collects_one_continuous_pinch_inside_target(self):
        self.assertIn('if not is_pinch:', self.main)
        self.assertIn('self.tli_calibration_samples[side] = []', self.main)
        self.assertIn('if not self._calibration_pinch_inside_window(landmarks):', self.main)
        self.assertIn('self.tli_calibration_sample_count = 12', self.main)
        self.assertIn('mean_relative_tli(samples)', self.main)

    def test_render_style_switches_between_connected_skin_and_skeleton(self):
        self.assertIn('if not self.hand_skin_effect_enabled:', self.main)
        self.assertIn('"RENDER STYLE  ·  "+("SKIN EFFECT" if self.hand_skin_effect_enabled else "SKELETON")', self.main)
        self.assertIn('palm_width_scale-1.0', self.main)


if __name__ == "__main__":
    unittest.main()
