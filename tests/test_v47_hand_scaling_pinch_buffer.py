"""Regression tests for v47 hand/TLI alignment and buffered pinch timers."""
from pathlib import Path
import unittest

from gesture_rules import (
    pinch_timer_elapsed_seconds,
    pinch_timer_pulse_seconds,
    pinch_timer_release_before_buffer,
    pinch_timer_gap_restarts_buffer,
)

ROOT = Path(__file__).resolve().parent.parent


class V47HandScalingPinchBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_hand_and_tli_share_the_same_scale_transform(self):
        self.assertIn('def _hand_visual_transform(self, points, side=None):', self.main)
        self.assertIn('def _visual_tli_position(self, side, cursor, points=None):', self.main)
        self.assertIn('scale_origin, user_scale = self._hand_visual_transform(points, side)', self.main)
        self.assertIn('cursor = self._visual_tli_position(side, cursor, points)', self.main)
        self.assertIn('"hand_visual_scale": (0.1, 1.0)', self.main)
        self.assertIn('value = minimum+ratio*(maximum-minimum)', self.main)

    def test_finger_thickness_reaches_10x_in_settings_and_renderer(self):
        self.assertIn('"finger_thickness": (0.8, 10.0)', self.main)
        self.assertIn('value = minimum+ratio*(maximum-minimum)', self.main)
        self.assertIn('finger_thickness = min(10.0, max(0.8, float(getattr(self, "finger_thickness", 4.24))))', self.main)

    def test_pinch_timer_has_an_actual_continuous_08_second_buffer(self):
        self.assertIn('def _menu_pinch_elapsed_seconds(self, side, now=None):', self.main)
        self.assertIn('pinch_timer_elapsed_seconds(', self.main)
        self.assertIn('pinch_timer_gap_restarts_buffer(', self.main)
        self.assertIn('pinch_timer_release_before_buffer(', self.main)
        self.assertIn('remaining = buffer_remaining if buffer_remaining > 0.0 else max(0.0, duration-elapsed)', self.main)
        self.assertAlmostEqual(pinch_timer_elapsed_seconds(10.0, 0, 10.0, None, 10.79), 0.0)
        self.assertAlmostEqual(pinch_timer_elapsed_seconds(10.0, 0, 10.0, None, 11.0), 0.2)

    def test_quick_double_pinch_cannot_pool_time_before_buffer(self):
        # First pulse releases at 0.24s; the second pinch is a new sequence, not old-time credit.
        self.assertTrue(pinch_timer_release_before_buffer(10.0, 10.24))
        self.assertTrue(pinch_timer_gap_restarts_buffer(10.0, 10.24, 10.35))
        self.assertAlmostEqual(pinch_timer_elapsed_seconds(10.35, 0, 10.35, None, 11.14), 0.0)
        self.assertAlmostEqual(pinch_timer_elapsed_seconds(10.35, 0, 10.35, None, 11.20), 0.05)

    def test_pulse_contributions_exclude_buffer_and_release_gaps(self):
        self.assertAlmostEqual(pinch_timer_pulse_seconds(10.0, 10.0, 10.5), 0.0)
        self.assertAlmostEqual(pinch_timer_pulse_seconds(10.0, 10.0, 11.1), 0.3)
        # A released pulse contributes only until release, and never counts the gap afterward.
        self.assertAlmostEqual(pinch_timer_elapsed_seconds(10.0, 0.3, 11.2, 11.5, 11.9), 0.6)

    def test_documentation_calls_out_buffer_and_new_ranges(self):
        self.assertIn('hand_visual_scale: 0.1–1.0', self.guide)
        self.assertIn('finger_thickness: 0.8–10.0', self.guide)
        self.assertIn('v47 hand scaling and pinch timer fixes', self.readme)
        self.assertIn('release gaps and duplicate frames contribute nothing', self.readme)


if __name__ == "__main__":
    unittest.main()
