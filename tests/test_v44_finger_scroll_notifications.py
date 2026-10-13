"""Regression checks for finger thickness, gentle grip scrolling, and notification layout."""
from pathlib import Path
import unittest

from gesture_rules import progressive_scroll_gain, smooth_scroll_delta

ROOT = Path(__file__).resolve().parent.parent


class V44FingerScrollNotificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")

    def test_finger_thickness_is_a_saved_settings_value_with_safe_bounds(self):
        self.assertIn('"finger_thickness": 4.24', self.main)
        self.assertIn('"finger_thickness": (0.8, 10.0)', self.main)
        self.assertIn("finger_thickness={clamped('finger_thickness', 0.8, 10.0, 4.24):.2f}", self.main)
        self.assertIn('"finger_thickness": self.finger_thickness', self.main)

    def test_finger_thickness_slider_is_drawn_hit_tested_and_saved(self):
        self.assertIn('("finger_thickness", -62, self.finger_thickness, 0.8, 10.0, "FINGER THICKNESS")', self.main)
        self.assertIn('"finger_thickness", -62,', self.main)
        self.assertIn('"finger_thickness": ("finger_thickness", 0.01),', self.main)
        self.assertIn('setattr(self, attr, value)', self.main)
        self.assertIn('finger_thickness = min(10.0, max(0.8, float(getattr(self, "finger_thickness", 4.24))))', self.main)
        self.assertIn('finger_thickness if (a, b) in finger_edges else 1.0', self.main)

    def test_scroll_starts_from_live_position_and_smooths_spikes(self):
        self.assertIn('"last_cursor": QPointF(cursor if (mode == "scroll_window" or not automatic_motion) else origin)', self.main)
        self.assertIn('smoothed_dy = smooth_scroll_delta(', self.main)
        self.assertIn('previous_weight=0.68, max_delta=22.0', self.main)
        self.assertIn('active["scroll_remainder"] += edge_rate', self.main)
        self.assertNotIn('active["scroll_remainder"] += edge_steps * 44.0', self.main)

    def test_scroll_delta_is_clamped_and_low_pass_filtered(self):
        result = smooth_scroll_delta(0.0, 1000.0, previous_weight=0.68, max_delta=22.0)
        self.assertAlmostEqual(result, 7.04, places=2)
        spike_after_motion = smooth_scroll_delta(result, -1000.0, previous_weight=0.68, max_delta=22.0)
        self.assertLess(abs(spike_after_motion), abs(result))
        self.assertEqual(smooth_scroll_delta(float('nan'), 4.0), 0.0)

    def test_scroll_gain_starts_slow_and_ramps_gradually(self):
        values = [progressive_scroll_gain(distance) for distance in (0, 30, 90, 180, 360, 720)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], 0.12, places=3)
        self.assertLessEqual(values[-1], 0.78)
        self.assertLess(values[2] / values[0], 2.0)

    def test_all_notifications_use_a_shared_stack_per_anchor(self):
        self.assertIn('def _stacked_notification_rect(self, position, width, height, stack_offsets', self.main)
        self.assertIn('notification_stack_offsets = {}', self.main)
        self.assertGreaterEqual(self.main.count('_stacked_notification_rect('), 5)
        self.assertIn('offset + float(height) + float(gap)', self.main)

    def test_notification_fills_are_semi_transparent(self):
        self.assertIn('QColor(4, 22, 36, 158)', self.main)
        self.assertIn('QColor(5, 18, 32, 148)', self.main)
        self.assertIn('QColor(5, 34, 28, 158)', self.main)
        self.assertIn('QColor(38, 27, 12, 158)', self.main)

    def test_fixed_gesture_buffers_are_preserved(self):
        self.assertIn('self.gesture_timer_delay_seconds = 0.8  # FIXED soft gesture buffer: do not remove or shorten', self.main)
        self.assertIn('self.menu_pinch_sequence_gap_seconds = 0.80  # FIXED 0.8s soft gap for repeated pinch/re-arm', self.main)


if __name__ == "__main__":
    unittest.main()
