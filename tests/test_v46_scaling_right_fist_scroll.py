"""Regression tests for expanded hand scale controls and timer-free right-fist scroll."""
from pathlib import Path
import unittest

from gesture_rules import should_start_right_fist_scroll

ROOT = Path(__file__).resolve().parent.parent


class V46HandScaleAndScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")
        cls.settings = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_hand_scale_slider_spans_0_3_to_1_and_rendering_honours_it(self):
        self.assertIn('"hand_visual_scale": (0.3, 1.0)', self.main)
        self.assertIn('"hand_visual_scale={clamped(\'hand_visual_scale\', 0.3, 1.0, 0.3):.2f}"', self.main)
        self.assertIn('("hand_scale", -119, self.hand_visual_scale, 0.3, 1.0, "HAND SCALE")', self.main)
        self.assertIn('value = minimum+ratio*(maximum-minimum)', self.main)
        self.assertIn('visual_scale = min(1.0, max(0.3, float(getattr(self, "hand_visual_scale", 0.3))))', self.main)
        self.assertIn('scale_origin, user_scale = self._hand_visual_transform(points, side)', self.main)
        self.assertIn('hand_visual_scale: 0.3–1.0', self.settings)

    def test_finger_thickness_slider_spans_0_8_to_10_and_renderer_honours_it(self):
        self.assertIn('"finger_thickness": (0.8, 10.0)', self.main)
        self.assertIn('finger_thickness={clamped(\'finger_thickness\', 0.8, 10.0, 4.24):.2f}', self.main)
        self.assertIn('("finger_thickness", -62, self.finger_thickness, 0.8, 10.0, "FINGER THICKNESS")', self.main)
        self.assertIn('value = minimum+ratio*(maximum-minimum)', self.main)
        self.assertIn('finger_thickness = min(10.0, max(0.8, float(getattr(self, "finger_thickness", 4.24))))', self.main)
        self.assertIn('finger_thickness: 0.8–10.0', self.settings)
        self.assertIn('closer = thicker; farther = thinner', self.settings)

    def test_only_right_hand_moving_fist_can_start_scroll(self):
        args = dict(is_fist=True, left_fist_active=False, modifier_active=False,
                    already_armed=False, dx=3.0, dy=15.0, hit_test=1)
        self.assertTrue(should_start_right_fist_scroll(side="right", **args))
        self.assertFalse(should_start_right_fist_scroll(side="left", **args))
        self.assertFalse(should_start_right_fist_scroll(side="right", **{**args, "left_fist_active": True}))
        self.assertFalse(should_start_right_fist_scroll(side="right", **{**args, "modifier_active": True}))
        self.assertFalse(should_start_right_fist_scroll(side="right", **{**args, "dy": 6.0}))
        self.assertFalse(should_start_right_fist_scroll(side="right", **{**args, "dx": 22.0, "dy": 15.0}))
        self.assertFalse(should_start_right_fist_scroll(side="right", **{**args, "hit_test": 2}))

    def test_right_fist_has_no_circle_timer_but_pinch_buffer_remains(self):
        self.assertIn('Only timed pinches show a progress circle', self.main)
        self.assertIn('Right-fist movement scrolls directly and intentionally has no fist timer ring.', self.main)
        self.assertIn('self.gesture_timer_delay_seconds = 0.8  # FIXED soft gesture buffer: do not remove or shorten', self.main)
        self.assertIn('self.menu_pinch_sequence_gap_seconds = 0.80  # FIXED 0.8s soft gap for repeated pinch/re-arm', self.main)
        self.assertIn('right-fist scrolling has no circle timer', self.settings.lower())

    def test_right_fist_scroll_arms_smoothed_scroll_path_without_waiting_for_timer(self):
        self.assertIn('proposed = should_start_right_fist_scroll(', self.main)
        self.assertIn('force_mode="scroll_window"', self.main)
        self.assertIn('smoothed_dy = smooth_scroll_delta(', self.main)
        self.assertIn('steps = int(active["scroll_remainder"] / 24.0)', self.main)
        self.assertIn('self.desktop_mode != "os" and force_mode != "scroll_window"', self.main)
        self.assertIn('movement-driven right-fist scrolling', self.readme.lower())


if __name__ == "__main__":
    unittest.main()
