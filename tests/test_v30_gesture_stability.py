"""Regression tests for edge tracking, fist-shake window drag and release cooldown."""
from pathlib import Path
import unittest

from gesture_rules import detect_fist_shake, should_start_scroll_action, should_arm_shake_drag

ROOT = Path(__file__).resolve().parent.parent


class FistShakeTests(unittest.TestCase):
    def test_intentional_horizontal_shake_is_detected(self):
        xs = [0, 8, 20, 32, 22, 10, 0, 10, 25, 40, 28, 12, -3, 15, 30]
        samples = [(i * 0.045, float(x), 100.0 + (i % 2)) for i, x in enumerate(xs)]
        # The hand must settle after shaking before move mode is armed.
        for i in range(1, 7):
            samples.append((samples[-1][0] + 0.04, 30.0, 100.0))
        self.assertTrue(detect_fist_shake(samples, now=samples[-1][0]))

    def test_window_is_not_armed_mid_shake(self):
        xs = [0, 8, 20, 32, 22, 10, 0, 10, 25, 40, 28, 12, -3, 15, 30]
        samples = [(i * 0.045, float(x), 100.0) for i, x in enumerate(xs)]
        self.assertFalse(detect_fist_shake(samples, now=samples[-1][0]))

    def test_small_jitter_is_not_a_shake(self):
        xs = [0, 2, 4, 3, 5, 2, 4, 1, 3, 2]
        samples = [(i * 0.04, float(x), 50.0) for i, x in enumerate(xs)]
        self.assertFalse(detect_fist_shake(samples, now=samples[-1][0]))

    def test_old_shake_samples_expire(self):
        xs = [0, 10, 25, 40, 28, 12, -3, 15, 30]
        samples = [(i * 0.15, float(x), 40.0) for i, x in enumerate(xs)]
        self.assertFalse(detect_fist_shake(samples, now=4.0))


class ScrollArbitrationTests(unittest.TestCase):
    def test_vertical_content_movement_scrolls(self):
        self.assertTrue(should_start_scroll_action(
            held_seconds=1.0, confirmation_delay=0.8, dx=25, dy=135,
            already_armed=False, scroll_mode_enabled=False, hit_test=1))

    def test_horizontal_motion_does_not_become_scroll(self):
        self.assertFalse(should_start_scroll_action(
            held_seconds=1.0, confirmation_delay=0.8, dx=140, dy=15,
            already_armed=False, scroll_mode_enabled=False, hit_test=1))

    def test_title_bar_motion_never_becomes_scroll(self):
        self.assertFalse(should_start_scroll_action(
            held_seconds=1.0, confirmation_delay=0.8, dx=10, dy=180,
            already_armed=False, scroll_mode_enabled=False, hit_test=2))

    def test_scrollbar_can_start_with_shorter_vertical_motion(self):
        self.assertTrue(should_start_scroll_action(
            held_seconds=0.9, confirmation_delay=0.8, dx=3, dy=42,
            already_armed=False, scroll_mode_enabled=False, hit_test=7))

    def test_shake_does_not_arm_on_minimise_maximise_or_close_controls(self):
        for hit_test in (8, 9, 20):
            self.assertFalse(should_arm_shake_drag(hit_test=hit_test, has_target_window=True))
        self.assertTrue(should_arm_shake_drag(hit_test=1, has_target_window=True))
        self.assertFalse(should_arm_shake_drag(hit_test=1, has_target_window=False))

    def test_stability_wiring(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("self.active_release_cooldown_seconds = 0.20", source)
        self.assertIn("self.gesture_timer_delay_seconds = 0.8", source)
        self.assertIn("self.interactive_visual_point(points[index][0], points[index][1])", source)
        self.assertIn('force_mode="drag_window"', source)
        self.assertIn('MOVE MODE  ·  SHAKE COMPLETE', source)
        self.assertIn("edge_tolerant=edge_tolerant", source)
        self.assertIn("_is_native_taskbar_point", source)
        self.assertIn("self.fist_start_hit_test[side] = self._native_nonclient_hit_test", source)


if __name__ == "__main__":
    unittest.main()
