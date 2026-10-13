"""Regression checks for stationary-pinch menu vs moving-pinch native drag/drop."""
from pathlib import Path
import unittest

from gesture_rules import pinch_drag_should_start

ROOT = Path(__file__).resolve().parent.parent


class V55PinchDragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")

    def test_jitter_below_threshold_does_not_start_drag(self):
        self.assertFalse(pinch_drag_should_start((100, 100), (108, 111), 22))
        self.assertFalse(pinch_drag_should_start((100, 100), (121, 100), 22))
        self.assertFalse(pinch_drag_should_start(None, (200, 100), 22))

    def test_deliberate_movement_beyond_threshold_starts_drag(self):
        self.assertTrue(pinch_drag_should_start((100, 100), (122, 100), 22))
        self.assertTrue(pinch_drag_should_start((100, 100), (116, 116), 22))

    def test_drag_path_uses_native_mouse_hold_and_release(self):
        self.assertIn("def _begin_pinch_drag(self, side, cursor, now):", self.main)
        self.assertIn('pyautogui.mouseDown(button="left", _pause=False)', self.main)
        self.assertIn("def _end_pinch_drag(self, side, status=True):", self.main)
        self.assertIn('pyautogui.mouseUp(button="left", _pause=False)', self.main)
        self.assertIn("self.pinch_drag_active = {side: False for side in sides}", self.main)
        self.assertIn("self.pinch_drag_release_candidate_since = {side: None for side in sides}", self.main)

    def test_motion_wins_only_before_timer_completion(self):
        self.assertIn("held < self.menu_pinch_hold_seconds", self.main)
        self.assertIn("pinch_drag_should_start(", self.main)
        self.assertIn("if held >= self.menu_pinch_hold_seconds and side not in self.menu_pinch_latched:", self.main)

    def test_drag_release_is_consumed_and_tracking_loss_releases_button(self):
        self.assertIn("if self.pinch_drag_active.get(side, False):", self.main)
        self.assertIn("self._end_pinch_drag(side)", self.main)
        self.assertIn("self._end_pinch_drag(side, status=False)", self.main)
        self.assertIn("now-self.pinch_drag_release_candidate_since[side] < 0.08", self.main)
        self.assertIn("Releasing the pinch drops the item and releases the mouse button", self.readme)

    def test_documentation_explains_stationary_menu_and_moving_drag(self):
        self.assertIn("stationary", self.readme.lower())
        self.assertIn("Pinch Menu", self.guide)
        self.assertIn("PINCH DRAG:", self.guide)


if __name__ == "__main__":
    unittest.main()
