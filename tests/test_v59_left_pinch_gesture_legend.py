"""Regression tests for left pinch/fist disambiguation and the always-visible menu legend."""
from pathlib import Path
import unittest

from gesture_rules import should_prefer_clear_left_pinch_over_fist

ROOT = Path(__file__).resolve().parent.parent


class V59LeftPinchAndGestureLegendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_clear_tight_left_pinch_can_override_an_ambiguous_fist(self):
        self.assertTrue(should_prefer_clear_left_pinch_over_fist(
            is_pinch=True, normalized_gap=0.25))

    def test_loose_or_non_overlapping_pose_cannot_override_fist(self):
        self.assertFalse(should_prefer_clear_left_pinch_over_fist(
            is_pinch=True, normalized_gap=0.50))
        self.assertFalse(should_prefer_clear_left_pinch_over_fist(
            is_pinch=False, normalized_gap=0.20))

    def test_only_left_hand_gets_pinch_disambiguation_right_fist_keeps_priority(self):
        start = self.main.index("            clear_left_pinch = False")
        end = self.main.index("            if raw_fist:\n                self.fist_pose_confirmation_frames", start)
        block = self.main[start:end]
        self.assertIn('if side == "left" and raw_pinch:', block)
        self.assertIn("elif raw_fist:", block)
        self.assertIn('"clear_left_pinch": clear_left_pinch', self.main)
        self.assertIn("and not clear_left_pinch", self.main)
        self.assertIn('elif left_pose.get("pinch", False) and left_pose.get("clear_left_pinch", False):', self.main)

    def test_normal_gesture_legend_is_painted_while_glass_menu_is_open(self):
        self.assertIn("if self.menu_open and self.hub_visible:\n            # Keep the normal gesture legend visible", self.main)
        self.assertIn(
            '"LEFT HAND FIST: MODIFIER  ·  LEFT OR RIGHT HAND PINCH: CLICK  ·  GRAB & DRAG: SCROLL PAGE"',
            self.main)
        self.assertIn("gesture_legend", self.main)


if __name__ == "__main__":
    unittest.main()
