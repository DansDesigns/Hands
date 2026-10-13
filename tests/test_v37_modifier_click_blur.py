"""Regression checks for the two-hand modifier click and frosted glass updates."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V37ModifierClickBlurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_left_fist_modifier_is_debounced_and_released_by_open_hand(self):
        self.assertIn("def _update_left_click_modifier(self, poses, now):", self.main)
        self.assertIn("self.left_modifier_fist_frames >= 2", self.main)
        self.assertIn("self.left_modifier_open_frames >= 2", self.main)

    def test_right_modifier_pinch_is_a_right_click_and_grip_is_a_drag(self):
        self.assertIn("def _handle_modifier_click_frame", self.main)
        self.assertIn('self._clear_menu_pinch_sequence("right")', self.main)
        self.assertIn('self.modifier_click_kind[side] = "pinch_right_click"', self.main)
        self.assertIn('self._activate_modifier_right_click(origin if origin is not None else cursor, now)', self.main)
        self.assertIn('self.modifier_click_kind[side] = "grip_drag"', self.main)
        self.assertIn('active["modifier_drag"] = True', self.main)

    def test_normal_double_click_stabilisation_remains_available(self):
        self.assertIn("<= 52):", self.main)
        self.assertIn('self.last_quick_grasp[side] = (now, QPointF(click_point))', self.main)

    def test_frosted_panels_use_a_gaussian_blurred_clipped_backdrop(self):
        self.assertIn("cv2.GaussianBlur", self.main)
        self.assertIn("def _paint_frosted_backdrop", self.main)
        self.assertIn("painter.setClipPath(path)", self.main)
        self.assertIn("QColor(4, 12, 28, 217)", self.main)
        self.assertGreaterEqual(self.main.count("self._capture_glass_backdrop()"), 1)
        self.assertIn('if image is None or image.isNull():\n            image = self._virtual_glass_backdrop()', self.main)
        self.assertIn('Do not synchronously grab/blur the screen here', self.main)

    def test_user_guides_document_the_new_gesture(self):
        self.assertIn("LEFT MODIFIER", self.main)
        self.assertIn("LEFT MODIFIER", self.guide)
        self.assertIn("v37 frosted backdrop, v38 in-menu help and modifier drag", self.readme)


if __name__ == "__main__":
    unittest.main()
