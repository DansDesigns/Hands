"""Regression checks for v48 TLI interaction scaling and startup defaults."""
from pathlib import Path
import unittest

from gesture_rules import scale_hand_point

ROOT = Path(__file__).resolve().parent.parent


class V48TLIInteractionScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_actual_interaction_point_uses_hand_transform(self):
        self.assertIn("def _scale_hand_tli_point(self, cursor, points=None, side=None):", self.main)
        self.assertIn("mapped_anchor = self._scale_hand_tli_point(", self.main)
        self.assertIn("cursor = self._smooth_cursor(side, mapped_anchor, now)", self.main)
        self.assertIn("return QPointF(origin), visual_scale", self.main)
        self.assertIn("scale_hand_point((q.x(), q.y()), (scale_origin.x(), scale_origin.y()), user_scale)", self.main)
        self.assertIn("cursor_points[side] = QPointF(cursor)", self.main)
        self.assertIn("self.side_cursor_positions[side] = QPointF(cursor)", self.main)
        self.assertIn("self.smoothed_side_cursor_positions[side] = QPointF(result)", self.main)
        # Rendering must not apply the transform twice; contact can pivot toward real fingertips.
        self.assertIn("side_cursor_positions now stores the transformed interaction point", self.main)
        self.assertIn("shared_contact_origins", self.main)
        self.assertIn("render touching at any hand scale", self.main)


    def test_wrist_pivot_keeps_wrist_fixed_when_grip_changes_fingertip_positions(self):
        wrist = (400.0, 300.0)
        open_fingertip = (430.0, 180.0)
        gripped_fingertip = (470.0, 250.0)
        scale = 0.3
        mapped_wrist = scale_hand_point(wrist, wrist, scale)
        mapped_open_tip = scale_hand_point(open_fingertip, wrist, scale)
        mapped_gripped_tip = scale_hand_point(gripped_fingertip, wrist, scale)
        self.assertEqual(mapped_wrist, wrist)
        self.assertAlmostEqual(mapped_open_tip[0], 409.0)
        self.assertAlmostEqual(mapped_open_tip[1], 264.0)
        self.assertAlmostEqual(mapped_gripped_tip[0], 421.0)
        self.assertAlmostEqual(mapped_gripped_tip[1], 285.0)

    def test_hand_size_defaults_to_minimum(self):
        self.assertIn('"hand_visual_scale": 0.3,', self.main)
        self.assertIn("clamped('hand_visual_scale', 0.1, 1.0, 0.3)", self.main)
        self.assertRegex(self.gestures, r"(?m)^hand_visual_scale=0\.30$")

    def test_startup_mode_is_os_desktop(self):
        self.assertIn('self.desktop_mode = "os"', self.main)
        self.assertIn("startup mode is OS Desktop", self.readme)


if __name__ == "__main__":
    unittest.main()
