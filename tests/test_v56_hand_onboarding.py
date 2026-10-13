"""Regression checks for first-run hand calibration and application-compatible scroll input."""
from pathlib import Path
import tempfile
import unittest

from hand_calibration import (
    default_calibration,
    fingertip_contact_ratio,
    load_hand_calibration,
    palm_measurements,
    proximity_thresholds,
    relative_palm_projection_scale,
    save_hand_calibration,
    screen_touch_offset,
)

ROOT = Path(__file__).resolve().parent.parent


def fake_hand(cx=0.5, cy=0.5, scale=1.0):
    """Return a simple 21-point hand-like normalized landmark cloud."""
    points = [(cx, cy, 0.0) for _ in range(21)]
    points[0] = (cx, cy+0.08*scale, 0.0)
    points[5] = (cx-0.035*scale, cy+0.015*scale, 0.0)
    points[9] = (cx, cy-0.01*scale, 0.0)
    points[17] = (cx+0.035*scale, cy+0.015*scale, 0.0)
    for tip, dx in ((4, -0.03), (8, 0.03), (12, 0.05), (16, 0.065), (20, 0.075)):
        points[tip] = (cx+dx*scale, cy-0.045*scale, 0.0)
    return points


class V56HandOnboardingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_palm_measurements_are_positive_and_scale_relative(self):
        a = palm_measurements(fake_hand(scale=1.0))
        b = palm_measurements(fake_hand(scale=1.7))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertGreater(b["size"], a["size"])
        self.assertAlmostEqual(relative_palm_projection_scale(b["size"], a["size"]), 1.7, places=1)

    def test_screen_touch_anchor_maps_measured_camera_point_to_target(self):
        offset = screen_touch_offset(0.5, 0.5, target_x=0.5, target_y=0.08)
        mapped_y = (0.5-0.16)/0.68 + offset["y"]
        self.assertAlmostEqual(mapped_y, 0.08)
        self.assertAlmostEqual(offset["x"], 0.0)

    def test_contact_ratio_and_proximity_thresholds_are_palm_relative(self):
        left = fake_hand(cx=0.4)
        right = fake_hand(cx=0.6)
        ratio = fingertip_contact_ratio(left, right)
        self.assertIsNotNone(ratio)
        small = proximity_thresholds({"left": 50.0, "right": 60.0}, 0.25, 2.5)
        large = proximity_thresholds({"left": 200.0, "right": 240.0}, 0.25, 2.5)
        self.assertAlmostEqual(large[0]/small[0], 4.0, delta=0.5)
        self.assertAlmostEqual(large[1]/small[1], 4.0, delta=0.5)

    def test_calibration_persists_atomically_and_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hand_calibration.json"
            data = default_calibration()
            data.update({"completed": True, "calibrated": True,
                         "screen_anchor_camera": {"x": 0.51, "y": 0.19},
                         "screen_offset": screen_touch_offset(0.51, 0.19),
                         "reference_palm_size": {"left": 0.08, "right": 0.09},
                         "proximity_touch_ratio": 0.32,
                         "proximity_interaction_ratio": 2.7})
            self.assertTrue(save_hand_calibration(path, data))
            loaded = load_hand_calibration(path)
            self.assertTrue(loaded["completed"])
            self.assertTrue(loaded["calibrated"])
            self.assertAlmostEqual(loaded["reference_palm_size"]["right"], 0.09)
            self.assertAlmostEqual(loaded["proximity_touch_ratio"], 0.32)

    def test_wizard_runs_once_and_is_available_from_hands_settings(self):
        for token in (
            "class HandCalibrationWizard(QWidget):",
            "QTimer.singleShot(1200, self._maybe_start_hand_onboarding)",
            "def _maybe_start_hand_onboarding(self):",
            "def _start_hand_onboarding(self, first_run=False):",
            "HAND_CALIBRATION_FILE = ROOT / \"hand_calibration.json\"",
            "RUN HAND SCALE & PROXIMITY CALIBRATION",
            "def _collect_hand_calibration_frame(self, hands_by_side):",
        ):
            self.assertIn(token, self.main)

    def test_calibration_consumes_normal_gestures_and_draws_live_hand_above_target(self):
        self.assertIn("if getattr(self, \"hand_calibration_active\", False):", self.main)
        self.assertIn("self._collect_hand_calibration_frame(hands_by_side)", self.main)
        self.assertIn("def set_hand_overlay(self, hands, tli_points):", self.main)
        self.assertIn("def _update_tli_calibration_live_overlay", self.main)
        self.assertIn("exact scaled/calibrated TLI point against the outlined target area", self.main)

    def test_proximity_and_mesh_scale_use_calibrated_palm_references(self):
        self.assertIn("proximity_thresholds(", self.main)
        self.assertIn("palm_sizes_screen, touch_ratio, interaction_ratio", self.main)
        self.assertIn("relative_palm_projection_scale(math.sqrt(max(1.0, palm_length*palm_width)), reference_size)", self.main)
        self.assertIn("reference_palm_size", self.main)

    def test_scroll_uses_native_wheel_input_at_live_cursor(self):
        self.assertIn("def _scroll_target_window(self, active, steps, cursor=None):", self.main)
        self.assertIn("pyautogui.moveTo(screen_point.x(), screen_point.y(), duration=0, _pause=False)", self.main)
        self.assertIn("pyautogui.scroll(wheel_steps, _pause=False)", self.main)
        self.assertNotIn("send_message(wintypes.HWND(int(hwnd)), 0x020A", self.main)


if __name__ == "__main__":
    unittest.main()
