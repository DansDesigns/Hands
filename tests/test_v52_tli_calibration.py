"""Regression tests for persistent, hand-relative TLI calibration (v52)."""
from pathlib import Path
import math
import unittest

from tli_calibration import encode_relative_tli, resolve_relative_tli, mean_relative_tli

ROOT = Path(__file__).resolve().parent.parent


def synthetic_hand():
    points = [(0.0, 0.0, 0.0) for _ in range(21)]
    points[0] = (0.0, 0.0, 0.0)       # wrist
    points[4] = (-0.10, -0.38, 0.02)  # thumb tip
    points[5] = (-0.30, -0.30, 0.0)   # index MCP
    points[8] = (0.05, -0.62, 0.02)   # index tip
    points[9] = (0.0, -0.45, 0.0)     # middle MCP
    points[17] = (0.30, -0.30, 0.0)   # pinky MCP
    return points


def transform(points, scale=1.0, translate=(0.0, 0.0, 0.0), angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    tx, ty, tz = translate
    result = []
    for x, y, z in points:
        x, y, z = x*scale, y*scale, z*scale
        result.append((c*x-s*y+tx, s*x+c*y+ty, z+tz))
    return result


class V52TLICalibrationTests(unittest.TestCase):
    def setUp(self):
        self.hand = synthetic_hand()
        self.calibration = encode_relative_tli(self.hand)
        self.assertIsNotNone(self.calibration)

    def test_encoding_reconstructs_registered_pinch_midpoint(self):
        point = resolve_relative_tli(self.hand, self.calibration)
        expected = tuple((a+b)*0.5 for a, b in zip(self.hand[4], self.hand[8]))
        for actual, want in zip(point, expected):
            self.assertAlmostEqual(actual, want, places=7)

    def test_calibration_follows_translation_scale_and_rotation(self):
        moved = transform(self.hand, scale=1.7, translate=(0.4, -0.2, 0.15), angle=0.63)
        point = resolve_relative_tli(moved, self.calibration)
        expected_base = tuple((a+b)*0.5 for a, b in zip(self.hand[4], self.hand[8]))
        expected = transform([expected_base], scale=1.7,
                             translate=(0.4, -0.2, 0.15), angle=0.63)[0]
        for actual, want in zip(point, expected):
            self.assertAlmostEqual(actual, want, places=6)

    def test_tli_remains_palm_relative_when_fingertips_move_after_calibration(self):
        expected = resolve_relative_tli(self.hand, self.calibration)
        changed = list(self.hand)
        changed[4] = (-0.27, -0.14, 0.08)
        changed[8] = (0.22, -0.49, -0.03)
        actual = resolve_relative_tli(changed, self.calibration)
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=7)

    def test_sample_average_reduces_frame_jitter(self):
        samples = [
            {"forward": 0.40 + i*0.002, "lateral": -0.12, "depth": 0.03}
            for i in range(5)
        ]
        average = mean_relative_tli(samples)
        self.assertAlmostEqual(average["forward"], 0.404, places=7)
        self.assertAlmostEqual(average["lateral"], -0.12, places=7)

    def test_bad_landmarks_and_coordinates_are_rejected(self):
        self.assertIsNone(encode_relative_tli([]))
        self.assertIsNone(resolve_relative_tli(self.hand, {"forward": float("nan"), "lateral": 0, "depth": 0}))
        self.assertIsNone(mean_relative_tli([{"forward": 0, "lateral": 99, "depth": 0}]))

    def test_calibration_has_settings_controls_and_is_used_by_activation_anchor(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('CALIBRATE {calibration_side.upper()} TPI · 9 POINTS', main)
        self.assertIn('self._process_tli_calibration_frame(', main)
        self.assertIn('resolve_relative_tli(landmarks, calibration)', main)
        self.assertIn('save_tpi_screen_calibrations(\n                TPI_SCREEN_CALIBRATION_FILE, self.tpi_screen_calibrations)', main)
        self.assertIn('self.interactive_hand_anchor(landmarks, side)', main)
        self.assertIn('tli_calibration.json', main)
        self.assertIn('tpi_screen_calibration.json', main)


if __name__ == "__main__":
    unittest.main()
