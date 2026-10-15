"""Regression checks for independent nine-point TPI screen calibration."""
from pathlib import Path
import math
import tempfile
import unittest

from tpi_screen_calibration import (
    TARGET_COUNT, TARGET_FRACTIONS, TARGET_NAMES, target_pixel_positions,
    calibration_from_samples, validate_calibration, map_point,
    load_calibrations, save_calibrations,
)

ROOT = Path(__file__).resolve().parent.parent


class TpiNinePointCalibrationTests(unittest.TestCase):
    def test_nine_target_layout_has_corners_edges_and_centre(self):
        self.assertEqual(TARGET_COUNT, 9)
        self.assertEqual(len(TARGET_NAMES), 9)
        positions = target_pixel_positions(1920, 1080)
        self.assertEqual(len(positions), 9)
        self.assertEqual(positions[4], (0.5 * 1919, 0.5 * 1079))
        # Top edge points share y; left/right edge points share their x by row.
        self.assertEqual(positions[0][1], positions[1][1])
        self.assertEqual(positions[1][1], positions[2][1])
        self.assertEqual(positions[3][1], positions[4][1])
        self.assertEqual(positions[4][1], positions[5][1])
        self.assertEqual(positions[3][0], positions[6][0])
        self.assertEqual(positions[1][0], positions[4][0])
        self.assertEqual(TARGET_FRACTIONS[0], (0.08, 0.08))
        self.assertEqual(TARGET_FRACTIONS[-1], (0.92, 0.92))

    def test_mesh_maps_each_observed_target_back_to_its_screen_target(self):
        width, height = 1920, 1080
        observed = []
        for x, y in TARGET_FRACTIONS:
            # Synthetic mild lens/hand-mapping drift in the raw cursor readings.
            observed.append({"x": (x + 0.018 * math.sin(math.pi*y)) * (width-1),
                             "y": (y + 0.012 * math.sin(math.pi*x)) * (height-1)})
        calibration = calibration_from_samples(observed, width, height)
        self.assertTrue(validate_calibration(calibration))
        for got, expected in zip(observed, TARGET_FRACTIONS):
            mapped = map_point((got["x"], got["y"]), calibration, width, height)
            self.assertIsNotNone(mapped)
            self.assertAlmostEqual(mapped[0]/(width-1), expected[0], places=6)
            self.assertAlmostEqual(mapped[1]/(height-1), expected[1], places=6)

    def test_mesh_is_resolution_independent(self):
        source = [{"x": x*1919, "y": y*1079} for x, y in TARGET_FRACTIONS]
        calibration = calibration_from_samples(source, 1920, 1080)
        point_1080 = map_point((1919*0.5, 1079*0.5), calibration, 1920, 1080)
        point_720 = map_point((1279*0.5, 719*0.5), calibration, 1280, 720)
        self.assertAlmostEqual(point_1080[0]/1919, point_720[0]/1279, places=6)
        self.assertAlmostEqual(point_1080[1]/1079, point_720[1]/719, places=6)

    def test_collapsed_or_incomplete_mesh_is_rejected(self):
        self.assertIsNone(calibration_from_samples([(100, 100)]*8, 1920, 1080))
        collapsed = {"version": 1, "observed": [{"x": 0.5, "y": 0.5} for _ in range(9)]}
        self.assertFalse(validate_calibration(collapsed))
        self.assertIsNone(map_point((300, 200), collapsed, 1920, 1080))

    def test_folded_mesh_is_rejected_instead_of_flipping_the_pointer(self):
        points = [{"x": x, "y": y} for x, y in TARGET_FRACTIONS]
        points[1], points[2] = points[2], points[1]
        folded = {"version": 1, "observed": points}
        self.assertFalse(validate_calibration(folded))
        self.assertIsNone(map_point((600, 300), folded, 1920, 1080))

    def test_left_and_right_calibrations_save_and_load_independently(self):
        left = calibration_from_samples(
            [{"x": x*1919, "y": y*1079} for x, y in TARGET_FRACTIONS], 1920, 1080)
        right = calibration_from_samples(
            [{"x": min(1919, (x+0.01)*1919), "y": min(1079, (y+0.01)*1079)}
             for x, y in TARGET_FRACTIONS], 1920, 1080)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tpi_screen_calibration.json"
            self.assertTrue(save_calibrations(path, {"left": left, "right": right}))
            loaded = load_calibrations(path)
            self.assertEqual(loaded["left"], left)
            self.assertEqual(loaded["right"], right)

    def test_application_runs_fullscreen_nine_step_wizard_and_applies_mesh_per_hand(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('TPI_SCREEN_CALIBRATION_FILE = ROOT / "tpi_screen_calibration.json"', main)
        self.assertIn('self.setGeometry(geometry)', main)
        self.assertIn('Point {target_index+1}/9', main)
        self.assertIn('make_tpi_screen_calibration(saved_points, width, height)', main)
        self.assertIn('map_tpi_screen_point(clamped, calibration, width, height)', main)
        self.assertIn('"CALIBRATE {calibration_side.upper()} TPI · 9 POINTS"', main)
        self.assertIn('"tli": "TPI"', main)


if __name__ == "__main__":
    unittest.main()
