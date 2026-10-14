import json
import tempfile
import unittest
from pathlib import Path

import custom_gestures as cg

HANDS = json.loads((Path(__file__).with_name("split_hand_landmarks.json")).read_text(encoding="utf-8"))["hands"]
OPEN_HANDS = ("open_relaxed", "open_relaxed_turned", "open_together", "open_wide")
VULCANS = ("vulcan_cartoon", "vulcan_cartoon_mirrored", "vulcan", "vulcan_turned", "vulcan_soft")


def features(name):
    return cg.hand_features(HANDS[name])


def template(name, label="X"):
    return {"kind": "pose", "side": "either", "features": features(name), "label": label}


class SplitPoseTests(unittest.TestCase):
    def test_vulcan_salute_is_recognised_and_recordable(self):
        for name in VULCANS:
            f = features(name)
            self.assertTrue(cg.is_split(f), name)
            self.assertEqual(cg.describe(f), "VULCAN", name)
            self.assertIsNone(cg.builtin_conflict(f), name)

    def test_ordinary_open_hands_are_not_splits_and_stay_reserved(self):
        for name in OPEN_HANDS:
            f = features(name)
            self.assertFalse(cg.is_split(f), name)
            self.assertEqual(cg.describe(f), "OPEN HAND", name)
            self.assertIn("open hand", cg.builtin_conflict(f), name)

    def test_other_splits_get_a_bar_label(self):
        f = features("split_index")
        self.assertTrue(cg.is_split(f))
        self.assertEqual(cg.describe(f), "T·I|M·R·P")
        self.assertIsNone(cg.builtin_conflict(f))

    def test_vulcan_gesture_fires_for_vulcans_only(self):
        gestures = [template("vulcan_cartoon", "VULCAN")]
        for name in VULCANS:
            if name == "vulcan_turned":
                continue  # rotated past the default tolerance for every pose type
            self.assertEqual(cg.best_pose_match(features(name), gestures, "right", 0.18), 0, name)
        for name in OPEN_HANDS + ("split_index",):
            self.assertIsNone(cg.best_pose_match(features(name), gestures, "right", 0.40), name)

    def test_open_hands_would_match_without_the_spread_gate(self):
        # The bend/thumb features alone can't tell these apart: the gate is what does it.
        vulcan = features("vulcan")
        for name in ("open_relaxed", "open_together", "open_wide"):
            self.assertLess(cg.feature_distance(features(name), vulcan), 0.18, name)
            self.assertFalse(cg.split_gate(features(name), vulcan), name)

    def test_vulcan_does_not_set_off_an_ordinary_open_finger_gesture(self):
        four = list(features("open_relaxed"))
        self.assertTrue(cg.split_gate(features("open_wide"), four))   # spread doesn't matter normally
        self.assertFalse(cg.split_gate(features("vulcan"), four))     # but a split is its own pose

    def test_poses_recorded_before_spread_still_load_and_match_as_before(self):
        old = features("open_relaxed")[:cg.PRE_SPREAD_FEATURE_COUNT]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_gestures.json"
            path.write_text(json.dumps({"gestures": [{"kind": "pose", "features": old, "action": "copy"}]}),
                            encoding="utf-8")
            loaded = cg.load_custom_gestures(path)
        self.assertEqual(len(loaded[0]["features"]), cg.FEATURE_COUNT)
        self.assertEqual(loaded[0]["features"][12:], [cg.SPREAD_UNKNOWN]*3)
        self.assertFalse(cg.is_split(loaded[0]["features"]))
        self.assertEqual(cg.best_pose_match(features("open_wide"), loaded, "left", 0.18), 0)

    def test_spread_is_not_part_of_the_distance(self):
        a = features("open_relaxed")
        b = list(a)
        b[12:] = [1.5, 1.5, 1.5]
        self.assertAlmostEqual(cg.feature_distance(a, b), 0.0)


if __name__ == "__main__":
    unittest.main()
