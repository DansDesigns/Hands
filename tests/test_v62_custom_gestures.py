import math
import tempfile
import unittest
from pathlib import Path

from custom_gestures import (
    FINGERS, HoldTrigger, best_pose_match, builtin_conflict, describe, feature_distance,
    hand_features, load_custom_gestures, mean_features, save_custom_gestures, tli_overlap,
)

KN = {"index": (-30, 90), "middle": (-10, 95), "ring": (10, 90), "pinky": (28, 80)}
SEG = {"index": (40, 25, 20), "middle": (45, 28, 20), "ring": (42, 26, 19), "pinky": (32, 20, 17)}


def make_hand(extended, thumb_out, yaw=0.0):
    pts = [None]*21
    pts[0] = (0, 0, 0)
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        kx, ky = KN[name]
        pts[mcp] = (kx, ky, 0)
        curl = 0.12 if name in extended else 1.45
        angle, p = 0.0, (kx, ky, 0)
        for idx, length in zip((pip, dip, tip), SEG[name]):
            angle += curl
            p = (p[0], p[1]+length*math.cos(angle), p[2]-length*math.sin(angle))
            pts[idx] = p
    pts[1] = (-30, 25, -5)
    if thumb_out:
        pts[2], pts[3], pts[4] = (-55, 45, -8), (-72, 62, -10), (-85, 78, -12)
    else:
        pts[2], pts[3], pts[4] = (-35, 50, -20), (-15, 60, -35), (5, 62, -40)
    c, s = math.cos(yaw), math.sin(yaw)
    return [((300 + x*c + z*s)/640, (200 - y)/360, (-x*s + z*c)/640) for x, y, z in pts]


class CustomGestureTests(unittest.TestCase):
    def test_example_poses_are_labelled(self):
        self.assertEqual(describe(hand_features(make_hand({"pinky"}, True))), "T·P")
        self.assertEqual(describe(hand_features(make_hand({"index", "middle", "ring"}, False))), "I·M·R")
        self.assertEqual(describe(hand_features(make_hand({"index", "middle", "ring", "pinky"}, False))), "I·M·R·P")

    def test_builtin_poses_are_refused(self):
        self.assertIsNotNone(builtin_conflict(hand_features(make_hand(set(), False))))
        self.assertIsNotNone(builtin_conflict(hand_features(make_hand({"index", "middle", "ring", "pinky"}, True))))
        self.assertIsNotNone(builtin_conflict(hand_features(make_hand({"index", "middle"}, False))))
        self.assertIsNone(builtin_conflict(hand_features(make_hand({"pinky"}, True))))

    def test_rotated_pose_matches_its_template_and_respects_side(self):
        shaka = mean_features([hand_features(make_hand({"pinky"}, True))])
        three = mean_features([hand_features(make_hand({"index", "middle", "ring"}, False))])
        gestures = [{"kind": "pose", "side": "left", "features": shaka},
                    {"kind": "pose", "side": "either", "features": three}]
        live = hand_features(make_hand({"pinky"}, True, yaw=math.radians(25)))
        self.assertEqual(best_pose_match(live, gestures, "left", 0.18), 0)
        self.assertIsNone(best_pose_match(live, gestures, "right", 0.18))
        live3 = hand_features(make_hand({"index", "middle", "ring"}, False, yaw=math.radians(-20)))
        self.assertEqual(best_pose_match(live3, gestures, "right", 0.18), 1)

    def test_thumb_alone_separates_four_fingers_from_open_hand(self):
        four = hand_features(make_hand({"index", "middle", "ring", "pinky"}, False))
        open_hand = hand_features(make_hand({"index", "middle", "ring", "pinky"}, True))
        self.assertGreater(feature_distance(four, open_hand), 0.25)

    def test_hold_trigger_fires_once_after_hold_and_rearms_on_release(self):
        t = HoldTrigger()
        self.assertIsNone(t.update(2, 0.0, 0.5))
        self.assertIsNone(t.update(2, 0.3, 0.5))
        self.assertEqual(t.update(2, 0.6, 0.5), 2)
        self.assertIsNone(t.update(2, 2.0, 0.5))
        self.assertIsNone(t.update(None, 2.1, 0.5))
        self.assertIsNone(t.update(2, 2.2, 0.5))
        self.assertEqual(t.update(2, 2.8, 0.5), 2)

    def test_tli_overlap(self):
        self.assertTrue(tli_overlap((100, 100), (120, 100), 30))
        self.assertFalse(tli_overlap((100, 100), (140, 100), 30))
        self.assertFalse(tli_overlap(None, (1, 1), 30))


    def test_custom_key_combo_is_preserved_by_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_gestures.json"
            features = hand_features(make_hand({"pinky"}, True))
            save_custom_gestures(path, [{
                "kind": "pose", "side": "either", "features": features,
                "label": "T·P", "action": "custom_key", "key_combo": "CTRL+ALT+SHIFT+K",
            }])
            loaded = load_custom_gestures(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["action"], "custom_key")
            self.assertEqual(loaded[0]["key_combo"], "CTRL+ALT+SHIFT+K")

    def test_save_and_load_round_trip_drops_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_gestures.json"
            features = hand_features(make_hand({"pinky"}, True))
            save_custom_gestures(path, [
                {"kind": "pose", "side": "left", "features": features, "label": "T·P", "action": "media_next"},
                {"kind": "tli_overlap", "action": "toggle_glass_menu"},  # now built in: dropped
                {"kind": "pose", "features": [1, 2]},
                {"kind": "pose", "side": "bogus", "features": features, "action": "rm -rf"},
                {"kind": "pose", "features": features[:10], "action": "copy"},  # legacy 10-value
            ])
            loaded = load_custom_gestures(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["action"], "media_next")
            self.assertEqual(loaded[1]["side"], "either")
            self.assertEqual(loaded[1]["action"], "none")
            self.assertEqual(len(loaded[2]["features"]), len(features))


def make_thumb_pose(direction):
    """Closed fingers with the thumb out, rotated so the thumb points up or down on screen."""
    pts = make_hand(set(), True)
    # Rotate about the wrist in the image plane so the thumb points straight up/down.
    import math as _m
    wx, wy = pts[0][0]*640, pts[0][1]*360
    tx, ty = pts[4][0]*640 - pts[2][0]*640, pts[4][1]*360 - pts[2][1]*360
    target = -_m.pi/2 if direction == "up" else _m.pi/2
    a = target - _m.atan2(ty, tx)
    c, s = _m.cos(a), _m.sin(a)
    out = []
    for x, y, z in pts:
        px, py = x*640 - wx, y*360 - wy
        out.append(((wx + px*c - py*s)/640, (wy + px*s + py*c)/360, z))
    return out


class ThumbAndFistTests(unittest.TestCase):
    def test_thumb_out_is_detected_and_tucked_thumb_is_not(self):
        from gesture_rules import thumb_sticking_out, thumb_across_palm
        tucked = make_hand(set(), False)
        # Put the tucked thumb tip over the index/middle fingers, as in a real fist.
        tucked[4] = ((300 - 15)/640, (200 - 85)/360, -45/640)
        self.assertFalse(thumb_sticking_out(tucked))
        self.assertGreater(thumb_across_palm(tucked), 0.0)
        self.assertTrue(thumb_sticking_out(make_hand(set(), True)))
        self.assertTrue(thumb_sticking_out(make_thumb_pose("up")))
        self.assertTrue(thumb_sticking_out(make_thumb_pose("down")))

    def test_fist_detector_requires_tucked_thumb(self):
        main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn("return fingers_closed and not thumb_sticking_out(landmarks, threshold=-0.55)", main)

    def test_thumbs_up_and_down_are_labelled_and_distinguished(self):
        up = hand_features(make_thumb_pose("up"))
        down = hand_features(make_thumb_pose("down"))
        self.assertEqual(describe(up), "THUMBS UP")
        self.assertEqual(describe(down), "THUMBS DOWN")
        self.assertIsNone(builtin_conflict(up))
        gestures = [{"kind": "pose", "side": "either", "features": up},
                    {"kind": "pose", "side": "either", "features": down}]
        self.assertEqual(best_pose_match(hand_features(make_thumb_pose("up")), gestures, "left", 0.18), 0)
        self.assertEqual(best_pose_match(hand_features(make_thumb_pose("down")), gestures, "left", 0.18), 1)

    def test_direction_is_ignored_when_thumb_is_tucked(self):
        a = hand_features(make_hand({"index", "middle", "ring"}, False))
        b = list(a)
        b[11] = -b[11] if b[11] else 1.0
        self.assertAlmostEqual(feature_distance(a, b), 0.0)


class BuiltInOverlapTests(unittest.TestCase):
    def test_overlap_is_built_in_glass_menu_toggle(self):
        main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn('fired = trigger.update("glass_menu" if overlapping else None, now, hold)', main)
        self.assertIn("self._toggle_glass_menu_from_voice()", main)
        self.assertNotIn("+ ADD: BOTH TLIs OVERLAP", main)


class CustomGestureDropdownTests(unittest.TestCase):
    def test_every_action_appears_exactly_once_in_the_dropdown(self):
        from custom_gestures import ACTION_GROUPS, ACTION_IDS
        listed = [a for column in ACTION_GROUPS for _title, ids in column for a in ids]
        self.assertEqual(sorted(listed), sorted(ACTION_IDS))
        self.assertEqual(len(listed), len(set(listed)))

    def test_open_dropdown_takes_presses_before_sliders(self):
        main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn("self._click_custom_gesture_dropdown(point, cx, cy, scale)", main)
        self.assertIn("return None  # the open action list covers the sliders", main)


class CustomGestureTimerToggleTests(unittest.TestCase):
    def test_timer_defaults_on_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_gestures.json"
            features = hand_features(make_hand({"pinky"}, True))
            save_custom_gestures(path, [
                {"kind": "pose", "side": "left", "features": features, "action": "media_next"},
                {"kind": "pose", "side": "right", "features": features, "action": "none", "timer": False},
            ])
            loaded = load_custom_gestures(path)
            self.assertTrue(loaded[0]["timer"])
            self.assertFalse(loaded[1]["timer"])

    def test_timer_off_uses_short_stability_hold(self):
        from custom_gestures import INSTANT_HOLD_SECONDS
        self.assertLess(INSTANT_HOLD_SECONDS, 0.2)
        t = HoldTrigger()
        self.assertIsNone(t.update(0, 0.0, INSTANT_HOLD_SECONDS))
        self.assertEqual(t.update(0, 0.15, INSTANT_HOLD_SECONDS), 0)

    def test_main_draws_ring_and_toggles_timer(self):
        main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn('elif row["timer"].contains(point):', main)
        self.assertIn('self.custom_gesture_rings = rings', main)
        self.assertIn('def _custom_gesture_hold(self, index):', main)


if __name__ == "__main__":
    unittest.main()
