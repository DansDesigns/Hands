import json
import unittest
from pathlib import Path

import numpy as np

import hand_occlusion as ho

ROOT = Path(__file__).resolve().parents[1]
HANDS = json.loads((Path(__file__).with_name("tuck_hand_landmarks.json")).read_text(encoding="utf-8"))["hands"]


def hidden(name):
    return {k for k, v in ho.hidden_digits(HANDS[name]).items() if v}


class HiddenDigitTests(unittest.TestCase):
    def test_fist_seen_from_the_back_hides_every_tip(self):
        self.assertEqual(hidden("fist_back_view"), {"thumb", "index", "middle", "ring", "pinky"})

    def test_pointing_from_the_back_hides_all_but_the_index(self):
        self.assertEqual(hidden("pointing_back_view"), {"thumb", "middle", "ring", "pinky"})

    def test_visible_tips_are_left_alone(self):
        for name in ("open_front", "open_back", "open_side", "pinch", "pinch_back",
                     "fist_front_view", "fist_side_view", "pointing_front_view"):
            self.assertEqual(hidden(name), set(), name)

    def test_tips_outside_the_camera_frame_are_not_tucked(self):
        points = [list(p) for p in HANDS["fist_back_view"]]
        points[8][1] = -0.05
        self.assertNotIn("index", {k for k, v in ho.hidden_digits(points).items() if v})


class TuckedPoseTests(unittest.TestCase):
    def test_tucked_fingers_fold_into_the_palm_and_the_palm_stays_put(self):
        points = HANDS["pointing_back_view"]
        digits = hidden("pointing_back_view")
        tucked = ho.tucked_landmarks(points, digits)
        p, t = ho._to_space(points), ho._to_space(tucked)
        palm_len = np.linalg.norm(p[9]-p[0])
        for i in (0, 1, 2, 5, 6, 7, 8, 9, 13, 17):          # palm, thumb base, index untouched
            self.assertTrue(np.allclose(p[i], t[i]), i)
        for tip in (12, 16, 20):
            self.assertLess(np.linalg.norm(t[tip]-t[0]), 1.0*palm_len)
        # The thumb lies over the curled middle/ring fingers, not up along the index.
        self.assertLess(np.linalg.norm(t[4]-(t[10]+t[11]+t[14]+t[15])/4), 0.3*palm_len)

    def test_untouched_when_nothing_is_hidden(self):
        points = HANDS["open_front"]
        self.assertEqual(ho.tucked_landmarks(points, set()), list(points))


class SmootherTests(unittest.TestCase):
    def test_tucking_fades_in_and_out(self):
        smoother = ho.TuckSmoother()
        fist, open_hand = HANDS["fist_back_view"], HANDS["open_front"]
        first = smoother.update("left", fist, 0.0)                  # a new hand snaps
        self.assertAlmostEqual(smoother.weights["left"]["index"], 1.0)
        target = ho.tucked_landmarks(fist, set(ho.DIGITS))
        self.assertTrue(np.allclose(np.array(first), np.array(target)))
        smoother.update("left", open_hand, 1/60)
        self.assertTrue(0.0 < smoother.weights["left"]["index"] < 1.0)  # easing, not popping
        for k in range(2, 40):
            out = smoother.update("left", open_hand, k/60)
        self.assertEqual(out, open_hand)

    def test_repeated_camera_frame_reuses_the_settled_result(self):
        smoother = ho.TuckSmoother()
        fist = HANDS["fist_back_view"]
        first = smoother.update("right", fist, 0.0)
        self.assertIs(smoother.update("right", [list(p) for p in fist], 1/60), first)

    def test_drawn_hand_uses_tucked_points_but_tracking_does_not(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("self._hand_tucker.update(side, self.hand_positions[side], now)", source)
        self.assertIn('if style != "skeleton" and display is not None and display[0] is points:', source)


if __name__ == "__main__":
    unittest.main()
