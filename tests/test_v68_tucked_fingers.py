import json
import unittest
from pathlib import Path

import numpy as np

import hand_occlusion as ho

ROOT = Path(__file__).resolve().parents[1]
HANDS = json.loads((Path(__file__).with_name("tuck_hand_landmarks.json")).read_text(encoding="utf-8"))["hands"]


# The fixtures are a physical left hand: MediaPipe's Tasks landmarker labels these
# renders "Right", and the app's side is the opposite of that label.
SIDE = "left"


def hidden(name, side=SIDE):
    return {k for k, v in ho.hidden_digits(HANDS[name], side).items() if v}


class HiddenDigitTests(unittest.TestCase):
    def test_fist_seen_from_the_back_hides_every_tip(self):
        self.assertEqual(hidden("fist_back_view"), {"thumb", "index", "middle", "ring", "pinky"})

    def test_pointing_from_the_back_hides_all_but_the_index(self):
        self.assertEqual(hidden("pointing_back_view"), {"thumb", "middle", "ring", "pinky"})

    def test_visible_tips_are_left_alone(self):
        for name in ("open_front", "open_back", "open_side", "pinch", "pinch_back",
                     "fist_front_view", "fist_side_view", "pointing_front_view"):
            self.assertEqual(hidden(name), set(), name)

    def test_folded_fingers_behind_a_palm_turned_away_count_as_hidden(self):
        # Even when the tracker's guessed tips stray outside the palm outline.
        points = [list(p) for p in HANDS["fist_back_view"]]
        palm = ho._to_space(points)
        side = ho.palm_side_normal(palm, SIDE)
        self.assertGreater(side[2], ho.PALM_AWAY_Z)
        for tip in (8, 12, 16, 20):
            points[tip][0] += 0.03
        self.assertEqual(hidden_points(points) & set(ho.FINGERS), set(ho.FINGERS))

    def test_one_hidden_finger_tucks_the_other_folded_fingers(self):
        points = HANDS["fist_back_view"]
        self.assertEqual(hidden("fist_back_view") & set(ho.FINGERS), set(ho.FINGERS))
        self.assertTrue(all(ho.hidden_digits(points, SIDE)[f] for f in ho.FINGERS))

    def test_palm_side_follows_handedness(self):
        # An open hand has no folded fingers, so only handedness decides.
        palm = ho._to_space(HANDS["open_back"])
        self.assertTrue(np.allclose(ho.palm_side_normal(palm, "left"), -ho.palm_side_normal(palm, "right")))
        mirrored = [[1.0-p[0], p[1], p[2]] for p in HANDS["open_back"]]
        self.assertAlmostEqual(float(ho.palm_side_normal(ho._to_space(mirrored), "right")[2]),
                               float(ho.palm_side_normal(palm, "left")[2]), places=6)

    def test_fingers_fold_towards_the_palm_for_either_label(self):
        # Every finger of the tracked fist folds to the palm side, and so does the drawn
        # tuck, even if the hand's left/right label is wrong for a moment.
        points = HANDS["fist_back_view"]
        palm = ho._to_space(points)
        for side in ("left", "right"):
            normal = ho.palm_side_normal(palm, side)
            for f in ho.FINGERS:
                mcp, tip = ho.DIGITS[f][0], ho.DIGITS[f][-1]
                self.assertGreater(float((palm[tip]-palm[mcp]) @ normal), 0.0, (side, f))
            tucked = ho._to_space(ho.tucked_landmarks(points, set(ho.FINGERS), side))
            for f in ho.FINGERS:
                mcp, tip = ho.DIGITS[f][0], ho.DIGITS[f][-1]
                self.assertGreater(float((tucked[tip]-tucked[mcp]) @ normal), 0.0, (side, f))

    def test_open_palm_towards_the_camera_is_not_palm_away(self):
        for name in ("open_back", "pinch_back", "fist_front_view", "pointing_front_view"):
            normal = ho.palm_side_normal(ho._to_space(HANDS[name]), SIDE)
            self.assertLess(float(normal[2]), ho.PALM_AWAY_Z, name)

    def test_thumb_after_a_pinch_is_not_stuck_to_the_palm(self):
        # Palm turned away, thumb just released from a pinch and out to the side: its tip
        # is within a palm length of the middle knuckle but not across the palm.
        pinch = np.array(HANDS["pinch"])
        self.assertLess(ho.thumb_across(ho._to_space(pinch).tolist()), ho.STAY_FOLDED_THUMB_ACROSS)
        away = pinch[4]-pinch[8]
        away[2] = 0.0
        released = pinch.copy()
        released[4] = pinch[4] + away/np.linalg.norm(away)*0.06
        smoother = ho.TuckSmoother()
        for k in range(20):
            frame = [tuple(r + [1e-7*k, 0, 0]) for r in (pinch if k < 8 else released)]
            smoother.update(SIDE, frame, k/30)
        self.assertEqual(smoother.weights[SIDE]["thumb"], 0.0)
        self.assertNotIn("thumb", hidden_points(released.tolist()))

    def test_tips_outside_the_camera_frame_are_not_tucked(self):
        points = [list(p) for p in HANDS["fist_back_view"]]
        points[8][1] = -0.05
        self.assertNotIn("index", {k for k, v in ho.hidden_digits(points).items() if v})


def hidden_points(points, side=SIDE):
    return {k for k, v in ho.hidden_digits(points, side).items() if v}


class TuckedPoseTests(unittest.TestCase):
    def test_tucked_fingers_fold_into_the_palm_and_the_palm_stays_put(self):
        points = HANDS["pointing_back_view"]
        digits = hidden("pointing_back_view")
        tucked = ho.tucked_landmarks(points, digits, SIDE)
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
        first = smoother.update(SIDE, fist, 0.0)                    # a new hand snaps
        self.assertAlmostEqual(smoother.weights[SIDE]["index"], 1.0)
        target = ho.tucked_landmarks(fist, set(ho.DIGITS), SIDE)
        self.assertTrue(np.allclose(np.array(first), np.array(target)))
        # Opening the hand: the fingers stay tucked for a few camera frames (hysteresis),
        # then ease out rather than popping.
        frame = [list(p) for p in open_hand]
        for k in range(1, ho.RELEASE_FRAMES):
            frame[0][0] += 1e-6                                     # a new camera frame
            smoother.update(SIDE, [tuple(p) for p in frame], k/30)
            self.assertEqual(smoother.weights[SIDE]["index"], 1.0)
        frame[0][0] += 1e-6
        smoother.update(SIDE, [tuple(p) for p in frame], ho.RELEASE_FRAMES/30)
        self.assertTrue(0.0 < smoother.weights[SIDE]["index"] < 1.0)
        for k in range(ho.RELEASE_FRAMES+1, 40):
            frame[0][0] += 1e-6
            out = smoother.update(SIDE, [tuple(p) for p in frame], k/30)
        self.assertEqual(smoother.weights[SIDE]["index"], 0.0)
        self.assertEqual(out[8], tuple(frame[8]))

    def test_wandering_guesses_for_a_hidden_fist_do_not_flap(self):
        fist, open_hand = np.array(HANDS["fist_back_view"]), np.array(HANDS["open_back"])
        rng = np.random.default_rng(3)
        smoother, previous, moves = ho.TuckSmoother(), None, []
        palm = np.linalg.norm((fist[9]-fist[0])*[1, 9/16, 1])
        for k in range(90):
            frame = fist.copy()
            for mcp in (5, 9, 13, 17):          # the tracker half-opens each hidden finger at random
                t = float(np.clip(0.35 + rng.normal(0, 0.2), 0.0, 0.7))
                for j in (mcp+1, mcp+2, mcp+3):
                    frame[j] = fist[j]*(1-t) + (fist[mcp] + open_hand[j]-open_hand[mcp])*t
            out = np.array(smoother.update(SIDE, [tuple(r) for r in frame], k/30))[:, :2]*[1, 9/16]
            if previous is not None:
                moves.append(np.mean([np.linalg.norm(out[t]-previous[t]) for t in (8, 12, 16, 20)])/palm)
            previous = out
        self.assertLess(float(np.mean(moves)), 0.03)    # raw guesses move ~0.24 palm per frame

    def test_repeated_camera_frame_reuses_the_settled_result(self):
        smoother = ho.TuckSmoother()
        fist = HANDS["fist_back_view"]
        first = smoother.update(SIDE, fist, 0.0)
        self.assertIs(smoother.update(SIDE, [list(p) for p in fist], 1/60), first)

    def test_drawn_hand_uses_tucked_points_but_tracking_does_not(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("self._hand_tucker.update(side, self.hand_positions[side], now)", source)
        self.assertIn('if style != "skeleton" and display is not None and display[0] is points:', source)


if __name__ == "__main__":
    unittest.main()
