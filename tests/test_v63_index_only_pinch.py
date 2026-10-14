import types
import unittest

import custom_gestures
from gesture_rules import thumb_pinches_index


def _p(x, y, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z)


FINGERS = {"index": (5, 0.455), "middle": (9, 0.50), "ring": (13, 0.545), "pinky": (17, 0.59)}


def _hand(pinch_finger):
    lm = [None]*21
    lm[0] = _p(0.52, 0.82)
    for name, (mcp, x) in FINGERS.items():
        lm[mcp] = _p(x, 0.62)
        if name == pinch_finger:
            lm[mcp+1], lm[mcp+2], lm[mcp+3] = _p(x-0.01, 0.53), _p(x-0.04, 0.49), _p(x-0.07, 0.50)
        elif name == "index" and pinch_finger in ("middle", "ring"):
            lm[mcp+1], lm[mcp+2], lm[mcp+3] = _p(x-0.005, 0.53), _p(x-0.02, 0.47), _p(x-0.035, 0.44)
        else:
            lm[mcp+1], lm[mcp+2], lm[mcp+3] = _p(x, 0.52), _p(x, 0.46), _p(x, 0.40)
    lm[1], lm[2], lm[3] = _p(0.45, 0.76), _p(0.40, 0.70), _p(0.37, 0.63)
    tip = lm[FINGERS[pinch_finger][0]+3]
    lm[4] = _p(tip.x-0.01, tip.y+0.01)
    return lm


class IndexOnlyPinchTests(unittest.TestCase):
    def test_rule_requires_index_to_be_the_nearest_fingertip(self):
        self.assertTrue(thumb_pinches_index([0.02, 0.10, 0.20, 0.30]))
        self.assertFalse(thumb_pinches_index([0.08, 0.02, 0.20, 0.30]))  # thumb on middle
        self.assertFalse(thumb_pinches_index([0.10, 0.12, 0.03, 0.30]))  # thumb on ring
        self.assertFalse(thumb_pinches_index([0.15, 0.12, 0.10, 0.02]))  # thumb on pinky
        self.assertFalse(thumb_pinches_index([float("nan"), 0.1, 0.2, 0.3]))

    def test_other_finger_pinches_stay_free_for_custom_gestures(self):
        self.assertLess(custom_gestures.pinch_ratio(_hand("index")), 0.38)
        for finger in ("middle", "ring", "pinky"):
            ratio = custom_gestures.pinch_ratio(_hand(finger))
            self.assertEqual(ratio, custom_gestures.NOT_AN_INDEX_PINCH)
            features = custom_gestures.hand_features(_hand(finger))
            conflict = custom_gestures.builtin_conflict(features, ratio) or ""
            self.assertNotIn("pinch", conflict)


if __name__ == "__main__":
    unittest.main()
