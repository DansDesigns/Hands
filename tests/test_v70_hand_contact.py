import unittest
from pathlib import Path

import hand_contact as hc

ROOT = Path(__file__).resolve().parents[1]


def frame_hand(tip_x, direction, palm=200.0):
    """Screen-space hand whose thumb points along `direction` (+1 right, -1 left) to tip_x."""
    pts = [(tip_x - direction*palm*1.5, 500.0)]*21
    pts = list(pts)
    pts[3] = (tip_x - direction*0.35*palm, 500.0)
    pts[4] = (tip_x, 500.0)
    # Fingers up, well away from the other hand.
    for joint, tip in ((7, 8), (11, 12), (15, 16), (19, 20)):
        pts[joint] = (tip_x - direction*palm*1.2, 260.0)
        pts[tip] = (tip_x - direction*palm*1.2, 200.0)
    pts[0] = (tip_x - direction*palm*1.0, 700.0)
    pts[9] = (tip_x - direction*palm*1.0, 500.0)
    return pts


def moved(points, offsets):
    out = list(points)
    for index, (dx, dy) in offsets.items():
        out[index] = (out[index][0]+dx, out[index][1]+dy)
    return out


class HandContactTests(unittest.TestCase):
    def test_nearly_touching_thumb_tips_are_drawn_touching(self):
        left, right = frame_hand(900.0, +1), frame_hand(930.0, -1)    # 30 px apart, 0.15 palm
        offsets = hc.contact_offsets(left, right, 200.0, 200.0, "3d")
        new_left, new_right = moved(left, offsets["left"]), moved(right, offsets["right"])
        # The mesh ends on the tip landmark, so touching ends means the tips meet.
        self.assertLess(abs(new_right[4][0] - new_left[4][0]), 1.0)
        self.assertAlmostEqual(offsets["left"][4][0], -offsets["right"][4][0])   # both move halfway

    def test_hands_apart_are_left_alone(self):
        left, right = frame_hand(800.0, +1), frame_hand(900.0, -1)    # half a palm apart
        offsets = hc.contact_offsets(left, right, 200.0, 200.0, "3d")
        self.assertEqual(offsets, {"left": {}, "right": {}})

    def test_already_touching_or_overlapping_is_left_alone(self):
        left, right = frame_hand(900.0, +1), frame_hand(895.0, -1)
        self.assertEqual(hc.contact_offsets(left, right, 200.0, 200.0, "3d"), {"left": {}, "right": {}})
        # Flat capsules are capped past the tip: 30 px apart already overlaps.
        left, right = frame_hand(900.0, +1), frame_hand(930.0, -1)
        self.assertEqual(hc.contact_offsets(left, right, 200.0, 200.0, "flat"), {"left": {}, "right": {}})

    def test_pull_fades_out_smoothly(self):
        self.assertEqual(hc.snap_weight(hc.SNAP_FULL), 1.0)
        self.assertEqual(hc.snap_weight(hc.SNAP_NONE), 0.0)
        middle = hc.snap_weight((hc.SNAP_FULL+hc.SNAP_NONE)/2)
        self.assertTrue(0.4 < middle < 0.6)

    def test_drawn_hands_use_the_contact_offsets(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("self._update_hand_contact_display()", source)
        self.assertIn("offsets = hand_contact_offsets(", source)


if __name__ == "__main__":
    unittest.main()
