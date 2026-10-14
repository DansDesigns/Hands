import unittest
from pathlib import Path

from gesture_rules import (
    PINCH_DRAG_MIN_HOLD_SECONDS, PINCH_RELEASE_SETTLE_SECONDS, pinch_release_click_verdict,
)

ROOT = Path(__file__).resolve().parents[1]


class PinchPassThroughTests(unittest.TestCase):
    def test_release_into_closed_fingers_is_not_a_click(self):
        # Open hand -> thumbs up sweeps the index past the thumb: a brief pinch that
        # releases into folded fingers.
        self.assertEqual(pinch_release_click_verdict(0.08, False, True), "cancel")
        self.assertEqual(pinch_release_click_verdict(0.20, False, True), "cancel")

    def test_release_into_an_open_hand_clicks_at_once(self):
        self.assertEqual(pinch_release_click_verdict(0.08, True, False), "click")

    def test_unclear_release_waits_briefly_then_clicks(self):
        self.assertEqual(pinch_release_click_verdict(0.10, False, False), "wait")
        self.assertEqual(pinch_release_click_verdict(PINCH_RELEASE_SETTLE_SECONDS, False, False), "click")
        self.assertLessEqual(PINCH_RELEASE_SETTLE_SECONDS, 0.35)

    def test_brief_moving_pinch_cannot_start_a_drag(self):
        self.assertGreaterEqual(PINCH_DRAG_MIN_HOLD_SECONDS, 0.12)
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("now - pulse_start >= PINCH_DRAG_MIN_HOLD_SECONDS and", source)
        self.assertIn('release_for, is_open, pose.get("fingers_closed", False))', source)
        self.assertIn('"fingers_closed": fingers_closed_pose(landmarks, edge_tolerant),', source)


if __name__ == "__main__":
    unittest.main()
