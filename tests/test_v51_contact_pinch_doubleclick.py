"""Regression tests for contact-aware hand scaling and quick-pinch double-click."""
from pathlib import Path
import unittest

from gesture_rules import (
    align_double_click_point,
    hand_contact_gap,
    shared_contact_weight,
    shared_contact_origins,
    ease_toward,
    hand_apparent_size_scale,
    quick_pinch_click_eligible,
    quick_pinch_reentry_eligible,
    scale_hand_point,
)

ROOT = Path(__file__).resolve().parent.parent


class V51ContactAndPinchDoubleClickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_two_quick_pinches_are_click_eligible_and_not_timer_progress(self):
        self.assertTrue(quick_pinch_click_eligible(10.0, 10.12, 0.06, 0.65))
        self.assertFalse(quick_pinch_click_eligible(10.0, 10.04, 0.06, 0.65))
        self.assertFalse(quick_pinch_click_eligible(10.0, 10.70, 0.06, 0.65))
        # OS-style double-click detection snaps the second nearby click to the first.
        point, is_second = align_double_click_point(10.1, (400, 300), 10.35, (435, 320), 0.45, 60)
        self.assertEqual(point, (400.0, 300.0))
        self.assertTrue(is_second)
        point, is_second = align_double_click_point(10.1, (400, 300), 10.7, (435, 320), 0.45, 60)
        self.assertEqual(point, (435.0, 320.0))
        self.assertFalse(is_second)
        # A release gap as short as one tracking frame must still let a genuine second quick pinch click.
        self.assertTrue(quick_pinch_reentry_eligible(
            10.0, 10.0, 10.12, 10.13, False, False, 0.8, 0.0, 0.06, 0.65
        ))
        # Any release before the 0.8s confirmation completes must restart the buffer.
        from gesture_rules import pinch_timer_gap_restarts_buffer
        self.assertTrue(pinch_timer_gap_restarts_buffer(10.0, 10.12, 10.15))

    def test_pinch_click_path_is_separate_and_handles_repinch_before_debounce(self):
        self.assertIn("def _activate_quick_pinch_click(self, point, side, now):", self.main)
        self.assertIn("self._activate_quick_pinch_click(click_point, side, now)", self.main)
        self.assertIn("quick_pinch_reentry_eligible(", self.main)
        self.assertIn("quick_pinch_click_eligible", self.main)
        self.assertIn("align_double_click_point", self.main)
        self.assertIn("self.quick_pinch_double_click_window = self.quick_grasp_double_click_window", self.main)

    def test_reentry_policy_rejects_noise_too_short_pulses_and_post_buffer_pulses(self):
        self.assertFalse(quick_pinch_reentry_eligible(
            10.0, 10.0, 10.04, 10.08, False, False, 0.8, 0.0, 0.06, 0.65
        ))
        self.assertFalse(quick_pinch_reentry_eligible(
            10.0, 10.0, 10.12, 10.15, True, False, 0.8, 0.0, 0.06, 0.65
        ))
        self.assertFalse(quick_pinch_reentry_eligible(
            10.0, 10.0, 10.86, 10.90, False, False, 0.8, 0.0, 0.06, 0.65
        ))

    @staticmethod
    def _hand(wx, wy, flip=False):
        base = [(0,0),(30,-30),(55,-50),(75,-60),(95,-65),(90,-35),(130,-38),(155,-40),(175,-41),
                (95,-10),(140,-10),(168,-10),(190,-10),(92,15),(132,17),(158,18),(178,19),
                (85,38),(115,42),(135,45),(152,47)]
        return [((wx - x) if flip else (wx + x), wy + y) for x, y in base]

    def test_shared_pivot_keeps_touching_hands_touching_at_small_scale(self):
        left = self._hand(400, 300)
        right = self._hand(782, 330, flip=True)  # middle fingertips nearly meet, wrists at different heights
        raw_gap = hand_contact_gap(left, right)
        for weight, expect_closed in ((0.0, False), (1.0, True)):
            origins = shared_contact_origins({"left": left[0], "right": right[0]}, weight)
            vl = [scale_hand_point(p, origins["left"], 0.3) for p in left]
            vr = [scale_hand_point(p, origins["right"], 0.3) for p in right]
            gap = hand_contact_gap(vl, vr)
            if expect_closed:
                self.assertAlmostEqual(gap, raw_gap*0.3, places=6)
            else:
                self.assertGreater(gap, 200.0)

    def test_shared_pivot_is_wrist_midpoint_so_pair_does_not_drift(self):
        wrists = {"left": (100.0, 240.0), "right": (400.0, 260.0)}
        self.assertEqual(shared_contact_origins(wrists, 1.0),
                         {"left": (250.0, 250.0), "right": (250.0, 250.0)})
        self.assertEqual(shared_contact_origins(wrists, 0.0), wrists)
        self.assertEqual(shared_contact_origins({"left": (1.0, 2.0)}, 1.0), {"left": (1.0, 2.0)})

    def test_contact_gap_detects_fingertip_on_palm(self):
        left = self._hand(400, 300)
        # Right index tip placed on the centre of the left palm.
        right = self._hand(400, 300, flip=True)
        cx = sum(left[i][0] for i in (0, 5, 9, 13, 17))/5.0
        cy = sum(left[i][1] for i in (0, 5, 9, 13, 17))/5.0
        dx, dy = cx - right[8][0], cy - right[8][1]
        right = [(x + dx, y + dy + 200.0) for x, y in right]  # move away vertically ...
        right = [(x, y - 200.0) for x, y in right]           # ... then back onto the palm
        self.assertLess(hand_contact_gap(left, right), 1e-6)

    def test_contact_weight_only_engages_when_really_close(self):
        self.assertEqual(shared_contact_weight(None, 28, 73), 0.0)
        self.assertEqual(shared_contact_weight(200, 28, 73), 0.0)
        self.assertEqual(shared_contact_weight(73, 28, 73), 0.0)
        self.assertEqual(shared_contact_weight(28, 28, 73), 1.0)
        self.assertEqual(shared_contact_weight(0, 28, 73), 1.0)
        mid = shared_contact_weight(50.5, 28, 73)
        self.assertAlmostEqual(mid, 0.5)

    def test_contact_easing_is_frame_rate_independent_and_converges(self):
        w = 0.0
        for _ in range(30):
            w = ease_toward(w, 1.0, 1/60)
        w2 = 0.0
        for _ in range(15):
            w2 = ease_toward(w2, 1.0, 1/30)
        self.assertAlmostEqual(w, w2, places=6)
        self.assertGreater(w, 0.99)
        self.assertEqual(ease_toward(0.3, 1.0, 5.0), 1.0)

    def test_main_runtime_uses_shared_contact_pivot(self):
        self.assertIn("gap = hand_contact_gap(screen_points[\"left\"], screen_points[\"right\"])", self.main)
        self.assertIn("origins = shared_contact_origins(wrists, weight)", self.main)
        self.assertIn("weight = ease_toward(", self.main)

    def test_finger_thickness_is_directly_proportional_to_apparent_hand_size(self):
        farther = hand_apparent_size_scale(65, 55, 180)
        closer = hand_apparent_size_scale(150, 130, 180)
        self.assertGreater(closer, farther)
        self.assertIn("return relative_palm_projection_scale(math.sqrt(max(1.0, palm_length*palm_width)), reference_size)", self.main)
        self.assertIn("reference_palm_size", self.main)
        self.assertIn('"finger_thickness": 4.24,', self.main)
        self.assertIn('"finger_thickness": (0.8, 10.0)', self.main)
        self.assertIn('value = minimum+ratio*(maximum-minimum)', self.main)
        # Finger widths stay steady when the hand rotates (foreshortening corrected).
        self.assertIn("palm_width *= self._landmark_foreshortening(points, 5, 17)", self.main)

    def test_tli_is_larger_and_the_activation_point_uses_the_same_fingertip_midpoint(self):
        self.assertIn("tx, ty = xy(landmarks[4])", self.main)
        self.assertIn("ix, iy = xy(landmarks[8])", self.main)
        self.assertIn("radius = diameter*0.5", self.main)
        self.assertIn("_scale_hand_tli_point(self.interactive_hand_anchor(points, side), points, side)", self.main)
        self.assertIn("resolve_relative_tli(landmarks, calibration)", self.main)
        self.assertIn("54px diameter", self.readme)
        self.assertIn("thumb/index fingertip midpoint", self.guide)

    def test_timer_still_requires_continuous_buffer_before_any_fill(self):
        self.assertIn("pinch_timer_release_before_buffer", self.main)
        self.assertIn("the circle stays empty during a continuous 0.8-second pinch confirmation", self.guide.lower())
        # The quick-pinch branch cannot count a separate early pulse against an old start.
        self.assertIn("self._clear_menu_pinch_sequence(side)", self.main)
        self.assertIn("self._start_menu_pinch_sequence(side, cursor, now)", self.main)


if __name__ == "__main__":
    unittest.main()
