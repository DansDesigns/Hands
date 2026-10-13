"""Regression checks for skin tone, TLI defaults, edge-on tracking, and fixed soft buffer."""
from pathlib import Path
import unittest

from gesture_rules import is_plausible_hand_landmarks

ROOT = Path(__file__).resolve().parent.parent


class V41SkinTrackingBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")
        cls.settings = (ROOT / "gestures.txt").read_text(encoding="utf-8")

    def test_skin_darkness_has_persistent_setting_and_hands_tab_slider(self):
        self.assertIn('"skin_tone_darkness": 0.0', self.main)
        self.assertIn('"skin_tone_darkness": (0.0, 100.0)', self.main)
        self.assertIn('("skin_tone_darkness", 52, self.skin_tone_darkness, 0.0, 100.0, "SKIN TONE DARKNESS")', self.main)
        self.assertIn('skin_color(246, 207, 179, 225)', self.main)
        self.assertIn('A single gradient over the final union', self.main)
        self.assertIn('(' + '"skin_tone_darkness"' + ', 0.5)', self.main)
        self.assertIn('if self.settings_page == "hands":', self.main)

    def test_default_tli_colours_are_left_red_right_blue(self):
        self.assertIn('"left_click_color": "#ff5555"', self.main)
        self.assertIn('"right_click_color": "#5091ff"', self.main)
        self.assertIn('left_click_color=#ff5555', self.settings)
        self.assertIn('right_click_color=#5091ff', self.settings)

    def test_side_on_geometry_accepts_a_thin_projected_hand(self):
        # Simplified 21-point chain: narrow x projection, clear y span and palm geometry.
        pts = [(0.500, 0.80, 0.0) for _ in range(21)]
        pts[0] = (0.500, 0.82, 0.00)
        pts[1] = (0.493, 0.78, -0.01)
        pts[5] = (0.504, 0.71, -0.02)
        pts[9] = (0.502, 0.68, -0.03)
        pts[13] = (0.497, 0.72, -0.01)
        pts[17] = (0.495, 0.75, 0.00)
        for i, y in zip(range(2, 21), [0.77,0.75,0.73,0.75,0.70,0.66,0.63,0.60,0.67,0.64,0.61,0.58,0.69,0.66,0.63,0.60,0.72,0.69,0.66]):
            pts[i] = (0.500 + ((i % 3)-1)*0.006, y, -0.01*(i%4))
        self.assertTrue(is_plausible_hand_landmarks(pts, 0.72))

    def test_fixed_soft_buffer_is_explicitly_preserved(self):
        self.assertIn('self.gesture_timer_delay_seconds = 0.8  # FIXED soft gesture buffer: do not remove or shorten', self.main)
        self.assertIn('self.menu_pinch_sequence_gap_seconds = 0.80  # FIXED 0.8s soft gap for repeated pinch/re-arm', self.main)
        self.assertIn('now-release_started >= self.gesture_timer_delay_seconds', self.main)
        self.assertIn('release_for <= self.menu_pinch_sequence_gap_seconds', self.main)
        self.assertIn('DO NOT REMOVE: the fixed 0.8-second continuous pinch-confirmation buffer', self.settings)


    def test_phantom_hardening_does_not_cache_or_render_stale_landmarks(self):
        self.assertNotIn("self._last_confirmed", self.main)
        self.assertNotIn("landmark_dropout_grace_seconds", self.main)
        self.assertIn("self._candidate_streak[label] >= 3", self.main)
        self.assertIn("min_hand_detection_confidence=0.60", self.main)
        self.assertIn("min_hand_presence_confidence=0.58", self.main)
        self.assertIn("if not ok:", self.main)
        self.assertIn("A camera read failure must not leave the previous frame's hands on screen.", self.main)
        self.assertIn("self._candidate_misses[side] > 1", self.main)

    def test_side_on_exception_requires_depth_and_confidence(self):
        self.assertIn("z_span >= 0.025", self.rules)
        self.assertIn("confidence >= 0.66", self.rules)
        # A fake flat line/edge-on blob has no depth support and must not be accepted.
        flat = [(0.5, 0.8 - i * 0.01, 0.0) for i in range(21)]
        self.assertFalse(is_plausible_hand_landmarks(flat, 0.99))



if __name__ == "__main__":
    unittest.main()
