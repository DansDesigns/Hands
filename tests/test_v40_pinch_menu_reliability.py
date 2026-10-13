"""Regression checks for v40 Pinch Menu transitions, blur latency and tracking state."""
from pathlib import Path
import unittest

from gesture_rules import should_recognize_menu_pinch

ROOT = Path(__file__).resolve().parent.parent


class V40PinchMenuReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")
        cls.guide = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_model_fallback_is_skin_like_and_live_landmark_driven(self):
        self.assertIn("def _paint_hand_skeleton(self, painter, points, side=None):", self.main)
        self.assertIn("gradient = QLinearGradient(bounds.topLeft(), bounds.bottomRight())", self.main)
        self.assertIn("hand_shape = hand_shape.united(stroker.createStroke(line))", self.main)
        self.assertIn("Qt.RoundCap, Qt.RoundJoin", self.main)
        self.assertIn("unrigged and untextured", self.readme)
        self.assertIn("CGTrader hand model listing", self.readme)

    def test_menu_uses_eased_fade_and_does_not_capture_at_timer_completion(self):
        self.assertIn("def _pinch_menu_alpha", self.main)
        self.assertIn("Smoothstep avoids a sharp pop", self.main)
        self.assertIn("def _paint_gesture_timer_fade", self.main)
        self.assertIn("Do not synchronously grab/blur the screen here", self.main)
        self.assertIn("_paint_pinch_menu(painter, side, now)", self.main)

    def test_blur_is_stronger_and_reduced_resolution(self):
        self.assertIn("cv2.GaussianBlur", self.main)
        self.assertIn("scale = min(1.0, 1100.0/max(width, height))", self.main)
        self.assertIn("sigma=40.0", self.main)
        self.assertIn("sigma=36.0", self.main)

    def test_pinch_rearm_requires_a_clear_open_hand_after_fist(self):
        self.assertIn("self.gesture_timer_delay_seconds = 0.8", self.main)
        self.assertIn("self.menu_pinch_sequence_gap_seconds = 0.80", self.main)
        self.assertIn("self.active_release_cooldown_seconds = 0.20", self.main)
        self.assertIn("if raw_open and not raw_pinch:", self.main)
        self.assertNotIn("Soft 0.8s re-arm bound", self.main)

    def test_confirmed_fist_keeps_priority_until_a_clear_open_hand(self):
        self.assertIn("effective_fist = (", self.main)
        self.assertIn("or self.fist_release_required_for_pinch[side]", self.main)
        self.assertIn("if raw_open and not raw_pinch:", self.main)
        self.assertIn("keep the fist action in control until the user clearly opens the hand", self.main)
        self.assertNotIn('active.get("mode") == "scroll_window" and is_pinch and not is_fist', self.main)
        self.assertNotIn('self.status = "Scroll stopped by pinch — Pinch Menu timer started"', self.main)
        self.assertNotIn("A stable pinch after a fist is itself a valid release transition.", self.main)

    def test_notification_does_not_speak_and_persists_when_disabled(self):
        self.assertIn('self.hand_tracking_notice_until = float("inf")', self.main)
        self.assertIn('self.hand_tracking_notice_text = "Hand tracking enabled"', self.main)
        self.assertIn("now + 5.0", self.main)
        self.assertNotIn('self.tts.speak(f"{side} hand tracking enabled")', self.main)
        self.assertIn("No tracking-toggle TTS", self.guide)

    def test_pinch_detector_remains_tolerant_without_accepting_closed_fist(self):
        self.assertTrue(should_recognize_menu_pinch(
            normalized_gap=0.35, finger_straightness=[0.48, 0.91, 0.77, 0.54]))
        self.assertFalse(should_recognize_menu_pinch(
            normalized_gap=0.30, finger_straightness=[0.42, 0.48, 0.43, 0.39]))


if __name__ == "__main__":
    unittest.main()
