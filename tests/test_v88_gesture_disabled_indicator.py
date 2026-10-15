"""Regression checks for the visible no-gesture indicator on virtual hands."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GestureDisabledIndicatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_disabled_indicator_is_gated_by_per_hand_gesture_state(self):
        start = self.main.index("    def _paint_gesture_disabled_indicator(")
        end = self.main.index("    def _paint_hand_skeleton(", start)
        helper = self.main[start:end]
        self.assertIn('getattr(self, "gesture_tracking_enabled", {}).get(side, True)', helper)
        self.assertIn("return", helper)
        self.assertIn("not points or len(points) < 18", helper)

    def test_indicator_is_a_red_circle_with_diagonal_slash(self):
        start = self.main.index("    def _paint_gesture_disabled_indicator(")
        end = self.main.index("    def _paint_hand_skeleton(", start)
        helper = self.main[start:end]
        self.assertIn("painter.drawEllipse(center, radius, radius)", helper)
        self.assertIn("painter.drawLine(", helper)
        self.assertIn("QColor(255, 20, 34, 245)", helper)
        self.assertIn("diameter*0.085", helper)

    def test_dedicated_windows_search_overlay_paints_disabled_indicator_over_hand(self):
        start = self.main.index("    def _paint_frame(self, _event):", self.main.index("class SearchHandOverlay"))
        end = self.main.index("class VoiceReplySignals(QObject):", start)
        frame = self.main[start:end]
        self.assertIn("owner._paint_hand_skeleton(painter, points, side)", frame)
        self.assertIn("owner._paint_gesture_disabled_indicator(painter, points, side)", frame)
        self.assertLess(frame.index("owner._paint_hand_skeleton(painter, points, side)"),
                        frame.index("owner._paint_gesture_disabled_indicator(painter, points, side)"))

    def test_main_overlay_paints_disabled_indicator_over_hand(self):
        self.assertIn("self._paint_hand_skeleton(painter, points, side)\n                self._paint_gesture_disabled_indicator(painter, points, side)", self.main)

    def test_visuals_remain_per_hand_for_thumb_and_voice_toggles(self):
        start = self.main.index("    def set_freehand_gestures_enabled(")
        end = self.main.index("    def set_hand_tracking_paused(", start)
        voice_toggle = self.main[start:end]
        self.assertIn('sides = ("left", "right")', voice_toggle)
        self.assertIn("self.gesture_tracking_enabled[side] = enabled", voice_toggle)
        start = self.main.index("            thumbs_up = False if calibration_modal_frame else self._is_thumbs_up(landmarks)")
        end = self.main.index("            # Gesture tracking and cursor/TLI tracking are one per-hand switch.", start)
        thumb_toggle = self.main[start:end]
        self.assertIn("self.gesture_tracking_enabled[side] = not self.gesture_tracking_enabled.get(side, True)", thumb_toggle)


if __name__ == "__main__":
    unittest.main()
