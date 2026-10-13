"""v36 regression checks for physical hand mapping, slider stability and glass styling."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from gesture_rules import is_plausible_hand_landmarks

ROOT = Path(__file__).resolve().parent.parent

class V36HandSliderPriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.rules = (ROOT / "gesture_rules.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")

    def test_mediapipe_handedness_is_normalized_to_physical_sides(self):
        self.assertIn('label = {"left": "right", "right": "left"}.get(model_label, model_label)', self.main)

    def test_priority_setting_drives_gesture_order_and_guard(self):
        self.assertIn('preferred = self.cursor_lock_priority if self.cursor_lock_priority in ("left", "right") else "right"', self.main)
        self.assertIn('gesture_order = (preferred, nonpreferred)', self.main)
        self.assertIn('side != preferred and preferred in cursor_points', self.main)

    def test_color_swatches_and_hitboxes_use_matching_sides(self):
        self.assertIn('("LEFT", cx-48*scale, self.left_click_color)', self.main)
        self.assertIn('("RIGHT", cx+48*scale, self.right_click_color)', self.main)
        self.assertIn('self._cycle_hand_colour("left")', self.main)
        self.assertIn('self._cycle_hand_colour("right")', self.main)

    def test_pinch_can_snap_to_settings_slider(self):
        self.assertIn('if (is_fist or is_pinch) else None', self.main)
        self.assertIn('Pinch on a slider is promoted to the same snapped/pinned slider drag as fist.', self.main)

    def test_slider_position_requires_three_stable_frames_after_large_jump(self):
        self.assertIn('self.slider_cursor_jump_candidate', self.main)
        self.assertIn('accept_cursor = count >= 3', self.main)
        self.assertIn('Keep the knob and TLI pinned to the last safe point', self.main)

    def test_plausible_hand_filter_rejects_low_confidence_and_degenerate_shape(self):
        self.assertIn('is_plausible_hand_landmarks(points, score)', self.main)
        self.assertIn('min_score=0.60', self.rules)
        points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        self.assertFalse(is_plausible_hand_landmarks(points, 0.99))
        points = [SimpleNamespace(x=0.2+(i%5)*0.12, y=0.2+(i//5)*0.15, z=0.0) for i in range(21)]
        self.assertFalse(is_plausible_hand_landmarks(points, 0.30))
        # A plausible spread of 21 landmarks with strong confidence should be retained.
        points = [SimpleNamespace(x=0.40+(i%5)*0.025, y=0.30+(i//5)*0.045, z=0.0) for i in range(21)]
        self.assertTrue(is_plausible_hand_landmarks(points, 0.95))

    def test_detection_requires_consecutive_stable_frames(self):
        self.assertIn('self._candidate_streak[label] >= 3', self.main)
        self.assertIn('self._candidate_center[label]', self.main)

    def test_frosted_glass_layers_exist_for_menu_and_settings(self):
        self.assertIn('def _paint_frosted_backdrop', self.main)
        self.assertIn('def _paint_pinch_menu', self.main)
        self.assertIn('self._paint_frosted_backdrop(painter, panel, 15, opacity=0.15)', self.main)
        self.assertIn('soft halos remain inside the fading layer', self.main.lower())
        self.assertIn('soft, layered halo edges', self.gestures)

if __name__ == "__main__":
    unittest.main()
