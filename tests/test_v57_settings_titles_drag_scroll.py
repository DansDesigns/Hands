"""Regression tests for tab-led Settings pages and distance-scaled drag scrolling."""
from pathlib import Path
import unittest

from gesture_rules import drag_scroll_rate_from_offset

ROOT = Path(__file__).resolve().parent.parent


class V57SettingsAndDragScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_tabbed_settings_pages_do_not_draw_redundant_titles(self):
        self.assertNotIn('"INTERACTION SETTINGS"', self.main)
        self.assertNotIn('"HAND SETTINGS"', self.main)
        self.assertNotIn('"NOTIFICATION PLACEMENT"', self.main)
        self.assertNotIn('"VOICE SETTINGS"', self.main)
        self.assertIn('tab_labels = {"interaction": "INTERACTION", "hands": "HANDS"', self.main)
        # These two pages have no tab row, so keep their identifying headings.
        self.assertIn('"gesture_map": "GESTURE MAP"', self.main)
        self.assertIn('"speech_commands": "SPEECH COMMANDS"', self.main)

    def test_scroll_pauses_near_anchor_and_accelerates_with_distance(self):
        self.assertEqual(drag_scroll_rate_from_offset(0), 0.0)
        self.assertEqual(drag_scroll_rate_from_offset(5), 0.0)
        near = drag_scroll_rate_from_offset(20)
        medium = drag_scroll_rate_from_offset(100)
        far = drag_scroll_rate_from_offset(300)
        self.assertGreater(near, 0.0)
        self.assertGreater(medium, near)
        self.assertGreater(far, medium)
        self.assertLessEqual(far, 20.0)
        self.assertAlmostEqual(drag_scroll_rate_from_offset(420), 20.0)

    def test_scroll_direction_follows_side_of_anchor(self):
        self.assertEqual(drag_scroll_rate_from_offset(100), -drag_scroll_rate_from_offset(-100))
        self.assertEqual(drag_scroll_rate_from_offset(-100), -drag_scroll_rate_from_offset(100))

    def test_active_scroll_uses_anchor_distance_and_faster_response(self):
        self.assertIn('drag_scroll_rate_from_offset(', self.main)
        self.assertIn('previous_weight=0.55, max_delta=420.0', self.main)
        self.assertIn('notches_per_second = signed_rate*self.hand_tracking_sensitivity*0.8', self.main)
        self.assertIn('origin = active.get("origin", cursor)', self.main)
        self.assertIn('displacement_y = float(cursor.y() - origin.y())', self.main)

    def test_detached_drag_scroll_never_repositions_native_pointer(self):
        self.assertIn('"side": side,', self.main)
        start = self.main.index('    def _scroll_target_window(')
        end = self.main.index('    def _begin_action_from_popup(', start)
        scroll_method = self.main[start:end]
        self.assertIn('detached = bool(side in ("left", "right")', scroll_method)
        self.assertIn('if screen_point is not None and not detached:', scroll_method)
        self.assertIn('if detached and screen_point is not None and sys.platform.startswith("win"):', scroll_method)
        self.assertIn('if not detached:', scroll_method)


if __name__ == "__main__":
    unittest.main()
