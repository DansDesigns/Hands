import unittest
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

class V35SourceRegressionTests(unittest.TestCase):
    def test_two_fist_drag_replaces_shake_activation(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        self.assertIn('side == "right"', source)
        self.assertIn('left_travel <= 24.0 and right_travel >= 34.0', source)
        self.assertIn('self.fist_target_window["right"] = self._find_native_window_at(right_start)', source)
        self.assertIn('both hands must stay in fists', source)

    def test_cursor_priority_setting_persists(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        self.assertIn('cursor_lock_priority', source)
        self.assertIn('CURSOR PRIORITY', source)
        self.assertIn('self.cursor_lock_priority in cursor_points', source)

    def test_tli_uses_real_thumb_index_fingertip_midpoint(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        self.assertIn('tx, ty = xy(landmarks[4])', source)
        self.assertIn('ix, iy = xy(landmarks[8])', source)
        self.assertIn('return (tx+ix)*0.5, (ty+iy)*0.5', source)

    def test_dictation_commands_and_asr_correction(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        self.assertIn('pause dictation', source)
        self.assertIn('resume dictation', source)
        self.assertIn('(?:end|and)\\s+dictation', source)
        commands = (ROOT / 'voice_pc_commands.py').read_text(encoding='utf-8')
        self.assertIn('"and dictation"', commands)

    def test_pinch_menu_right_click_is_direct_action(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        self.assertIn('self._activate_canvas_right_click(QPointF(cursor), side, now)', source)
        self.assertIn('Right-click sent at the Pinch Menu selection point', source)

if __name__ == '__main__':
    unittest.main()
