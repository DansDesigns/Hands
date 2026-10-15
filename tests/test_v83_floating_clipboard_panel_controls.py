"""Regression coverage for moving the floating clipboard and per-item delete controls."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


class FloatingClipboardPanelControlsTests(unittest.TestCase):
    def test_panel_position_can_be_overridden_and_is_screen_clamped(self):
        self.assertIn("self.floating_clipboard_position = None", MAIN)
        self.assertIn('saved_position = getattr(self, "floating_clipboard_position", None)', MAIN)
        self.assertIn("self.width()-panel_w-4.0", MAIN)
        self.assertIn("self.height()-panel_h-4.0", MAIN)

    def test_title_bar_is_hit_tested_as_a_move_handle(self):
        self.assertIn('"header": header', MAIN)
        self.assertIn('if layout["header"].contains(point):', MAIN)
        self.assertIn('clipboard_state.get("control") == "header"', MAIN)
