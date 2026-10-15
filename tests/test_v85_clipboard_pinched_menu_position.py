"""Regression tests for opening the floating clipboard where the Pinch Menu was."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")
TREE = ast.parse(MAIN)


def method_source(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(MAIN, node)
    raise AssertionError(f"Missing method: {name}")


class ClipboardPinchMenuPositionTests(unittest.TestCase):
    def test_menu_geometry_is_captured_before_closing(self):
        source = method_source("_begin_action_from_popup")
        capture = source.index("clipboard_menu_anchor, _ = self._action_menu_geometry(side)")
        close = source.index("self._set_pinch_menu_open(side, False, now)")
        self.assertLess(capture, close)

    def test_clipboard_is_centered_on_menu_anchor_when_opened_from_pinch_menu(self):
        source = method_source("_begin_action_from_popup")
        self.assertIn("self._position_floating_clipboard_over_rect(clipboard_menu_anchor)", source)
        position = method_source("_position_floating_clipboard_over_rect")
        self.assertIn("anchor_rect.center().x() - panel.width() * 0.5", position)
        self.assertIn("anchor_rect.center().y() - panel.height() * 0.5", position)

    def test_clipboard_position_is_clamped_to_screen(self):
        position = method_source("_position_floating_clipboard_over_rect")
        self.assertIn("self.width() - panel.width() - 4.0", position)
        self.assertIn("self.height() - panel.height() - 4.0", position)

    def test_hiding_clipboard_does_not_apply_menu_repositioning(self):
        source = method_source("_begin_action_from_popup")
        guard = source.index("if action == \"clipboard_toggle\" and not self.floating_clipboard_visible:")
        anchor_call = source.index("self._position_floating_clipboard_over_rect(clipboard_menu_anchor)")
        self.assertLess(guard, anchor_call)
        self.assertIn("if self.floating_clipboard_visible and clipboard_menu_anchor is not None:", source)

    def test_main_source_parses(self):
        ast.parse(MAIN)


if __name__ == "__main__":
    unittest.main()
