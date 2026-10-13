"""Regression coverage for duplicate hand painting and glass-menu labels/icons."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V43HandLayerAndMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.main)

    def test_main_hand_layer_yields_to_search_overlay(self):
        self.assertIn("search_overlay_draws_hand_layer = bool(", self.main)
        self.assertIn("if not search_overlay_draws_hand_layer:", self.main)
        self.assertIn("offset duplicate/ghost hands, often noticed around the screen centre", self.main)
        self.assertIn("This overlay is the sole hand-visual owner while Search is visible", self.main)
        self.assertIn("if not search_overlay_draws_hand_layer:\n            for side in (\"right\", \"left\"):", self.main)
        self.assertIn("on the main window can create a fixed-centre ghost ring during menu gestures", self.main)

    def test_search_overlay_does_not_synthesize_missing_tli(self):
        self.assertIn("Do not synthesize a second/fallback TLI in this overlay.", self.main)
        self.assertIn('cursor = owner.side_cursor_positions.get(side)\n            if cursor is None:\n                continue', self.main)

    def test_home_tile_is_renamed_search_and_action_routes_to_search_bar(self):
        self.assertIn('("SEARCH", cx+116*scale, cy+79*scale, "⌕")', self.main)
        self.assertIn('QRectF(cx+66*scale, cy+35*scale, 100*scale, 88*scale), "search"', self.main)
        self.assertIn('elif action == "search":\n                    self.open_search_bar(from_voice=False)', self.main)
        self.assertNotIn('("FILES", cx+116*scale, cy+79*scale', self.main)

    def test_speech_commands_icon_is_distinct_from_stt_voice_icon(self):
        self.assertIn('("SPEECH COMMANDS", cx+116*scale, cy-101*scale, "≡")', self.main)
        self.assertIn('("VOICE ON" if self.stt_enabled else "VOICE OFF"), cx-116*scale, cy+79*scale, "◖))"', self.main)
        self.assertIn('if label == "SPEECH COMMANDS":', self.main)
        self.assertIn('if label == "SEARCH":', self.main)

    def test_button_labels_use_a_bounded_font_fit_helper(self):
        self.assertIn("def _draw_fitted_button_label", self.main)
        self.assertIn("QFontMetricsF(font)", self.main)
        self.assertIn('self._draw_fitted_button_label(painter, label_rect, button_label', self.main)
        self.assertIn('self._draw_fitted_button_label(painter, rect, label_text', self.main)

    def test_fixed_08_second_buffers_remain_unchanged(self):
        self.assertIn('self.gesture_timer_delay_seconds = 0.8  # FIXED soft gesture buffer: do not remove or shorten', self.main)
        self.assertIn('self.menu_pinch_sequence_gap_seconds = 0.80  # FIXED 0.8s soft gap for repeated pinch/re-arm', self.main)


if __name__ == "__main__":
    unittest.main()
