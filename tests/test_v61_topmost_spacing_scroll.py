"""Regression checks for native topmost hand overlay, calibration spacing, and 10x scroll range."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V61TopmostSpacingScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_active_hand_overlay_reasserts_topmost_without_focus(self):
        self.assertIn("def _maintain_native_topmost(self, now=None):", self.main)
        self.assertIn("self._maintain_native_topmost(now)", self.main)
        self.assertIn("wintypes.HWND(-1)", self.main)
        self.assertIn("0x0001 | 0x0002 | 0x0010", self.main)
        self.assertIn("target = search_overlay", self.main)
        self.assertIn("target = self.tli_calibration_window", self.main)
        self.assertIn("target = self.hand_calibration_window", self.main)

    def test_close_banner_uses_bottom_panel_geometry(self):
        self.assertIn("cy+(panel_height*0.5-54.0)*scale", self.main)
        self.assertIn('painter.drawText(close_banner, Qt.AlignCenter, "CLOSE")', self.main)

    def test_calibration_buttons_have_wider_gap_and_shared_hit_geometry(self):
        self.assertIn("def _tli_calibration_button_rects(self, cx=None, cy=None, scale=None):", self.main)
        self.assertIn('"left": QRectF(cx-210.0*scale, cy+181.0*scale, 190.0*scale, 36.0*scale)', self.main)
        self.assertIn('"right": QRectF(cx+20.0*scale, cy+181.0*scale, 190.0*scale, 36.0*scale)', self.main)
        self.assertEqual(self.main.count("self._tli_calibration_button_rects(cx, cy, scale).items()"), 2)

    def test_scroll_acceleration_range_is_10x_through_slider_and_persistence(self):
        self.assertIn('("scroll_acceleration", 101, self.scroll_acceleration, 0.5, 10.0, "SCROLL ACCELERATION")', self.main)
        self.assertIn('"scroll_acceleration": (0.5, 10.0)', self.main)
        self.assertIn("scroll_acceleration={clamped('scroll_acceleration', 0.5, 10.0, 1.0):.2f}", self.main)
        self.assertIn("scroll_acceleration=1.00", self.gestures)
        self.assertIn("0.5–10.0x", self.gestures)
        self.assertIn("0.5–10.0×", self.readme)


if __name__ == "__main__":
    unittest.main()
