"""Regression tests for Qtile work-area clipping at the bottom of the screen."""
from pathlib import Path
import unittest

from linux_screen_geometry import resolve_fullscreen_bounds

ROOT = Path(__file__).resolve().parents[1]


class LinuxBottomQtileBarFixTests(unittest.TestCase):
    def test_root_screen_restores_reserved_bottom_bar_height(self):
        bounds, native_warp_safe = resolve_fullscreen_bounds(
            (0, 0, 1920, 1032), (1920, 1080))
        self.assertEqual(bounds, (0, 0, 1920, 1080))
        self.assertTrue(native_warp_safe)

    def test_root_screen_restores_work_area_with_top_offset(self):
        bounds, native_warp_safe = resolve_fullscreen_bounds(
            (0, 24, 1920, 1032), (1920, 1080))
        self.assertEqual(bounds, (0, 0, 1920, 1080))
        self.assertTrue(native_warp_safe)

    def test_high_dpi_or_multimonitor_width_mismatch_uses_qt_coordinates(self):
        qt_bounds = (0, 0, 1536, 864)
        bounds, native_warp_safe = resolve_fullscreen_bounds(qt_bounds, (1920, 1080))
        self.assertEqual(bounds, qt_bounds)
        self.assertFalse(native_warp_safe)

    def test_implausibly_large_height_gap_is_not_assumed_to_be_a_panel(self):
        qt_bounds = (0, 0, 1920, 780)
        bounds, native_warp_safe = resolve_fullscreen_bounds(qt_bounds, (1920, 1080))
        self.assertEqual(bounds, qt_bounds)
        self.assertFalse(native_warp_safe)

    def test_native_pointer_warp_is_used_for_compatible_x11_root_geometry(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        method_start = source.index("    def _set_os_pointer_global(self, global_pos):")
        method_end = source.index("    def _set_os_pointer(self, point):", method_start)
        method = source[method_start:method_end]
        self.assertIn("_linux_native_pointer_warp_compatible", method)
        self.assertIn("XWarpPointer", method)
        self.assertIn("XFlush", method)
        self.assertIn("QCursor.setPos(global_pos)", method)

    def test_all_main_cursor_mapping_uses_reconciled_full_screen_geometry(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("def _interaction_screen_geometry(self):", source)
        for method_name, next_marker in (
            ("interactive_visual_point", "    def interactive_hand_anchor"),
            ("_smooth_cursor", "    def _notification_stack_opacity"),
            ("_scale_hand_tli_point", "    def _visual_tli_position"),
        ):
            start = source.index(f"    def {method_name}(")
            end = source.index(next_marker, start)
            self.assertIn("_interaction_screen_geometry()", source[start:end], method_name)

    def test_fullscreen_overlay_and_tpi_calibration_share_corrected_geometry(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        sync_start = source.index("    def _sync_linux_fullscreen_geometry(self):")
        sync_end = source.index("    # Repaint interval", sync_start)
        self.assertIn("screen_rect = self._interaction_screen_geometry()", source[sync_start:sync_end])
        self.assertIn("window.set_full_screen_geometry(self._interaction_screen_geometry())", source)
        self.assertIn("def set_full_screen_geometry(self, geometry):", source)


if __name__ == "__main__":
    unittest.main()
