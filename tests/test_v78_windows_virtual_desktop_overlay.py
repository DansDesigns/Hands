from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from windows_virtual_desktop import _GUID, pin_window_on_all_virtual_desktops


class WindowsVirtualDesktopOverlayTests(unittest.TestCase):
    def test_virtual_desktop_guid_is_parsed_as_windows_guid_layout(self):
        guid = _GUID.parse("C2F03A33-21F5-47FA-B4BB-156362A2F239")
        self.assertEqual(guid.Data1, 0xC2F03A33)
        self.assertEqual(guid.Data2, 0x21F5)
        self.assertEqual(guid.Data3, 0x47FA)
        self.assertEqual(bytes(guid.Data4), bytes.fromhex("B4BB156362A2F239"))

    def test_pinning_helper_is_safe_on_non_windows(self):
        if not sys.platform.startswith("win"):
            self.assertFalse(pin_window_on_all_virtual_desktops(12345))

    def test_main_overlay_retries_pin_until_windows_shell_accepts_it(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from windows_virtual_desktop import pin_window_on_all_virtual_desktops", source)
        start = source.index("    def _maintain_native_topmost(self, now=None):")
        end = source.index("    def _sync_linux_fullscreen_geometry(self):", start)
        method = source[start:end]
        self.assertIn("pin_window_on_all_virtual_desktops(int(self.winId()))", method)
        self.assertIn("_windows_virtual_desktop_pinned", method)
        self.assertIn(">= 2.0", method)
        self.assertIn('if not sys.platform.startswith("win"):', method)

    def test_windows_uses_a_dedicated_clickthrough_hand_layer_and_pins_it(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("class SearchHandOverlay(QWidget):")
        end = source.index("class VoiceReplySignals(QObject):", start)
        overlay = source[start:end]
        self.assertIn('overlay_kind = Qt.Window if sys.platform.startswith("win") else Qt.Tool', overlay)
        self.assertIn("windows_global_layer", overlay)
        self.assertIn("pin_window_on_all_virtual_desktops(int(self.winId()))", overlay)
        self.assertIn("self._global_hand_layer_active", overlay)
        self.assertIn("SetWindowPos(hwnd, wintypes.HWND(-1)", overlay)
        self.assertIn("ctypes.c_ssize_t", overlay)
        self.assertIn('getattr(owner, "custom_gesture_rings"', overlay)
        self.assertIn("Task View/Alt+Tab", overlay)
        self.assertNotIn("0x00000080", overlay)

    def test_dedicated_hand_layer_keeps_thumbs_up_tracking_timer_visible(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("class SearchHandOverlay(QWidget):")
        end = source.index("class VoiceReplySignals(QObject):", start)
        overlay = source[start:end]
        self.assertIn('getattr(owner, "thumbs_up_hold_started", {})', overlay)
        self.assertIn('getattr(owner, "thumbs_up_toggle_latched", {})', overlay)
        self.assertIn("owner.gesture_tracking_toggle_hold_seconds", overlay)
        self.assertIn("QColor(105, 240, 255, 245)", overlay)
        self.assertIn("owner._scale_hand_tli_point(", overlay)

    def test_virtual_desktop_pin_is_verified_after_pin_call(self):
        source = (ROOT / "windows_virtual_desktop.py").read_text(encoding="utf-8")
        self.assertIn("IVirtualDesktopPinnedApps::PinView", source)
        self.assertIn("IVirtualDesktopPinnedApps::IsViewPinned", source)
        self.assertIn("verify_hr >= 0 and bool(confirmed.value)", source)

    def test_linux_qtile_bypass_and_physical_top_bar_mapping_remain_unchanged(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("overlay_flags |= Qt.X11BypassWindowManagerHint", source)
        self.assertIn("def _virtual_point_to_global(self, point):", source)
        self.assertIn("def _sync_linux_fullscreen_geometry(self):", source)


if __name__ == "__main__":
    unittest.main()
