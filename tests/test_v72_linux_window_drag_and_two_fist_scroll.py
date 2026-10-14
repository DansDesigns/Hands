"""Regression checks for Linux native window dragging and mode-aware two-fist gestures."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent

class LinuxWindowDragAndTwoFistScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")

    def test_linux_native_window_discovery_geometry_and_move(self):
        self.assertIn('["wmctrl", "-lG"]', self.main)
        self.assertIn('["wmctrl", "-ir", hex(int(hwnd)), "-e"', self.main)
        self.assertIn('sys.platform.startswith("linux")', self.main)

    def test_two_fist_action_is_mode_aware(self):
        self.assertIn('clutch_mode = "scroll_window" if self.scroll_mode_enabled else "drag_window"', self.main)
        self.assertIn('active_clutch["two_hand_clutch"] = True', self.main)
        self.assertIn('active.get("two_hand_clutch")', self.main)

    def test_docs_describe_linux_requirements_and_release(self):
        self.assertIn('sudo apt install wmctrl xdotool', self.readme)
        self.assertIn('opening either hand stops', self.readme.lower())
        self.assertIn('TWO-FIST CLUTCH', self.gestures)

if __name__ == "__main__":
    unittest.main()
