"""Regression tests for stable one-hand fist scroll release and launcher naming."""
from pathlib import Path
import unittest

from gesture_rules import active_fist_scroll_release_confirmed

ROOT = Path(__file__).resolve().parent.parent


class FistScrollReleaseAndLauncherNameTests(unittest.TestCase):
    def test_scroll_does_not_release_on_short_open_classification_flicker(self):
        common = dict(is_open=True, is_fist=False, is_pinch=False, fingers_closed=False)
        self.assertFalse(active_fist_scroll_release_confirmed(**common, open_duration=0.10))
        self.assertFalse(active_fist_scroll_release_confirmed(**common, open_duration=0.47))
        self.assertTrue(active_fist_scroll_release_confirmed(**common, open_duration=0.48))

    def test_folded_or_ambiguous_pose_never_confirms_scroll_release(self):
        self.assertFalse(active_fist_scroll_release_confirmed(
            is_open=True, is_fist=False, is_pinch=False, fingers_closed=True, open_duration=1.0))
        self.assertFalse(active_fist_scroll_release_confirmed(
            is_open=True, is_fist=True, is_pinch=False, fingers_closed=False, open_duration=1.0))
        self.assertFalse(active_fist_scroll_release_confirmed(
            is_open=True, is_fist=False, is_pinch=True, fingers_closed=False, open_duration=1.0))

    def test_active_scroll_tracks_a_separate_open_candidate_clock(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('active.get("scroll_open_candidate_since")', source)
        self.assertIn('active["scroll_open_candidate_since"] = candidate_since', source)
        self.assertIn('release_seconds=0.48', source)
        self.assertIn('one_hand_fist_scroll = (mode == "scroll_window"', source)
        self.assertIn('(0.48 if one_hand_fist_scroll else self.active_release_cooldown_seconds)', source)

    def test_linux_launcher_displays_new_brand_and_replaces_old_entry_in_place(self):
        source = (ROOT / "install_linux.sh").read_text(encoding="utf-8")
        self.assertIn('Name=Freehand: Spatial Computing', source)
        self.assertIn("echo 'Find Freehand: Spatial Computing in your app launcher.'", source)

    def test_windows_start_menu_uses_new_name_and_removes_old_shortcut(self):
        source = (ROOT / "install_windows.bat").read_text(encoding="utf-8")
        self.assertIn("$name='Freehand'+[char]0xFF1A+' Spatial Computing'", source)
        self.assertIn("$s.Description='Freehand: Spatial Computing'", source)
        self.assertIn("'Spatial Desktop Prototype.lnk'", source)
        self.assertIn("Remove-Item -LiteralPath $old -Force", source)


if __name__ == "__main__":
    unittest.main()
