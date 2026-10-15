"""Regression checks for spoken confirmation of thumbs-up gesture toggles."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ThumbToggleTTSAnnouncementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        marker = "            thumbs_up = False if calibration_modal_frame else self._is_thumbs_up(landmarks)"
        start = cls.main.index(marker)
        end = cls.main.index("            # Gesture tracking and cursor/TLI tracking are one per-hand switch.", start)
        cls.thumb_toggle = cls.main[start:end]

    def test_thumbs_up_toggle_speaks_on_and_off_phrases(self):
        self.assertIn('spoken_state = "on" if self.gesture_tracking_enabled[side] else "off"', self.thumb_toggle)
        self.assertIn('self.tts.speak(f"Freehand gestures {spoken_state}.")', self.thumb_toggle)

    def test_simultaneous_same_state_toggles_are_debounced(self):
        self.assertIn('last_toggle_tts[0] == spoken_state', self.thumb_toggle)
        self.assertIn('now-last_toggle_tts[1] < 0.8', self.thumb_toggle)

    def test_tts_confirmation_only_occurs_on_actual_toggle_branch(self):
        self.assertIn('if (now-self.thumbs_up_hold_started[side] >= self.gesture_tracking_toggle_hold_seconds and', self.thumb_toggle)
        self.assertLess(self.thumb_toggle.index('if (now-self.thumbs_up_hold_started[side]'),
                        self.thumb_toggle.index('self.tts.speak(f"Freehand gestures {spoken_state}.")'))


if __name__ == "__main__":
    unittest.main()
