"""Regression checks for v33 settings pages, voice shortcuts and feedback."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import voice_services

ROOT = Path(__file__).resolve().parent.parent


class V33SettingsVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.voice = (ROOT / "voice_services.py").read_text(encoding="utf-8")

    def test_settings_are_separated_into_three_pages(self):
        self.assertIn('"interaction": QRectF', self.main)
        self.assertIn('"notifications": QRectF', self.main)
        self.assertIn('"voice": QRectF', self.main)
        self.assertNotIn('"NOTIFICATION PLACEMENT"', self.main)
        self.assertNotIn('"VOICE SETTINGS"', self.main)
        self.assertIn('"gesture_map": "GESTURE MAP"', self.main)

    def test_notification_positions_are_persisted_and_used(self):
        self.assertIn('UI_SETTINGS_FILE = ROOT / "ui_settings.json"', self.main)
        self.assertIn("save_ui_preferences(self.ui_preferences)", self.main)
        self.assertIn('self.ui_preferences.get("heard_bar_position", "bottom-center")', self.main)
        self.assertIn('self.ui_preferences.get("paused_notice_position", "bottom-center")', self.main)
        self.assertIn('NOTIFICATION_POSITIONS = ("top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right")', self.main)

    def test_show_menu_and_context_are_global_voice_commands(self):
        start = self.main.index("    def on_speech(self, text):")
        end = self.main.index("    def apply_desktop_mode", start)
        method = self.main[start:end]
        self.assertIn('"show menu"', method)
        self.assertIn('"show context"', method)
        self.assertLess(method.index('"show menu"'), method.index("dialog.set_dictation_text(text, final=True)"))
        self.assertIn("def _toggle_glass_menu_from_voice", self.main)
        self.assertIn("def _toggle_context_menu_from_voice", self.main)

    def test_scroll_inversion_applies_to_both_scroll_paths(self):
        self.assertIn('"invert_scroll": False', self.main)
        self.assertIn('invert_scroll={str(bool(values.get(\'invert_scroll\', False))).lower()}', self.main)
        self.assertIn('wheel_steps = int(steps) if self.invert_scroll else -int(steps)', self.main)
        self.assertIn('pyautogui.scroll(wheel_steps, _pause=False)', self.main)
        self.assertIn('WM_MOUSEWHEEL straight to a captured top-level HWND', self.main)
        self.assertIn('"SCROLL  ·  "+("INVERTED" if self.invert_scroll else "NORMAL")', self.main)

    def test_drag_engagement_has_animated_hand_wave(self):
        self.assertIn('active.get("mode") != "drag_window"', self.main)
        self.assertIn("phase = (now*1.25 + wave_index/3.0) % 1.0", self.main)
        self.assertIn("MOVE MODE  ·  SHAKE COMPLETE", self.main)

    def test_tts_preload_is_background_and_voice_speed_wider(self):
        self.assertIn('name="Kokoro-preload"', self.voice)
        self.assertIn("def _preload_engine(self)", self.voice)
        self.assertIn("0.7 <= speed <= 1.5", self.voice)
        self.assertIn("def update_settings(self, *, enabled=None, voice=None, speed=None, language=None, persist=True)", self.voice)
        # The Voice page speed slider is drawn by the shared settings slider renderer.
        self.assertIn('("tts_speed", float(self.tts.settings["speed"]), 0.7, 1.5, "SPEECH SPEED")', self.main)

    def test_tts_settings_can_save_voice_and_speed(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "voice_settings.json"
            with patch.object(voice_services, "TTS_SETTINGS", settings_path):
                tts = voice_services.KokoroTTS()
                tts.update_settings(voice="bm_george", speed=1.4, enabled=True)
                saved = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["voice"], "bm_george")
                self.assertEqual(saved["speed"], 1.4)
                self.assertTrue(saved["enabled"])
                tts.update_settings(speed=1.6)
                saved = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["speed"], 1.4)  # out-of-range value is rejected
                tts.close()


if __name__ == "__main__":
    unittest.main()
