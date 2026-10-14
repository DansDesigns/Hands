import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class StreamTTSConfigGUITests(unittest.TestCase):
    def test_voice_page_has_open_config_button_and_launcher(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('stack("streamtts_config", h=42.0)', source)
        self.assertIn('elif pressed("streamtts_config"):', source)
        self.assertNotIn('d1_tts_config', source)
        self.assertIn('"OPEN STREAMTTS CONFIG"', source)
        self.assertIn('subprocess.Popen([sys.executable, str(config_gui)]', source)

    def test_config_gui_edits_shared_settings_file(self):
        source = (ROOT / "streamtts_config.py").read_text(encoding="utf-8")
        self.assertIn('SETTINGS = ROOT / "voice_settings.json"', source)
        self.assertIn('"enabled": bool(self.enabled_var.get())', source)
        self.assertIn('"voice": voice_id', source)
        self.assertIn('"speed": round(', source)


    def test_config_gui_has_original_streamtts_voice_controls(self):
        source = (ROOT / "streamtts_config.py").read_text(encoding="utf-8")
        for option in (
            "Apply selected voice", "Download other…", "Import folder…",
            "Speech rate", "Pitch shift", "Streaming mode",
            "Test selected voice", "Save settings", "immediate", "balanced", "natural",
        ):
            self.assertIn(option, source)
        self.assertIn('"pitch": round(', source)
        self.assertIn('"mode": self.mode_var.get()', source)

    def test_runtime_reads_pitch_mode_and_system_voice(self):
        source = (ROOT / "voice_services.py").read_text(encoding="utf-8")
        self.assertIn('"pitch": 0.0', source)
        self.assertIn('"mode": "balanced"', source)
        self.assertIn('voice.startswith("system:")', source)
        self.assertIn('pitch_semitones=float(self.settings.get("pitch", 0.0))', source)

if __name__ == "__main__":
    unittest.main()
