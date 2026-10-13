"""Regression tests ensuring TLI diameter starts at 30px by default."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent


class TliDiameterDefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.gestures = (ROOT / "gestures.txt").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_bundled_settings_use_30px_not_a_saved_12px_override(self):
        matches = re.findall(r"^\s*tli_diameter\s*=\s*([0-9.]+)\s*$", self.gestures, re.MULTILINE)
        self.assertEqual(matches, ["30.0"])

    def test_code_fallback_and_startup_value_are_30px(self):
        self.assertIn('"tli_diameter": 30.0', self.main)
        self.assertIn('settings.get("tli_diameter", 30.0)', self.main)
        self.assertIn('float(getattr(self, "tli_diameter", 30.0))', self.main)

    def test_settings_range_and_documentation_agree_on_30px_default(self):
        self.assertIn('("tli_diameter", 64, self.tli_diameter, 12.0, 96.0, "TLI DIAMETER")', self.main)
        self.assertIn('diameter=12–96px (default 30px)', self.gestures)
        self.assertIn('defaults to 30px', self.readme)


if __name__ == "__main__":
    unittest.main()
