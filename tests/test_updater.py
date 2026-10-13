import unittest
import tempfile
from pathlib import Path

from updater import compare_versions, read_version, install_update, REPOSITORY_URL, REMOTE_VERSION_URL


class VersionCheckerTests(unittest.TestCase):
    def test_version_order(self):
        self.assertEqual(compare_versions("0.9", "0.8"), 1)
        self.assertEqual(compare_versions("0.8", "0.8.0"), 0)
        self.assertEqual(compare_versions("0.7.9", "0.8"), -1)
        self.assertEqual(compare_versions("v1.2.0", "1.1.99"), 1)

    def test_rejects_non_numeric_versions(self):
        with self.assertRaises(ValueError):
            compare_versions("latest", "0.8")

    def test_repository_target_and_current_version(self):
        self.assertEqual(read_version(), "0.8")
        self.assertEqual(REPOSITORY_URL, "https://github.com/DansDesigns/Hands")
        self.assertEqual(REMOTE_VERSION_URL,
                         "https://raw.githubusercontent.com/DansDesigns/Hands/main/version.txt")

    def test_install_update_refreshes_local_version_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "main.py").write_text("# updated app\n", encoding="utf-8")
            (source / "version.txt").write_text("1.4\n", encoding="utf-8")
            (destination / "main.py").write_text("# old app\n", encoding="utf-8")
            (destination / "version.txt").write_text("1.3\n", encoding="utf-8")
            install_update(source, destination)
            self.assertEqual((destination / "version.txt").read_text(encoding="utf-8").strip(), "1.4")


if __name__ == "__main__":
    unittest.main()
