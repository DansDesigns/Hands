"""Dependency-light tests for local face-profile privacy and lifecycle."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import face_welcome
from face_welcome import FaceWelcomeService


class FaceWelcomeProfileTests(unittest.TestCase):
    def test_profile_stores_embedding_and_name_not_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(face_welcome, "PROFILE_DIR", root), patch.object(
                face_welcome, "PROFILE_FILE", root / "face_profile.json"
            ):
                service = FaceWelcomeService(lambda: None)
                vector = np.arange(1, 129, dtype=np.float32)
                service._write_profile(vector, "Taylor", time.time())
                saved = json.loads((root / "face_profile.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["username"], "Taylor")
                self.assertEqual(len(saved["embedding"]), 128)
                self.assertNotIn("image", saved)
                self.assertNotIn("photo", saved)
                self.assertNotIn("frame", saved)
                self.assertEqual(service.profile_name(), "Taylor")
                self.assertTrue(service.has_profile())
                self.assertTrue(service.remove_profile())
                self.assertFalse((root / "face_profile.json").exists())

    def test_bad_or_missing_profile_is_not_enrolled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(face_welcome, "PROFILE_DIR", root), patch.object(
                face_welcome, "PROFILE_FILE", root / "missing.json"
            ):
                service = FaceWelcomeService(lambda: None)
                self.assertFalse(service.has_profile())
                self.assertEqual(service.profile_name(), "")

    def test_last_seen_defaults_safely(self):
        before = time.time()
        value = FaceWelcomeService._profile_last_seen({"last_seen_epoch": "invalid"})
        self.assertGreaterEqual(value, before)
        self.assertLess(value, before + 2.0)


if __name__ == "__main__":
    unittest.main()
