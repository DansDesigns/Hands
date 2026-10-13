"""Tests for automatic, non-blocking Face Welcome model setup."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import face_welcome
from face_welcome import FaceWelcomeService


class _FakeDetector:
    def setInputSize(self, _size):
        pass


class _FakeRecognizer:
    pass


class FaceAutoInstallTests(unittest.TestCase):
    def test_enrollment_does_not_start_sampling_deadline_before_models_ready(self):
        statuses = []
        service = FaceWelcomeService(lambda: None, status_callback=statuses.append)
        with patch.object(service, "start"):
            self.assertTrue(service.begin_enrollment("Taylor"))
        self.assertEqual(service._collect_deadline, 0.0)
        self.assertGreater(service._prepare_deadline, 0.0)
        self.assertTrue(any("automatically" in msg.lower() for msg in statuses))

    def test_missing_models_triggers_auto_download_not_manual_script_message(self):
        statuses = []
        service = FaceWelcomeService(lambda: None, status_callback=statuses.append)
        detector = Path("user-cache/face_detection_yunet_2026may.onnx")
        recognizer = Path("user-cache/face_recognition_sface_2021dec_int8.onnx")
        fake_cv2 = types.SimpleNamespace(
            FaceDetectorYN_create=lambda *args: _FakeDetector(),
            FaceRecognizerSF_create=lambda *args: _FakeRecognizer(),
        )
        with patch.object(face_welcome, "find_face_model_pair", return_value=None), patch.object(
            face_welcome, "download_face_models", return_value=(detector, recognizer)
        ) as downloader, patch.dict(sys.modules, {"cv2": fake_cv2}):
            self.assertTrue(service._ensure_models())
        downloader.assert_called_once()
        self.assertTrue(any("downloading" in text.lower() for text in statuses))
        self.assertFalse(any("run install_face_models.py" in text.lower() for text in statuses))

    def test_download_failure_is_graceful_and_throttled(self):
        statuses = []
        service = FaceWelcomeService(lambda: None, status_callback=statuses.append)
        with patch.object(face_welcome, "find_face_model_pair", return_value=None), patch.object(
            face_welcome, "download_face_models", side_effect=RuntimeError("offline")
        ):
            self.assertFalse(service._ensure_models())
            # Retry is throttled rather than repeatedly spamming/failing every frame.
            self.assertFalse(service._ensure_models())
        self.assertTrue(any("will retry" in text.lower() for text in statuses))

    def test_runtime_and_installer_share_model_download_urls(self):
        installer = (Path(__file__).parent / "install_face_models.py").read_text(encoding="utf-8")
        self.assertIn("download_face_models", installer)
        self.assertIn("FACE_MODELS", installer)


if __name__ == "__main__":
    unittest.main()
