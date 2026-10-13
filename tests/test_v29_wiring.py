"""Static safeguards for v28 GUI lifecycle and voice command routing."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class V29WiringTests(unittest.TestCase):
    def test_search_close_does_not_quit_application(self):
        launcher = (ROOT / "search_launcher.py").read_text(encoding="utf-8")
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("Qt.WA_QuitOnClose, False", launcher)
        self.assertIn("app.setQuitOnLastWindowClosed(False)", main)
        self.assertIn("self.close_button.clicked.connect(self.close)", launcher)

    def test_voice_commands_are_processed_before_dictation(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        start = source.index("    def on_speech(self, text):")
        end = source.index("    def apply_desktop_mode", start)
        method = source[start:end]
        self.assertLess(method.index('"close window"'), method.index("dialog.set_dictation_text(text, final=True)"))
        self.assertIn('"open user folder"', method)

    def test_face_welcome_is_opt_in_and_installer_wired(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        win = (ROOT / "install_windows.bat").read_text(encoding="utf-8")
        linux = (ROOT / "install_linux.sh").read_text(encoding="utf-8")
        self.assertIn("ENROL FACE FOR WELCOME-BACK", source)
        self.assertIn("not secure authentication", source)
        self.assertIn("install_face_models.py", win)
        self.assertIn("install_face_models.py", linux)


    def test_face_enrollment_uses_automatic_background_setup(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        service = (ROOT / "face_welcome.py").read_text(encoding="utf-8")
        self.assertNotIn("Run install_face_models.py from the project folder", source)
        self.assertIn("download_face_models(FACE_MODELS, self.status_callback)", service)
        self.assertIn("MODEL_PREPARE_TIMEOUT_SECONDS", service)
        self.assertIn("self._collect_deadline = 0.0", service)

    def test_face_detector_uses_opencv5_dynamic_model(self):
        source = (ROOT / "face_welcome.py").read_text(encoding="utf-8")
        installer = (ROOT / "install_face_models.py").read_text(encoding="utf-8")
        self.assertIn("face_detection_yunet_2026may.onnx", source)
        self.assertIn("download_face_models", installer)
        self.assertIn('OPENCV_FORCE_DNN_ENGINE", "4"', source)
        self.assertNotIn("face_detection_yunet_2023mar_int8.onnx", source)


if __name__ == "__main__":
    unittest.main()
