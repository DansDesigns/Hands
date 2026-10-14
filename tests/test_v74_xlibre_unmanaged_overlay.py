from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_overlay_bypasses_x11_window_manager_on_linux_only():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("class SpatialDesktop(QWidget):")
    end = source.index("    def ", source.index("        self.setWindowFlags(overlay_flags)", start))
    init = source[start:end]
    assert 'if sys.platform.startswith("linux"):' in init
    assert "overlay_flags |= Qt.X11BypassWindowManagerHint" in init
    assert "self.setWindowFlags(overlay_flags)" in init


def test_unmanaged_flag_is_limited_to_main_overlay():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "class TliCalibrationWindow(QWidget):" in source
    assert "class HandCalibrationWizard(QWidget):" in source
    # Only the SpatialDesktop initializer adds the bypass hint.
    assert source.count("Qt.X11BypassWindowManagerHint") == 1
