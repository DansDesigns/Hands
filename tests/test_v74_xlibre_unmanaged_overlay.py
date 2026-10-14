from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_overlay_is_window_manager_managed_and_always_on_top_on_linux():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("class SpatialDesktop(QWidget):")
    end = source.index("    def ", source.index("self.setWindowFlags(overlay_flags)", start))
    init = source[start:end]
    assert "overlay_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool" in init
    assert "Qt.X11BypassWindowManagerHint" not in init


def test_native_stacking_reassertion_runs_without_dragging():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _maintain_native_topmost(self, now=None):")
    end = source.index("    def _sync_linux_fullscreen_geometry(self):", start)
    method = source[start:end]
    assert 'sys.platform.startswith("linux")' in method
    assert "XRaiseWindow" in method
    assert "if not target.isVisible():" in method
    assert 'active.get("mode") == "drag_window"' not in method
    assert "target = search_overlay" in method
    assert "ctypes.c_ulong(int(target.winId()))" in method
    assert "<= 0.06" in method
