"""Regression coverage for dragging app windows without selecting our own hand layer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_windows_native_hit_test_excludes_every_window_owned_by_spatial_desktop():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _find_native_window_at(self, point, prefer_smallest=False):")
    end = source.index("    def _native_window_rect(self, hwnd):", start)
    method = source[start:end]

    assert "own_pid = os.getpid()" in method
    assert "get_window_pid = user32.GetWindowThreadProcessId" in method
    assert "get_window_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]" in method
    assert "int(process_id.value) == own_pid" in method
    assert "if not thread_id or int(process_id.value) == own_pid:" in method
    # The process-owner check must run during each EnumWindows callback, before a
    # same-process transparent overlay can be added to the list of hit-test matches.
    assert method.index("if not thread_id or int(process_id.value) == own_pid:") < method.index('found["matches"].append')


def test_two_hand_drag_keeps_hand_visuals_owned_by_the_overlay_not_the_target_window():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _find_native_window_at(self, point, prefer_smallest=False):")
    end = source.index("    def _native_window_rect(self, hwnd):", start)
    method = source[start:end]
    assert "transparent, full-screen window" in method
    assert "moves the hand layer instead of the application" in method
