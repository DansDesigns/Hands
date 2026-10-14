from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_os_pointer_uses_overlay_local_to_global_mapping():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _set_os_pointer(self, point):")
    end = source.index("    def ", start + 8)
    method = source[start:end]
    assert "self.mapToGlobal(QPoint(round(point.x()), round(point.y())))" in method
    assert "screen.geometry().topLeft() + QPoint" not in method


def test_linux_overlay_is_synchronized_to_full_screen_geometry():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "def _sync_linux_fullscreen_geometry(self):" in source
    assert "screen_rect = screen.geometry()" in source
    assert "window._sync_linux_fullscreen_geometry()" in source


def test_linux_drag_reasserts_overlay_above_dragged_window():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _maintain_native_topmost(self, now=None):")
    end = source.index("    def _sync_linux_fullscreen_geometry(self):", start)
    method = source[start:end]
    assert 'sys.platform.startswith("linux")' in method
    assert 'active.get("mode") == "drag_window"' in method
    assert "XRaiseWindow" in method


def test_linux_installer_creates_launcher_without_a_second_installer():
    installer = (ROOT / "install_linux.sh").read_text(encoding="utf-8")
    assert 'cat > "$APPS_DIR/spatial-desktop-prototype.desktop" <<DESKTOP' in installer
    assert 'Exec="$APP_DIR/spatial_desktop_launcher.sh"' in installer
    assert 'TryExec="$APP_DIR/spatial_desktop_launcher.sh"' in installer
    assert "Path=" not in installer
    assert 'chmod +x "$APP_DIR/run_linux.sh" "$APP_DIR/spatial_desktop_launcher.sh"' in installer
    assert 'sh "$APP_DIR/install_desktop_launcher.sh"' not in installer
    assert not (ROOT / "install_desktop_launcher.sh").exists()
