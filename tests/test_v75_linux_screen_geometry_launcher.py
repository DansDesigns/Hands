from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_os_pointer_maps_virtual_hand_points_to_physical_screen_coordinates():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _virtual_point_to_global(self, point):")
    end = source.index("    def _release_fist_mouse(self, side):", start)
    method = source[start:end]
    assert "QApplication.primaryScreen()" in method
    assert "geometry.topLeft() + pixel" in method
    assert "self.mapToGlobal(pixel)" in method
    assert "self._virtual_point_to_global(point)" in method
    assert "QCursor.setPos(global_pos)" in method


def test_virtual_tli_clamps_to_physical_screen_not_qtile_work_area():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _scale_hand_tli_point(self, cursor, points=None, side=None):")
    end = source.index("    def _visual_tli_position", start)
    method = source[start:end]
    assert "screen.geometry()" in method
    assert "geometry.width()" in method
    assert "geometry.height()" in method
    assert "self.width()-1.0" not in method
    assert "self.height()-1.0" not in method


def test_fullscreen_geometry_is_rechecked_during_live_linux_tracking():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    topmost_start = source.index("    def _maintain_native_topmost(self, now=None):")
    geometry_start = source.index("    def _sync_linux_fullscreen_geometry(self):", topmost_start)
    topmost = source[topmost_start:geometry_start]
    geometry = source[geometry_start:source.index("    # Repaint interval", geometry_start)]
    assert "_last_linux_geometry_sync_check" in topmost
    assert "self._sync_linux_fullscreen_geometry()" in topmost
    assert "reported_origin == screen_rect.topLeft()" in geometry


def test_linux_overlay_is_synchronized_to_full_screen_geometry():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "def _sync_linux_fullscreen_geometry(self):" in source
    assert "screen_rect = screen.geometry()" in source
    assert "window._sync_linux_fullscreen_geometry()" in source


def test_linux_overlay_reasserts_native_topmost_even_without_dragging():
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


def test_linux_desktop_launcher_uses_local_freehand_icon_without_fetching():
    installer = (ROOT / "install_linux.sh").read_text(encoding="utf-8")
    assert 'ICON_PATH="$APP_DIR/Freehand_logo_small.png"' in installer
    assert "Icon=$ICON_PATH" in installer
    assert "curl " not in installer
    assert "wget " not in installer
    assert "raw.githubusercontent.com" not in installer


def test_windows_start_menu_shortcut_uses_local_freehand_icon_without_fetching():
    installer = (ROOT / "install_windows.bat").read_text(encoding="utf-8")
    assert 'set "ICON_PNG=%CD%\\Freehand_logo_small.png"' in installer
    assert "Freehand_logo_small.ico" in installer
    assert "$s.IconLocation=$env:ICON_ICO + ',0'" in installer
    icon_setup = installer[installer.index("[9/9] Assigning the local Freehand launcher icon"):installer.index('set "STARTMENU=')]
    assert "Invoke-WebRequest" not in icon_setup
    assert "raw.githubusercontent.com" not in icon_setup


def test_linux_installer_creates_launcher_without_a_second_installer():
    installer = (ROOT / "install_linux.sh").read_text(encoding="utf-8")
    assert 'cat > "$APPS_DIR/spatial-desktop-prototype.desktop" <<DESKTOP' in installer
    assert 'Exec="$APP_DIR/run_linux.sh"' in installer
    assert 'TryExec="$APP_DIR/run_linux.sh"' in installer
    assert "Path=" not in installer
    assert 'chmod +x "$APP_DIR/run_linux.sh"' in installer
    assert 'sh "$APP_DIR/install_desktop_launcher.sh"' not in installer
    assert not (ROOT / "install_desktop_launcher.sh").exists()


def test_cursor_smoothing_uses_full_screen_height_not_qtile_work_area():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _smooth_cursor(self, side, target, now):")
    end = source.index("    def _notification_stack_opacity", start)
    method = source[start:end]
    assert "screen.geometry()" in method
    assert "screen_geometry.height()" in method
    assert "else self.height()" in method  # fallback only when screen data is unavailable


def test_bottom_edge_detection_and_two_hand_contact_use_full_screen_dimensions():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    edge_start = source.index("            edge_geometry = screen.geometry()")
    edge_end = source.index("            # This toggle remains active", edge_start)
    edge_block = source[edge_start:edge_end]
    assert "edge_geometry.height()" in edge_block
    assert "cursor.y() >= edge_height*0.945" in edge_block
    contact_start = source.index("    def _update_hand_contact_display(self):")
    contact_end = source.index("    def _hand_visual_transform", contact_start)
    contact_block = source[contact_start:contact_end]
    assert "contact_geometry.height()" in contact_block
    assert "px_y = max(1.0, contact_height-1)/0.68" in contact_block


def test_linux_overlay_has_native_x11_full_screen_geometry_fallback():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _sync_linux_fullscreen_geometry(self):")
    end = source.index("    # Repaint interval", start)
    method = source[start:end]
    assert "screen_rect = screen.geometry()" in method
    assert "XMoveResizeWindow" in method
    assert "screen_rect.height()" in method
