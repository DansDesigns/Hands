"""Regression tests for Virtual desktop appearance, dock drawing, and persistence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / "main.py").read_text(encoding="utf-8")


def test_virtual_settings_page_exposes_grid_style_and_customisation_sliders():
    source = _source()
    tab_rects = source[source.index("    def _settings_tab_rects("):source.index("    def _notification_rect(")]
    assert '"virtual"' in tab_rects
    assert '"virtual": "VIRTUAL"' in source
    layout = source[source.index("    def _settings_layout("):source.index("    def _settings_control_rects(")]
    assert 'elif page == "virtual"' in layout
    assert 'stack("virtual_grid_toggle")' in layout
    assert 'stack("virtual_background_style")' in layout
    slider_method = source[source.index("    def _settings_slider_rows("):source.index("    def _settings_slider_specs(")]
    for key in ("virtual_grid_brightness", "virtual_grid_glow", "virtual_grid_spacing", "virtual_grid_thickness"):
        assert key in slider_method


def test_virtual_background_has_four_presets_and_glowing_grid_passes():
    source = _source()
    constants = source[source.index("VIRTUAL_BACKGROUND_STYLES ="):source.index("VIRTUAL_PEN_PALETTE =")]
    assert all(name in constants for name in ("MIDNIGHT BLUE", "NEON PURPLE", "DEEP TEAL", "AMBER GRID"))
    method = source[source.index("    def _paint_virtual_background("):source.index("    def _save_virtual_desktop_data(")]
    assert "self.virtual_grid_brightness" in method
    assert "self.virtual_grid_glow" in method
    assert "outer" in method and "middle" in method and "core" in method


def test_virtual_app_dock_has_pen_palette_eraser_and_pointer_modes():
    source = _source()
    palette = source[source.index("VIRTUAL_PEN_PALETTE ="):source.index("VIRTUAL_DESKTOP_DEFAULTS =")]
    assert all(name in palette for name in ("WHITE", "CYAN", "RED", "ORANGE", "GREEN", "PURPLE", "YELLOW"))
    geometry = source[source.index("    def _virtual_pen_popup_geometry("):source.index("    def _virtual_pen_ui_contains(")]
    assert 'items["eraser"]' in geometry
    assert 'items["pointer"]' in geometry
    assert "pen.top()-height-12.0" in geometry
    click = source[source.index("    def _handle_virtual_dock_click("):source.index("    def _virtual_normalized_point(")]
    assert 'name == "eraser"' in click
    assert 'name == "pointer"' in click
    assert 'self.virtual_pen_color = color' in click


def test_virtual_pen_draws_with_pinch_and_stores_normalized_points():
    source = _source()
    normalizer = source[source.index("def normalise_virtual_desktop_data("):source.index("def load_virtual_desktop_data(")]
    assert '"points": points' in normalizer
    assert "min(1.0, max(0.0, x))" in normalizer
    normalised_point = source[source.index("    def _virtual_normalized_point("):source.index("    def _virtual_pixel_point(")]
    assert "float(point.x())/max(1.0, float(self.width()))" in normalised_point
    handler = source[source.index("    def _handle_virtual_pen_frame("):source.index("    def _paint_virtual_app_dock(")]
    assert "not is_pinch" in handler
    assert '"points": []' in handler
    assert "self._append_virtual_pen_point(cursor)" in handler


def test_virtual_drawing_and_background_are_persisted_and_flushed_on_exit():
    source = _source()
    assert 'VIRTUAL_DESKTOP_DATA_FILE = ROOT / "virtual_desktop.json"' in source
    assert "def save_virtual_desktop_data(value, path=VIRTUAL_DESKTOP_DATA_FILE):" in source
    save = source[source.index("    def _save_virtual_desktop_data("):source.index("    def _schedule_virtual_desktop_save(")]
    assert '"strokes": self.virtual_canvas_strokes' in save
    assert '"glow": float(self.virtual_grid_glow)' in save
    finish = source[source.index("    def _finish_virtual_pen_stroke("):source.index("    def _handle_virtual_pen_frame(")]
    assert "self._save_virtual_desktop_data()" in finish
    close = source[source.index("    def closeEvent(self, event):"):source.index("    def mouseMoveEvent(self, event):")]
    assert "self._finish_virtual_pen_stroke()" in close
    assert "self._save_virtual_desktop_data()" in close


def test_virtual_dock_is_only_drawn_on_virtual_desktop_and_does_not_touch_qtile_mapping():
    source = _source()
    dock = source[source.index("    def _paint_virtual_app_dock("):source.index("    def _capture_glass_backdrop(")]
    assert 'self.desktop_mode != "virtual"' in dock
    # Screen mapping is a protected regression: virtual drawing is layered on top and
    # must not replace or clamp the native X11 geometry/cursor route.
    geometry_file = (ROOT / "linux_screen_geometry.py").read_text(encoding="utf-8")
    assert "resolve_fullscreen_bounds" in geometry_file
    main_mapping = source[source.index("    def _interaction_screen_geometry(self):"):source.index("    def _virtual_point_to_global(self, point):")]
    assert "resolve_fullscreen_bounds" in main_mapping
