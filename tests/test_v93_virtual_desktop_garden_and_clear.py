"""Regression checks for the separate Garden scene and persistent canvas clear control."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / "main.py").read_text(encoding="utf-8")


def test_garden_mode_is_persisted_with_backward_compatible_default():
    source = _source()
    defaults = source[source.index("VIRTUAL_DESKTOP_DEFAULTS ="):source.index("def _finite_clamped")]
    assert '"garden_mode": False' in defaults
    normalizer = source[source.index("def normalise_virtual_desktop_data("):source.index("def load_virtual_desktop_data(")]
    assert '"garden_mode": (background_source.get("garden_mode")' in normalizer
    assert 'self.virtual_garden_mode = bool(virtual_bg.get("garden_mode", False))' in source
    save = source[source.index("def _save_virtual_desktop_data("):source.index("def _schedule_virtual_desktop_save(")]
    assert '"garden_mode": bool(self.virtual_garden_mode)' in save


def test_garden_is_a_separate_scene_and_hides_the_saved_canvas():
    source = _source()
    background = source[source.index("def _paint_virtual_background("):source.index("def _paint_virtual_garden(")]
    assert 'if getattr(self, "virtual_garden_mode", False)' in background
    assert 'self._paint_virtual_garden(painter, width, height)' in background
    paint = source[source.index("def _paint_frame("):source.index("# Paint scroll-circle backdrops")]
    assert 'if not self.virtual_garden_mode:' in paint
    assert 'painter.drawImage(QPointF(0, 0), canvas_layer)' in paint


def test_garden_scene_has_cached_weather_sky_sun_moon_hills_and_hand_reactive_grass():
    source = _source()
    garden = source[source.index("def _paint_virtual_garden_background("):source.index("def _save_virtual_desktop_data(")]
    for marker in ("QRadialGradient", "sun_radius", "moon_radius", "distant = QPainterPath()",
                   "middle = QPainterPath()", "foreground = QPainterPath()",
                   "side_cursor_positions", "influence_radius", "path.cubicTo", "painter.drawPath(path)",
                   "_garden_background_cache", "_garden_grass_specs_for_size"):
        assert marker in garden
    assert "cached, weather-aware Garden scene" in source


def test_virtual_settings_has_garden_mode_toggle_and_saves_state():
    source = _source()
    layout = source[source.index("def _settings_layout("):source.index("def _settings_control_rects(")]
    assert 'stack("virtual_garden_toggle")' in layout
    click = source[source.index('if self.settings_page == "virtual":'):source.index('if self.settings_page == "hands":', source.index('if self.settings_page == "virtual":'))]
    assert 'if pressed("virtual_garden_toggle"):' in click
    assert "self.virtual_garden_mode = not self.virtual_garden_mode" in click
    assert "self._schedule_virtual_desktop_save()" in click
    assert '"GARDEN MODE  ·  "+("ON" if self.virtual_garden_mode else "OFF")' in source


def test_pen_popup_includes_clear_canvas_and_clear_resets_saved_strokes():
    source = _source()
    geometry = source[source.index("def _virtual_pen_popup_geometry("):source.index("def _virtual_pen_ui_contains(")]
    assert 'items["clear"]' in geometry
    handler = source[source.index("def _handle_virtual_dock_click("):source.index("def _virtual_normalized_point(")]
    clear = handler[handler.index('elif name == "clear":'):handler.index("else:", handler.index('elif name == "clear":'))]
    assert "self.virtual_canvas_strokes = []" in clear
    assert "self.virtual_canvas_layer_dirty = True" in clear
    assert "self._save_virtual_desktop_data()" in clear
    dock = source[source.index("def _paint_virtual_app_dock("):source.index("def _capture_glass_backdrop(")]
    assert 'elif name == "clear":' in dock
    assert 'self.virtual_garden_mode or' in dock
    assert '"PEN · GRID MODE" if self.virtual_garden_mode' not in dock


def test_garden_mode_does_not_modify_qtile_screen_mapping_implementation():
    source = _source()
    mapping = source[source.index("def _interaction_screen_geometry("):source.index("def _virtual_point_to_global(")]
    assert "resolve_fullscreen_bounds" in mapping
    geometry = (ROOT / "linux_screen_geometry.py").read_text(encoding="utf-8")
    assert "X11" in geometry and "root" in geometry.lower()
