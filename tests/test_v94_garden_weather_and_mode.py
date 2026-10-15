"""Regression tests for Garden Mode's live weather, day/night and lightweight render path."""
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import garden_weather


def _source():
    return (ROOT / "main.py").read_text(encoding="utf-8")


def test_weather_codes_are_mapped_to_scene_effect_groups():
    assert garden_weather.weather_kind(0) == "clear"
    assert garden_weather.weather_kind(1) == "mostly-clear"
    assert garden_weather.weather_kind(2) == "partly-cloudy"
    assert garden_weather.weather_kind(3) == "overcast"
    assert garden_weather.weather_kind(45) == "fog"
    assert garden_weather.weather_kind(61) == "rain"
    assert garden_weather.weather_kind(75) == "snow"
    assert garden_weather.weather_kind(95) == "storm"
    assert garden_weather.weather_kind(None) == "clear"


def test_sunlight_uses_daylight_progress_and_moon_is_used_at_night():
    noon = garden_weather.calculate_solar_state(datetime(2026, 6, 1, 12, 0),
                                                "2026-06-01T05:00", "2026-06-01T21:00")
    midnight = garden_weather.calculate_solar_state(datetime(2026, 6, 1, 23, 0),
                                                    "2026-06-01T05:00", "2026-06-01T21:00")
    before_dawn = garden_weather.calculate_solar_state(datetime(2026, 6, 1, 4, 0),
                                                       "2026-06-01T05:00", "2026-06-01T21:00")
    assert noon["is_day"] and noon["solar_factor"] > 0.95
    assert not midnight["is_day"] and midnight["solar_factor"] == 0.0 and midnight["progress"] == 1.0
    assert 0.0 < midnight["night_progress"] < 1.0
    assert not before_dawn["is_day"] and before_dawn["progress"] == 0.0
    assert before_dawn["night_progress"] > midnight["night_progress"]


def test_live_weather_payload_uses_open_meteo_current_and_sunrise_sunset(monkeypatch):
    def fake_request_json(url, timeout=5.0):
        assert "api.open-meteo.com/v1/forecast" in url
        assert "current=" in url and "daily=sunrise%2Csunset" in url
        return {
            "timezone": "Europe/London",
            "current": {
                "time": "2026-06-01T12:00", "temperature_2m": 20.5,
                "relative_humidity_2m": 48, "is_day": 1, "precipitation": 0,
                "rain": 0, "showers": 0, "snowfall": 0, "cloud_cover": 17,
                "weather_code": 1, "wind_speed_10m": 8.5,
            },
            "daily": {"sunrise": ["2026-06-01T05:00"], "sunset": ["2026-06-01T21:00"]},
        }
    monkeypatch.setattr(garden_weather, "_request_json", fake_request_json)
    result = garden_weather.fetch_live_weather(
        location={"latitude": 51.5, "longitude": -0.1, "city": "Example City", "country": "UK"}
    )
    assert result["source"] == "open-meteo"
    assert result["city"] == "Example City"
    assert result["weather_code"] == 1
    assert result["temperature"] == 20.5
    assert result["sunrise"] == "2026-06-01T05:00"
    assert result["sunset"] == "2026-06-01T21:00"


def test_garden_mode_hides_dock_visual_and_interaction_and_caches_background():
    source = _source()
    dock = source[source.index("def _paint_virtual_app_dock("):source.index("def _capture_glass_backdrop(")]
    hit_test = source[source.index("def _virtual_pen_ui_contains("):source.index("def _handle_virtual_dock_click(")]
    click = source[source.index("def _handle_virtual_dock_click("):source.index("def _virtual_normalized_point(")]
    garden = source[source.index("def _paint_virtual_garden("):source.index("def _save_virtual_desktop_data(")]
    assert "self.virtual_garden_mode or" in dock
    assert 'getattr(self, "virtual_garden_mode", False)' in hit_test
    assert "self.virtual_garden_mode" in click
    assert "_garden_background_cache_key" in garden
    assert "_paint_garden_weather_effects" in garden
    assert "def _garden_grass_specs_for_size(" in source
    assert "def _paint_garden_weather_label(" in source


def test_weather_is_fetched_off_thread_only_for_garden_and_has_attribution():
    source = _source()
    worker = source[source.index("def _maybe_refresh_garden_weather("):source.index("def _garden_scene_state(")]
    state = source[source.index("def _garden_scene_state("):source.index("def _garden_mix_colour(")]
    label = source[source.index("def _paint_garden_weather_label("):source.index("def _paint_virtual_garden(")]
    assert "if not getattr(self, \"virtual_garden_mode\", False)" in worker
    assert "threading.Thread" in worker and "fetch_live_weather" in worker
    assert "calculate_solar_state" in state and "weather_kind" in state
    assert "Open-Meteo" in label and "WEATHER OFFLINE" in label


def test_virtual_desktop_dock_and_weather_changes_do_not_touch_qtile_mapping():
    source = _source()
    mapping = source[source.index("def _interaction_screen_geometry("):source.index("def _virtual_point_to_global(")]
    geometry = (ROOT / "linux_screen_geometry.py").read_text(encoding="utf-8")
    assert "resolve_fullscreen_bounds" in mapping
    assert "X11" in geometry and "root" in geometry.lower()
