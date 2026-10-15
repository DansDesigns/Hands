"""Lightweight live-weather support for Freehand's Virtual Garden.

Weather is fetched only while Garden Mode is active. IP geolocation is approximate;
only latitude/longitude and the current-weather request are sent to Open-Meteo.
All networking is intended to run on a background thread, never during painting.
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, time as datetime_time, timedelta
from typing import Any

USER_AGENT = "Freehand-Spatial-Computing/1.0 (Virtual Garden weather)"
WEATHER_REFRESH_SECONDS = 15 * 60


def _request_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Weather provider returned an unexpected response")
    return payload


def resolve_approximate_location(timeout: float = 4.0) -> dict[str, Any]:
    """Resolve approximate city-level location from the public IP (no GPS permission)."""
    data = _request_json("https://ipapi.co/json/", timeout=timeout)
    if data.get("error"):
        raise RuntimeError("Approximate location could not be resolved")
    latitude = float(data.get("latitude"))
    longitude = float(data.get("longitude"))
    if not (math.isfinite(latitude) and math.isfinite(longitude)
            and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Location provider returned invalid coordinates")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "city": str(data.get("city") or data.get("region") or data.get("country_name") or "Local area"),
        "country": str(data.get("country_name") or ""),
        "timezone": str(data.get("timezone") or ""),
    }


def fetch_live_weather(location: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch current Open-Meteo conditions and sunrise/sunset for an approximate location."""
    if not isinstance(location, dict) or "latitude" not in location or "longitude" not in location:
        location = resolve_approximate_location(timeout=min(4.0, timeout))
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid weather coordinates")
    query = urllib.parse.urlencode({
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "current": ("temperature_2m,relative_humidity_2m,is_day,precipitation,rain,showers,"
                    "snowfall,cloud_cover,weather_code,wind_speed_10m"),
        "daily": "sunrise,sunset",
        "forecast_days": "1",
        "timezone": "auto",
    })
    payload = _request_json("https://api.open-meteo.com/v1/forecast?" + query, timeout=timeout)
    current = payload.get("current") or {}
    daily = payload.get("daily") or {}

    def number(key: str, default: float = 0.0) -> float:
        try:
            value = float(current.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    try:
        code = int(current.get("weather_code"))
    except (TypeError, ValueError):
        code = 0
    sunrise_values = daily.get("sunrise") or []
    sunset_values = daily.get("sunset") or []
    sunrise = str(sunrise_values[0]) if sunrise_values else None
    sunset = str(sunset_values[0]) if sunset_values else None
    place = dict(location)
    return {
        "source": "open-meteo",
        "city": str(place.get("city") or place.get("region") or place.get("country") or "Local area"),
        "country": str(place.get("country") or ""),
        "location": place,
        "timezone": str(payload.get("timezone") or place.get("timezone") or ""),
        "weather_code": code,
        "temperature": number("temperature_2m", float("nan")),
        "humidity": number("relative_humidity_2m", 0),
        "is_day": bool(int(number("is_day", 1))),
        "precipitation": number("precipitation", 0),
        "rain": number("rain", 0),
        "showers": number("showers", 0),
        "snowfall": number("snowfall", 0),
        "cloud_cover": max(0.0, min(100.0, number("cloud_cover", 0))),
        "wind_speed": max(0.0, number("wind_speed_10m", 5)),
        "sunrise": sunrise,
        "sunset": sunset,
        "observed_at": str(current.get("time") or ""),
    }


def weather_kind(code: Any) -> str:
    """Map WMO weather codes into the low-cost visual effect groups used by Garden."""
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "clear"
    if value == 0:
        return "clear"
    if value == 1:
        return "mostly-clear"
    if value == 2:
        return "partly-cloudy"
    if value == 3:
        return "overcast"
    if value in (45, 48):
        return "fog"
    if 51 <= value <= 67 or 80 <= value <= 82:
        return "rain"
    if 71 <= value <= 77 or value in (85, 86):
        return "snow"
    if value in (95, 96, 99):
        return "storm"
    return "cloudy"


def calculate_solar_state(now: datetime, sunrise: str | None = None,
                          sunset: str | None = None) -> dict[str, float | bool]:
    """Return daylight progress using provider sunrise/sunset or safe local-time defaults."""
    wall_now = now.replace(tzinfo=None)

    def parse(value: str | None, default: datetime_time) -> datetime:
        if value:
            try:
                parsed = datetime.fromisoformat(str(value))
                parsed = parsed.replace(tzinfo=None)
                # Providers return today's local sunrise/sunset. If stale data crosses
                # midnight, retain the local clock time but use today's date.
                if parsed.date() != wall_now.date():
                    parsed = datetime.combine(wall_now.date(), parsed.time())
                return parsed
            except (TypeError, ValueError):
                pass
        return datetime.combine(wall_now.date(), default)

    dawn = parse(sunrise, datetime_time(6, 30))
    dusk = parse(sunset, datetime_time(19, 30))
    if dusk <= dawn:
        dawn = datetime.combine(wall_now.date(), datetime_time(6, 30))
        dusk = datetime.combine(wall_now.date(), datetime_time(19, 30))
    if dawn <= wall_now <= dusk:
        progress = min(1.0, max(0.0, (wall_now - dawn).total_seconds() /
                                max(1.0, (dusk - dawn).total_seconds())))
        solar = max(0.0, math.sin(math.pi * progress)) ** 0.58
        return {"is_day": True, "progress": progress, "night_progress": 0.0, "solar_factor": solar}
    # During night the moon crosses the opposite side of the sky smoothly: it starts
    # opposite the sun at sunset and reaches the sunrise side just before dawn.
    if wall_now < dawn:
        night_start = dusk - timedelta(days=1)
        night_end = dawn
    else:
        night_start = dusk
        night_end = dawn + timedelta(days=1)
    night_duration = max(1.0, (night_end-night_start).total_seconds())
    night_progress = min(1.0, max(0.0, (wall_now-night_start).total_seconds()/night_duration))
    return {
        "is_day": False,
        "progress": 0.0 if wall_now < dawn else 1.0,
        "night_progress": night_progress,
        "solar_factor": 0.0,
    }
