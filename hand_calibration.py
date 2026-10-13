"""First-run hand/proximity calibration helpers for Spatial Desktop.

The wizard stores a screen-touch translation anchor, per-hand apparent palm-size
references, and a measured fingertip-contact ratio. Runtime code uses these values to
keep hand overlay sizing and two-hand proximity thresholds relative to the tracked hands.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def default_calibration() -> dict:
    return {
        "version": 1,
        "completed": False,
        "calibrated": False,
        "skipped": False,
        "screen_offset": {"x": 0.0, "y": 0.0},
        "screen_anchor_camera": None,
        "extended_palm_size": {"left": None, "right": None},
        "reference_palm_size": {"left": None, "right": None},
        "proximity_touch_ratio": 0.28,
        "proximity_interaction_ratio": 2.5,
    }


def _xy(point: Any) -> tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def palm_measurements(landmarks) -> dict[str, float] | None:
    """Measure palm length/width in normalized camera coordinates."""
    if landmarks is None or len(landmarks) < 21:
        return None
    try:
        points = [_xy(landmarks[i]) for i in range(21)]
        if not all(math.isfinite(v) for p in points for v in p):
            return None
        def dist(a, b):
            return math.hypot(points[a][0]-points[b][0], points[a][1]-points[b][1])
        width = dist(5, 17)
        length = dist(0, 9)
        size = math.sqrt(max(0.0, width*length))
        if min(width, length, size) <= 1e-5:
            return None
        return {"width": width, "length": length, "size": size}
    except (TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return None


def fingertip_contact_ratio(left_landmarks, right_landmarks) -> float | None:
    """Measure closest cross-hand fingertip gap divided by average palm size."""
    left = palm_measurements(left_landmarks)
    right = palm_measurements(right_landmarks)
    if left is None or right is None:
        return None
    try:
        tip_ids = (4, 8, 12, 16, 20)
        left_tips = [_xy(left_landmarks[i]) for i in tip_ids]
        right_tips = [_xy(right_landmarks[i]) for i in tip_ids]
        gap = min(math.hypot(a[0]-b[0], a[1]-b[1]) for a in left_tips for b in right_tips)
        palm_unit = (left["size"] + right["size"])*0.5
        ratio = gap / max(1e-6, palm_unit)
        return ratio if math.isfinite(ratio) else None
    except (TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return None


def mean_finite(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else None


def median_finite(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(clean) if clean else None


def screen_touch_offset(camera_x: float, camera_y: float,
                        target_x: float = 0.5, target_y: float = 0.08) -> dict[str, float]:
    """Translation correction that maps a known camera landmark to its touch target."""
    # Match SpatialDesktop.interactive_visual_point's current fixed camera crop.
    mapped_x = (float(camera_x)-0.20)/(0.80-0.20)
    mapped_y = (float(camera_y)-0.16)/(0.84-0.16)
    dx = float(target_x)-mapped_x
    dy = float(target_y)-mapped_y
    if not all(math.isfinite(v) for v in (dx, dy)):
        return {"x": 0.0, "y": 0.0}
    # Vertical camera crops often require a larger correction to place the fingertip
    # directly beneath the camera. Keep horizontal translation conservative, but allow
    # enough vertical range for a real arm-extended calibration pose.
    return {"x": min(0.35, max(-0.35, dx)),
            "y": min(0.65, max(-0.65, dy))}


def relative_palm_projection_scale(current_palm_size: float,
                                   reference_palm_size: float,
                                   min_scale: float = 0.45,
                                   max_scale: float = 2.25) -> float:
    """Scale stroke widths based on apparent hand size relative to a calibrated pose."""
    current = max(1e-6, float(current_palm_size))
    reference = max(1e-6, float(reference_palm_size))
    if not all(math.isfinite(v) for v in (current, reference, min_scale, max_scale)):
        return 1.0
    return min(float(max_scale), max(float(min_scale), current/reference))


def proximity_thresholds(palm_sizes: Mapping[str, float],
                         touch_ratio: float = 0.28,
                         interaction_ratio: float = 2.5) -> tuple[float, float]:
    """Convert calibrated palm-relative ratios into current screen-space thresholds."""
    values = [float(value) for value in palm_sizes.values()
              if value is not None and math.isfinite(float(value)) and float(value) > 0]
    unit = statistics.mean(values) if values else 40.0
    touch = max(8.0, unit*min(0.75, max(0.12, float(touch_ratio))))
    interaction = max(touch+1.0, unit*min(5.0, max(1.2, float(interaction_ratio))))
    return touch, interaction


def load_hand_calibration(path: Path) -> dict:
    """Read persisted hand calibration, filling safe defaults for missing fields."""
    result = default_calibration()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return result
    if not isinstance(data, dict):
        return result
    result["completed"] = bool(data.get("completed", False))
    result["calibrated"] = bool(data.get("calibrated", False))
    result["skipped"] = bool(data.get("skipped", False))
    offset = data.get("screen_offset")
    if isinstance(offset, dict):
        try:
            x, y = float(offset.get("x", 0.0)), float(offset.get("y", 0.0))
            if math.isfinite(x) and math.isfinite(y) and abs(x) <= 0.35 and abs(y) <= 0.65:
                result["screen_offset"] = {"x": x, "y": y}
        except (TypeError, ValueError):
            pass
    anchor = data.get("screen_anchor_camera")
    if isinstance(anchor, dict):
        try:
            x, y = float(anchor["x"]), float(anchor["y"])
            if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                result["screen_anchor_camera"] = {"x": x, "y": y}
        except (KeyError, TypeError, ValueError):
            pass
    for key in ("extended_palm_size", "reference_palm_size"):
        supplied = data.get(key)
        if isinstance(supplied, dict):
            for side in ("left", "right"):
                try:
                    value = float(supplied.get(side))
                    if math.isfinite(value) and 1e-5 < value < 1.0:
                        result[key][side] = value
                except (TypeError, ValueError):
                    pass
    for key, default, low, high in (
        ("proximity_touch_ratio", 0.28, 0.12, 0.75),
        ("proximity_interaction_ratio", 2.5, 1.2, 5.0),
    ):
        try:
            value = float(data.get(key, default))
            if math.isfinite(value):
                result[key] = min(high, max(low, value))
        except (TypeError, ValueError):
            pass
    return result


def save_hand_calibration(path: Path, values: Mapping[str, Any]) -> bool:
    """Atomically persist only validated calibration fields."""
    defaults = default_calibration()
    data = dict(defaults)
    data["completed"] = bool(values.get("completed", False))
    data["calibrated"] = bool(values.get("calibrated", False))
    data["skipped"] = bool(values.get("skipped", False))
    supplied_offset = values.get("screen_offset", defaults["screen_offset"])
    try:
        ox = float(supplied_offset.get("x", 0.0))
        oy = float(supplied_offset.get("y", 0.0))
        if not (math.isfinite(ox) and math.isfinite(oy)):
            return False
        data["screen_offset"] = {"x": min(0.35, max(-0.35, ox)),
                                 "y": min(0.65, max(-0.65, oy))}
    except (AttributeError, TypeError, ValueError):
        return False
    data["screen_anchor_camera"] = values.get("screen_anchor_camera")
    for key in ("extended_palm_size", "reference_palm_size"):
        supplied = values.get(key, {})
        data[key] = {side: supplied.get(side) for side in ("left", "right")} if isinstance(supplied, Mapping) else defaults[key]
    try:
        data["proximity_touch_ratio"] = min(0.75, max(0.12, float(values.get("proximity_touch_ratio", 0.28))))
        data["proximity_interaction_ratio"] = min(5.0, max(1.2, float(values.get("proximity_interaction_ratio", 2.5))))
    except (TypeError, ValueError):
        return False
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, target)
        return True
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
