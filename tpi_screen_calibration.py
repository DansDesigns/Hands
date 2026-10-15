"""Nine-point, per-hand screen calibration for the Tap Point Indicator (TPI).

The calibration records where the raw TPI landed when the user deliberately placed it
on a screen target. At runtime a piecewise-affine 3x3 mesh corrects position errors while
keeping left and right hands independent. Stored coordinates are screen-normalized, so
calibration survives a display resolution change.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

# Row-major target order: four corners, top/bottom/left/right midpoints, centre.
TARGET_NAMES = (
    "TOP LEFT", "TOP EDGE", "TOP RIGHT",
    "LEFT EDGE", "CENTRE", "RIGHT EDGE",
    "BOTTOM LEFT", "BOTTOM EDGE", "BOTTOM RIGHT",
)
TARGET_FRACTIONS = (
    (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
    (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
    (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
)
TARGET_COUNT = 9


def target_pixel_positions(width: int | float, height: int | float) -> list[tuple[float, float]]:
    """Return all nine targets in screen-local physical pixel coordinates."""
    w, h = max(1.0, float(width) - 1.0), max(1.0, float(height) - 1.0)
    return [(x * w, y * h) for x, y in TARGET_FRACTIONS]


def calibration_from_samples(samples: Sequence[Any], width: int | float,
                             height: int | float) -> dict[str, Any] | None:
    """Normalize nine observed pixel positions for use as the source calibration mesh."""
    if len(samples) != TARGET_COUNT:
        return None
    w, h = max(1.0, float(width) - 1.0), max(1.0, float(height) - 1.0)
    observed = []
    for point in samples:
        try:
            if isinstance(point, Mapping):
                x, y = float(point["x"]), float(point["y"])
            else:
                x, y = float(point[0]), float(point[1])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        observed.append({"x": min(1.0, max(0.0, x / w)),
                         "y": min(1.0, max(0.0, y / h))})
    if not validate_calibration({"version": 1, "observed": observed}):
        return None
    return {"version": 1, "observed": observed}


def validate_calibration(value: Any) -> bool:
    """Reject incomplete, non-finite, or collapsed 3x3 calibration meshes."""
    if not isinstance(value, Mapping) or not isinstance(value.get("observed"), list):
        return False
    raw = value["observed"]
    if len(raw) != TARGET_COUNT:
        return False
    points = []
    for p in raw:
        try:
            x, y = float(p["x"]), float(p["y"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(x) or not math.isfinite(y) or not (-0.1 <= x <= 1.1 and -0.1 <= y <= 1.1):
            return False
        points.append((x, y))
    # Every neighbouring cell needs non-collapsed triangles for interpolation.
    for row in range(2):
        for col in range(2):
            a = row * 3 + col
            b, c, d = a + 1, a + 3, a + 4
            # Reject flipped/folded cells as well as collapsed triangles. All target
            # triangles use the same screen-coordinate orientation (y increases down).
            if _area2(points[a], points[b], points[d]) <= 1e-5:
                return False
            if _area2(points[a], points[d], points[c]) <= 1e-5:
                return False
    return True


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _barycentric(point, a, b, c):
    den = _area2(a, b, c)
    if not math.isfinite(den) or abs(den) < 1e-8:
        return None
    w0 = _area2(point, b, c) / den
    w1 = _area2(a, point, c) / den
    w2 = 1.0 - w0 - w1
    return (w0, w1, w2)


def map_point(point: Any, calibration: Mapping[str, Any] | None,
              width: int | float, height: int | float) -> tuple[float, float] | None:
    """Map one raw pixel point through the 3x3 piecewise-affine calibration mesh.

    Outside the sampled mesh, the nearest triangle is extrapolated, then clamped to the
    screen. Adjacent triangles share the same registered vertices, avoiding jumps at edges.
    """
    if not validate_calibration(calibration):
        return None
    try:
        px, py = (float(point.x()), float(point.y())) if hasattr(point, "x") else (float(point[0]), float(point[1]))
        w, h = max(1.0, float(width) - 1.0), max(1.0, float(height) - 1.0)
    except (TypeError, ValueError, IndexError, AttributeError):
        return None
    if not all(math.isfinite(v) for v in (px, py, w, h)):
        return None
    source = [(float(p["x"]), float(p["y"])) for p in calibration["observed"]]
    target = list(TARGET_FRACTIONS)
    normalized_point = (px / w, py / h)
    triangles = []
    for row in range(2):
        for col in range(2):
            a = row * 3 + col
            b, c, d = a + 1, a + 3, a + 4
            triangles.extend(((a, b, d), (a, d, c)))
    best = None
    best_penalty = float("inf")
    for tri in triangles:
        weights = _barycentric(normalized_point, *(source[i] for i in tri))
        if weights is None:
            continue
        penalty = sum(max(0.0, -v, v - 1.0) for v in weights)
        if penalty < best_penalty:
            best, best_penalty = (tri, weights), penalty
        if penalty <= 1e-5:
            best = (tri, weights)
            break
    if best is None:
        return None
    tri, weights = best
    out_x = sum(weights[i] * target[idx][0] for i, idx in enumerate(tri))
    out_y = sum(weights[i] * target[idx][1] for i, idx in enumerate(tri))
    if not math.isfinite(out_x) or not math.isfinite(out_y):
        return None
    return (min(w, max(0.0, out_x * w)), min(h, max(0.0, out_y * h)))


def load_calibrations(path: str | Path) -> dict[str, dict[str, Any] | None]:
    """Load valid left/right calibration meshes; ignore a corrupt side independently."""
    result = {"left": None, "right": None}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return result
    if not isinstance(data, Mapping):
        return result
    for side in result:
        value = data.get(side)
        if validate_calibration(value):
            result[side] = {"version": 1,
                            "observed": [{"x": float(p["x"]), "y": float(p["y"])}
                                         for p in value["observed"]]}
    return result


def save_calibrations(path: str | Path, calibrations: Mapping[str, Any]) -> bool:
    """Atomically save independently validated per-hand calibration meshes."""
    clean = {"left": None, "right": None}
    for side in clean:
        value = calibrations.get(side) if isinstance(calibrations, Mapping) else None
        if validate_calibration(value):
            clean[side] = {"version": 1,
                           "observed": [{"x": float(p["x"]), "y": float(p["y"])}
                                        for p in value["observed"]]}
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return True
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False
