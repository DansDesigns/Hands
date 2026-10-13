"""Palm-relative TLI calibration helpers.

Calibration stores the thumb/index pinch midpoint in a hand-local coordinate frame.
The stored coordinates are normalized by palm length/width, so the point follows
translation, rotation, and apparent hand-size changes without using screen coordinates.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _xyz(point: Any) -> tuple[float, float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y), float(getattr(point, "z", 0.0))
    values = tuple(point)
    return float(values[0]), float(values[1]), float(values[2] if len(values) > 2 else 0.0)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, factor):
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(max(0.0, _dot(a, a)))


def _unit(a):
    length = _norm(a)
    if not math.isfinite(length) or length < 1e-7:
        return None
    return _mul(a, 1.0 / length)


def _hand_frame(landmarks):
    """Return origin, orthogonal palm axes and scale units, or None for bad landmarks."""
    if landmarks is None or len(landmarks) < 21:
        return None
    try:
        points = [_xyz(point) for point in landmarks]
        if not all(math.isfinite(c) for point in points for c in point):
            return None
        origin = points[0]
        palm_long = _sub(points[9], origin)  # wrist -> middle-finger MCP
        length_scale = _norm(palm_long)
        axis_long = _unit(palm_long)
        index_to_pinky = _sub(points[17], points[5])  # transverse palm direction
        width_scale = _norm(index_to_pinky)
        if axis_long is None or length_scale < 1e-6 or width_scale < 1e-6:
            return None
        lateral_orthogonal = _sub(index_to_pinky, _mul(axis_long, _dot(index_to_pinky, axis_long)))
        axis_lateral = _unit(lateral_orthogonal)
        if axis_lateral is None:
            return None
        axis_normal = _unit(_cross(axis_long, axis_lateral))
        if axis_normal is None:
            return None
        # Recompute lateral for a fully orthonormal right-handed frame.
        axis_lateral = _unit(_cross(axis_normal, axis_long))
        if axis_lateral is None:
            return None
        return origin, axis_long, axis_lateral, axis_normal, length_scale, width_scale, points
    except (TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return None


def encode_relative_tli(landmarks) -> dict[str, float] | None:
    """Encode the current thumb/index midpoint in normalized palm-local coordinates."""
    frame = _hand_frame(landmarks)
    if frame is None:
        return None
    origin, axis_long, axis_lateral, axis_normal, length_scale, width_scale, points = frame
    pinch_midpoint = _mul(_add(points[4], points[8]), 0.5)
    delta = _sub(pinch_midpoint, origin)
    result = {
        "forward": _dot(delta, axis_long) / length_scale,
        "lateral": _dot(delta, axis_lateral) / width_scale,
        "depth": _dot(delta, axis_normal) / width_scale,
    }
    if not all(math.isfinite(value) and abs(value) <= 4.0 for value in result.values()):
        return None
    return result


def resolve_relative_tli(landmarks, calibration: Mapping[str, Any] | None):
    """Reconstruct a calibrated TLI point in current MediaPipe normalized x/y/z space."""
    if not isinstance(calibration, Mapping):
        return None
    try:
        forward = float(calibration["forward"])
        lateral = float(calibration["lateral"])
        depth = float(calibration["depth"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and abs(v) <= 4.0 for v in (forward, lateral, depth)):
        return None
    frame = _hand_frame(landmarks)
    if frame is None:
        return None
    origin, axis_long, axis_lateral, axis_normal, length_scale, width_scale, _points = frame
    result = _add(
        origin,
        _add(
            _mul(axis_long, forward * length_scale),
            _add(_mul(axis_lateral, lateral * width_scale),
                 _mul(axis_normal, depth * width_scale)),
        ),
    )
    return result if all(math.isfinite(v) for v in result) else None


def mean_relative_tli(samples: Iterable[Mapping[str, Any]]) -> dict[str, float] | None:
    """Average stable calibration samples, rejecting incomplete/non-finite sample sets."""
    rows = []
    for sample in samples:
        try:
            row = tuple(float(sample[key]) for key in ("forward", "lateral", "depth"))
        except (TypeError, ValueError, KeyError):
            return None
        if not all(math.isfinite(v) and abs(v) <= 4.0 for v in row):
            return None
        rows.append(row)
    if not rows:
        return None
    return {
        key: sum(row[i] for row in rows) / len(rows)
        for i, key in enumerate(("forward", "lateral", "depth"))
    }
