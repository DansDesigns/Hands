"""Recover the full X11 screen rectangle when Qt reports only a Qtile work area.

Qt's QScreen.geometry() is normally the physical screen rectangle, but some Qtile/X11
setups expose the work-area height there as well. X11's root dimensions are independent
of EWMH panel struts, so they can recover the hidden bottom strip when the widths prove
that Qt and X11 are using the same coordinate scale.
"""
from __future__ import annotations

from typing import Sequence


def resolve_fullscreen_bounds(qt_bounds: Sequence[int], root_size: Sequence[int],
                              max_reserved_pixels: int = 200):
    """Return ``((x, y, width, height), native_warp_safe)``.

    Only repairs a likely work-area-height truncation. If Qt and X11 widths differ (as
    they do under most high-DPI scaling or multi-monitor root geometries), the Qt bounds
    are retained and raw root-coordinate warping must not be used.
    """
    try:
        qx, qy, qw, qh = (int(v) for v in qt_bounds)
        nw, nh = (int(v) for v in root_size)
    except (TypeError, ValueError):
        return tuple(qt_bounds), False
    if min(qw, qh, nw, nh) <= 0:
        return (qx, qy, qw, qh), False

    width_tolerance = max(4, round(qw * 0.01))
    delta_h = nh - qh
    if abs(nw - qw) > width_tolerance:
        return (qx, qy, qw, qh), False
    if delta_h < 0 or delta_h > max_reserved_pixels:
        return (qx, qy, qw, qh), False
    # A single X11 root/display with matching width uses (0, 0) as the origin. A
    # nonzero QScreen origin is allowed only when it is plausibly the top work-area
    # offset; a side-by-side monitor layout is rejected by the width guard above.
    if abs(qx) > width_tolerance or qy < -2 or qy > delta_h + 2:
        return (qx, qy, qw, qh), False
    return (0, 0, nw, nh), True
