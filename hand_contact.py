"""Make touching fingertips of the two drawn hands touch on screen.

Each hand is tracked separately, and the drawn fingers are never exactly as thick as
the real ones. So when the real thumbs (or fingertips) touch, the drawn ones can stop
a few pixels short of each other. When two fingertips of different hands come within a
small band, the drawn ends of those two fingers are pulled together until their
surfaces meet, fading out smoothly as the hands separate.

Works on screen-space points (the drawn hand), never on what gestures or the TLI use.
Pure Python, no Qt.
"""
from __future__ import annotations

import math

# Last bone of each digit: (joint, tip).
DISTAL = {"thumb": (3, 4), "index": (7, 8), "middle": (11, 12), "ring": (15, 16), "pinky": (19, 20)}

# Surface gaps (in palm lengths) at which two fingertips are fully snapped together,
# and beyond which they are left alone; in between the pull fades out.
SNAP_FULL = 0.15
SNAP_NONE = 0.35

# Drawn finger radius at the fingertip, in palm lengths, for each render style, and
# whether the drawn finger stops at the tip landmark ("inset") or is capped past it.
STYLE_RADII = {
    # 3D mesh (bundled model): fingertip ends exactly on the tip landmark.
    "3d": ({"thumb": 0.090, "index": 0.075, "middle": 0.075, "ring": 0.072, "pinky": 0.066}, True),
    # Flat capsules: the round cap is centred on the tip landmark. Scaled by thickness.
    "flat": ({"thumb": 0.115, "index": 0.103, "middle": 0.106, "ring": 0.100, "pinky": 0.090}, False),
}


def _closest_points(p1, q1, p2, q2):
    """Closest points between 2D segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1 = (q1[0]-p1[0], q1[1]-p1[1])
    d2 = (q2[0]-p2[0], q2[1]-p2[1])
    r = (p1[0]-p2[0], p1[1]-p2[1])
    a = d1[0]*d1[0] + d1[1]*d1[1]
    e = d2[0]*d2[0] + d2[1]*d2[1]
    f = d2[0]*r[0] + d2[1]*r[1]
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s, t = 0.0, min(1.0, max(0.0, f/e))
    else:
        c = d1[0]*r[0] + d1[1]*r[1]
        if e <= eps:
            t, s = 0.0, min(1.0, max(0.0, -c/a))
        else:
            b = d1[0]*d2[0] + d1[1]*d2[1]
            denom = a*e - b*b
            s = min(1.0, max(0.0, (b*f - c*e)/denom)) if denom > eps else 0.0
            t = (b*s + f)/e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c/a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b-c)/a))
    c1 = (p1[0]+d1[0]*s, p1[1]+d1[1]*s)
    c2 = (p2[0]+d2[0]*t, p2[1]+d2[1]*t)
    return c1, c2


def _distal_segment(points, digit, radius, inset):
    joint, tip = DISTAL[digit]
    a, b = points[joint], points[tip]
    if inset:
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            pull = min(radius, length*0.9)
            b = (b[0]-dx/length*pull, b[1]-dy/length*pull)
    return a, b


def snap_weight(gap_palms):
    if gap_palms <= SNAP_FULL:
        return 1.0
    if gap_palms >= SNAP_NONE:
        return 0.0
    t = (SNAP_NONE-gap_palms)/(SNAP_NONE-SNAP_FULL)
    return t*t*(3.0-2.0*t)


def contact_offsets(left, right, palm_left, palm_right, style="3d", thickness=1.0):
    """Per-hand {landmark index: (dx, dy)} that make nearly touching fingertips touch.

    left/right: 21 screen-space (x, y) points of each drawn hand. palm_*: each drawn
    hand's palm length in the same units. thickness scales the flat style's fingers.
    """
    out = {"left": {}, "right": {}}
    if style not in STYLE_RADII or left is None or right is None:
        return out
    if len(left) < 21 or len(right) < 21 or palm_left <= 1e-6 or palm_right <= 1e-6:
        return out
    radii, inset = STYLE_RADII[style]
    k = 1.0 if style == "3d" else max(0.2, float(thickness))
    unit = 0.5*(palm_left+palm_right)
    for dl in DISTAL:
        rl = radii[dl]*k*palm_left
        a1, b1 = _distal_segment(left, dl, rl, inset)
        for dr in DISTAL:
            rr = radii[dr]*k*palm_right
            a2, b2 = _distal_segment(right, dr, rr, inset)
            c1, c2 = _closest_points(a1, b1, a2, b2)
            dx, dy = c2[0]-c1[0], c2[1]-c1[1]
            distance = math.hypot(dx, dy)
            gap = distance - rl - rr
            if gap <= 0.0 or distance <= 1e-9:
                continue
            w = snap_weight(gap/unit)
            if w <= 0.0:
                continue
            # Each fingertip moves half the gap toward the other; its middle joint
            # follows by half that, so the finger bends a little rather than shearing.
            mx, my = dx/distance*gap*0.5*w, dy/distance*gap*0.5*w
            for side, digit, sign in (("left", dl, 1.0), ("right", dr, -1.0)):
                joint, tip = DISTAL[digit]
                for index, share in ((tip, 1.0), (joint, 0.5)):
                    ox, oy = out[side].get(index, (0.0, 0.0))
                    nx, ny = sign*mx*share, sign*my*share
                    # Several contacts on one joint: keep the largest pull per axis.
                    out[side][index] = (nx if abs(nx) > abs(ox) else ox, ny if abs(ny) > abs(oy) else oy)
    return out
