"""Tuck hidden fingers and thumb into the palm for the drawn hand.

MediaPipe always returns all 21 landmarks, including fingertips the camera can't see:
when a fingertip is hidden behind the hand (a fist seen from the back, a thumb folded
behind the fingers) the landmarks are guesses, often a half-open finger. A hidden digit
is almost always tucked in, so the drawn hand shows it tucked in.

A fingertip counts as hidden when, in the camera image, it lies over the palm or over
another digit and is farther from the camera than that part of the hand. A fingertip
outside the camera frame is not hidden in this sense (it is cut off, not tucked), so it
keeps its tracked position.

This only changes what is drawn. Gestures, the TLI and the cursor still use the
tracked landmarks. Pure Python + numpy, no Qt, so it can be unit-tested.
"""
from __future__ import annotations

import math

import numpy as np

# Landmark y is normalised by frame height and z roughly by frame width (16:9 frames).
Y_TO_X = 9.0/16.0

DIGITS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGERS = ("index", "middle", "ring", "pinky")
PALM_OUTLINE = (0, 1, 5, 9, 13, 17)

# Sizes relative to palm length (wrist to middle knuckle).
FINGER_RADIUS = 0.10          # half a finger's width
DEPTH_MARGIN = 0.06           # must be this much behind the covering part
# Behind another digit only counts for a digit that is already folded towards the
# palm: in a side-on open hand the straight fingers overlap each other in the image,
# but they are not tucked.
FOLDED_FINGER_REACH = 1.25    # fingertip within this of the wrist
# A folded thumb lies across the palm: its tip is near the middle knuckle AND over the
# fingers rather than out past the index side (across = 0 at the index knuckle, 1 at
# the pinky knuckle). Distance alone is not enough: a relaxed or pinching thumb tip is
# also within a palm length of the middle knuckle.
FOLDED_THUMB_REACH = 0.80     # thumb tip within this of the middle knuckle
FOLDED_THUMB_ACROSS = 0.0
# Once a digit is drawn tucked it stays tucked until it is clearly out again: looser
# reach limits, and several frames in a row of it showing (hysteresis). The tracker's
# guesses for fingers it can't see wander, and without this the drawn fingers flap.
STAY_FOLDED_FINGER_REACH = 1.45
STAY_FOLDED_THUMB_REACH = 0.95
STAY_FOLDED_THUMB_ACROSS = -0.15
RELEASE_FRAMES = 4            # camera frames (~130 ms) a tucked digit must be seen out
# The palm faces away from the camera when its palm-side normal points this far away.
PALM_AWAY_Z = 0.35
# Two fingertips this close are touching (a pinch), not tucked.
TOUCHING_TIPS = 0.35
# Nominal phalanx lengths (proximal, middle, distal) used to sanity-limit tracked ones.
NOMINAL_PHALANGES = {
    "index": (0.46, 0.27, 0.20),
    "middle": (0.50, 0.31, 0.22),
    "ring": (0.46, 0.29, 0.21),
    "pinky": (0.36, 0.22, 0.19),
}
# Cumulative bend of a tucked finger at the knuckle, middle and end joints (degrees).
TUCK_ANGLES = (85.0, 185.0, 245.0)
# Smoothing: how quickly the drawn digit moves to/from its tucked pose.
BLEND_RATE = 12.0             # per second (~80 ms time constant)
FORGET_AFTER = 0.5            # seconds without a hand before its state is reset


def _to_space(points):
    """(21, 3) array in an isotropic space (x units), or None for bad input."""
    try:
        if points is None or len(points) < 21:
            return None
        arr = np.array([[float(p[0]), float(p[1])*Y_TO_X, float(p[2]) if len(p) > 2 else 0.0]
                        for p in points[:21]], dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _from_space(arr, template):
    out = []
    for row, original in zip(arr, template):
        if len(original) > 2:
            out.append((float(row[0]), float(row[1])/Y_TO_X, float(row[2])))
        else:
            out.append((float(row[0]), float(row[1])/Y_TO_X))
    return out


def _normalize(v):
    n = float(np.linalg.norm(v))
    return v/n if n > 1e-12 else v*0.0


def _convex_hull(points2d):
    pts = sorted(set((float(x), float(y)) for x, y in points2d))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside(hull, x, y):
    if len(hull) < 3:
        return False
    sign = 0
    for (ax, ay), (bx, by) in zip(hull, hull[1:] + hull[:1]):
        c = (bx-ax)*(y-ay) - (by-ay)*(x-ax)
        if abs(c) < 1e-15:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _segment_hit(p, a, b):
    """(2D distance from p to segment ab, depth of the segment at the closest point)."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    length_sq = dx*dx + dy*dy
    t = 0.0 if length_sq < 1e-18 else min(1.0, max(0.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/length_sq))
    cx, cy = a[0]+dx*t, a[1]+dy*t
    return math.hypot(p[0]-cx, p[1]-cy), a[2]+(b[2]-a[2])*t


def _near_hull(hull, x, y, slack):
    """Inside the polygon, or within `slack` of its outline."""
    if _inside(hull, x, y):
        return True
    for (ax, ay), (bx, by) in zip(hull, hull[1:] + hull[:1]):
        if _segment_hit((x, y), (ax, ay, 0.0), (bx, by, 0.0))[0] <= slack:
            return True
    return False


def _dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


# Strong evidence from the tracked fingers overrides the handedness rule (a hand whose
# left/right label the tracker got wrong for a moment): the mean fold of at least two
# folded fingers, in palm lengths, pointing the other way.
LEAN_OVERRIDE = 0.25


def palm_side_normal(p, side=None):
    """Unit vector out of the palm (the side fingers fold to), or None.

    The landmarks of a left and a right hand are mirror images, so for a known hand
    the palm side follows from the knuckle order alone, not from where the tracker
    guesses hidden fingertips are. With up = wrist->middle knuckle and across =
    pinky->index knuckle, the palm is on the +(up x across) side of a physical right
    hand and the -(up x across) side of a left one. (Checked with MediaPipe's Tasks
    hand landmarker, which the app uses, on a mirrored frame; the app's left/right are
    the opposite of MediaPipe's label.)
    """
    arr = np.asarray(p, dtype=np.float64)
    up = _normalize(arr[9]-arr[0])
    across = arr[5]-arr[17]
    across = _normalize(across - up*float(across @ up))
    normal = _normalize(np.cross(up, across))
    if float(np.linalg.norm(normal)) < 0.5 or side not in ("left", "right"):
        return None
    palm_side = normal if side == "right" else -normal
    # Safety net: clearly folded fingers already show which way they bend.
    palm_len = float(np.linalg.norm(arr[9]-arr[0]))
    leans = [float((arr[DIGITS[f][-1]]-arr[DIGITS[f][0]]) @ palm_side)/palm_len
             for f in FINGERS if np.linalg.norm(arr[DIGITS[f][-1]]-arr[0]) < FOLDED_FINGER_REACH*palm_len]
    if len(leans) >= 2 and sum(leans)/len(leans) < -LEAN_OVERRIDE:
        palm_side = -palm_side
    return palm_side


def thumb_across(p):
    """Thumb tip across the palm: 0 at the index knuckle, 1 at the pinky knuckle,
    negative out past the index side."""
    ax = [p[17][k]-p[5][k] for k in range(3)]
    width_sq = sum(c*c for c in ax)
    if width_sq < 1e-12:
        return -1.0
    return sum((p[4][k]-p[5][k])*ax[k] for k in range(3))/width_sq


def hidden_digits(points, side=None, tucked=frozenset()):
    """{digit: True/False}: which fingertips/thumb tip the camera can't see.

    side: "left"/"right" (the physical hand) enables the palm-facing-away rule.
    tucked: digits currently drawn tucked, which use looser limits to stay tucked.
    """
    arr = _to_space(points)
    result = {name: False for name in DIGITS}
    if arr is None:
        return result
    p = arr.tolist()   # plain floats: this runs every frame, and numpy is slow on 3-vectors
    palm_len = _dist3(p[9], p[0])
    if palm_len < 1e-6:
        return result
    margin = DEPTH_MARGIN*palm_len
    radius = FINGER_RADIUS*palm_len

    # Palm plane through wrist, index knuckle and pinky knuckle: depth at any (x, y).
    u = [p[5][k]-p[0][k] for k in range(3)]
    v = [p[17][k]-p[0][k] for k in range(3)]
    normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    length = math.sqrt(sum(c*c for c in normal)) or 1.0
    normal = tuple(c/length for c in normal)
    palm_faces_camera_enough = abs(normal[2]) > 0.25   # edge-on palm covers almost nothing
    hull = _convex_hull([(p[i][0], p[i][1]) for i in PALM_OUTLINE]) if palm_faces_camera_enough else []

    def palm_depth(x, y):
        return p[0][2] - (normal[0]*(x-p[0][0]) + normal[1]*(y-p[0][1]))/normal[2]

    # With the palm turned away from the camera, folded fingers are behind the hand.
    palm_side = palm_side_normal(arr, side)
    palm_away = palm_side is not None and float(palm_side[2]) > PALM_AWAY_Z

    folded_by_name = {}
    for name, chain in DIGITS.items():
        tip = p[chain[-1]]
        raw = points[chain[-1]]
        if not (0.0 <= float(raw[0]) <= 1.0 and 0.0 <= float(raw[1]) <= 1.0):
            continue  # cut off by the frame edge, not tucked
        staying = name in tucked
        if name == "thumb":
            reach = STAY_FOLDED_THUMB_REACH if staying else FOLDED_THUMB_REACH
            min_across = STAY_FOLDED_THUMB_ACROSS if staying else FOLDED_THUMB_ACROSS
            folded = _dist3(tip, p[9]) < reach*palm_len and thumb_across(p) >= min_across
        else:
            reach = STAY_FOLDED_FINGER_REACH if staying else FOLDED_FINGER_REACH
            folded = _dist3(tip, p[0]) < reach*palm_len
        folded_by_name[name] = folded
        # Thumb and a fingertip touching is a pinch or an "OK" ring, not a tucked digit.
        if name == "thumb":
            touching_thumb = any(_dist3(tip, p[DIGITS[f][-1]]) < TOUCHING_TIPS*palm_len for f in FINGERS)
        else:
            touching_thumb = _dist3(tip, p[4]) < TOUCHING_TIPS*palm_len
        hidden = folded and palm_away and not touching_thumb
        if not hidden:
            hidden = (palm_faces_camera_enough and tip[2] > palm_depth(tip[0], tip[1]) + margin and
                      _near_hull(hull, tip[0], tip[1], radius))
        if not hidden:
            for other, other_chain in DIGITS.items():
                if not folded:
                    break
                if other == name:
                    # Curled right over, a fingertip sits behind its own first bone.
                    segments = [other_chain[:2]]
                elif _dist3(tip, p[other_chain[-1]]) < TOUCHING_TIPS*palm_len:
                    continue  # fingertips touching, e.g. a pinch
                else:
                    segments = list(zip(other_chain, other_chain[1:]))
                for a, b in segments:
                    distance, depth = _segment_hit(tip, p[a], p[b])
                    if distance <= radius and tip[2] > depth + margin:
                        hidden = True
                        break
                if hidden:
                    break
        result[name] = hidden
    # A fist: once one finger is hidden, the other folded fingers are tucked with it,
    # rather than each following the tracker's wandering guess on its own.
    if any(result[f] for f in FINGERS):
        for f in FINGERS:
            if folded_by_name.get(f):
                result[f] = True
    return result


def tucked_landmarks(points, digits, side=None):
    """The landmarks with each digit in `digits` folded into a fist position.

    The palm, wrist and thumb base keep their tracked positions.
    """
    p = _to_space(points)
    if p is None or not digits:
        return list(points) if points is not None else points
    out = p.copy()
    palm_len = float(np.linalg.norm(p[9]-p[0]))
    if palm_len < 1e-6:
        return list(points)
    up = _normalize(p[9]-p[0])
    across = p[5]-p[17]
    across = _normalize(across - up*float(across @ up))
    normal = _normalize(np.cross(up, across))
    # Which side of the palm the fingers fold to. For a known hand this is fixed by its
    # handedness; otherwise the side the tracked (guessed) hidden tips are on, or away
    # from the camera, since a tip hidden by the palm is behind it.
    palm_side = palm_side_normal(p, side)
    if palm_side is None:
        lean = sum(float((p[DIGITS[d][-1]]-p[DIGITS[d][0]]) @ normal) for d in digits)
        if abs(lean) > 0.05*palm_len:
            palm_side = normal if lean > 0 else -normal
        else:
            palm_side = normal if normal[2] >= 0 else -normal

    for name in FINGERS:
        if name not in digits:
            continue
        mcp, pip, dip, tip = DIGITS[name]
        base = p[mcp]-p[0]
        finger_up = _normalize(base - palm_side*float(base @ palm_side))
        nominal = NOMINAL_PHALANGES[name]
        tracked = (np.linalg.norm(p[pip]-p[mcp]), np.linalg.norm(p[dip]-p[pip]), np.linalg.norm(p[tip]-p[dip]))
        position = p[mcp].copy()
        for joint, angle, n_len, t_len in zip((pip, dip, tip), TUCK_ANGLES, nominal, tracked):
            length = min(1.3*n_len*palm_len, max(0.7*n_len*palm_len, float(t_len)))
            theta = math.radians(angle)
            position = position + (math.cos(theta)*finger_up + math.sin(theta)*palm_side)*length
            out[joint] = position

    if "thumb" in digits:
        # Folded over the front of the first two curled fingers (index and middle in a
        # fist, middle and ring when pointing), on their middle bones.
        curled = [f for f in ("index", "middle", "ring")
                  if f in digits or np.linalg.norm(p[DIGITS[f][-1]]-p[0]) < FOLDED_FINGER_REACH*palm_len]
        pair = (curled + ["index", "middle"])[:2] if len(curled) < 2 else curled[:2]
        rest = sum(out[DIGITS[f][1]]+out[DIGITS[f][2]] for f in pair)*0.25
        target_tip = rest + palm_side*0.10*palm_len
        out[4] = target_tip
        out[3] = p[2] + (target_tip-p[2])*0.55 + palm_side*0.06*palm_len
    return _from_space(out, points)


class TuckSmoother:
    """Per-hand fade between tracked and tucked digits, so tucking never pops."""

    def __init__(self):
        self.weights = {}
        self.last_seen = {}
        self.tucked = {}     # side -> digits currently tucked (after hysteresis)
        self.seen_out = {}   # side -> {digit: frames in a row it has been seen out}
        self._last_input = {}  # side -> last camera frame's points (count each frame once)
        self._cache = {}     # side -> (input points, output) once the fade has settled

    def update(self, side, points, now):
        last = self.last_seen.get(side)
        fresh = last is None or now-last > FORGET_AFTER
        cached = self._cache.get(side)
        if not fresh and cached is not None and cached[0] == points:
            # The screen refreshes faster than the camera: same frame, same answer.
            self.last_seen[side] = now
            return cached[1]
        self._cache.pop(side, None)
        p = _to_space(points)
        if p is None:
            return points
        if fresh:
            self.tucked[side] = set()
            self.seen_out[side] = {}
        tucked = self.tucked.setdefault(side, set())
        seen_out = self.seen_out.setdefault(side, {})
        hand = side if side in ("left", "right") else None
        new_frame = fresh or self._last_input.get(side) != points
        self._last_input[side] = list(points)
        hidden = hidden_digits(points, hand, frozenset(tucked)) if new_frame else None
        for name in (DIGITS if new_frame else ()):
            if hidden[name]:
                tucked.add(name)
                seen_out[name] = 0
            elif name in tucked:
                seen_out[name] = seen_out.get(name, 0) + 1
                if seen_out[name] >= RELEASE_FRAMES:
                    tucked.discard(name)
                    seen_out[name] = 0
        self.last_seen[side] = now
        weights = self.weights.setdefault(side, {name: 0.0 for name in DIGITS})
        step = 1.0 if fresh else min(1.0, max(0.0, (now-last)*BLEND_RATE))
        for name in DIGITS:
            target = 1.0 if name in tucked else 0.0
            weights[name] += (target-weights[name])*step
            if abs(weights[name]-target) < 0.01:
                weights[name] = target
        settled = all(w in (0.0, 1.0) for w in weights.values())
        active = [name for name in DIGITS if weights[name] > 0.0]
        if not active:
            result = points
        else:
            folded = _to_space(tucked_landmarks(points, set(active), hand))
            blended = p.copy()
            for name in active:
                w = weights[name]
                joints = DIGITS[name][1:] if name != "thumb" else (3, 4)
                for j in joints:
                    blended[j] = p[j]*(1.0-w) + folded[j]*w
            result = _from_space(blended, points)
        if settled and not any(seen_out.values()):
            self._cache[side] = (list(points), result)
        return result

    def forget(self, side):
        self.weights.pop(side, None)
        self.last_seen.pop(side, None)
        self.tucked.pop(side, None)
        self.seen_out.pop(side, None)
        self._last_input.pop(side, None)
        self._cache.pop(side, None)
