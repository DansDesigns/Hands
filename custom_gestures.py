"""User-recorded custom hand gestures for Spatial Desktop.

A recorded pose is a small feature vector describing how bent each finger is, measured
in 3D so it does not depend on where the hand is, how big it looks, or (mostly) which
way it is turned. Matching compares live features with each recorded template.

It also records how far apart neighbouring fingers are (measured in the palm plane), so
"split" poses such as the Vulcan salute (index+middle together, ring+pinky together,
a gap between middle and ring) are told apart from an ordinary open hand.

Overlapping both TLIs is a built-in glass-menu toggle (handled in main.py), so it is
not a recordable gesture; tli_overlap() below is the shared overlap test.

Everything here is pure Python with no Qt dependency so it can be unit-tested.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Camera frames are 640x360; MediaPipe x/y are normalised per axis and z uses the
# x scale, so convert to a common pixel-like space before measuring angles.
_FRAME_W, _FRAME_H = 640.0, 360.0

FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_SHORT = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}
FEATURE_COUNT = 15
LEGACY_FEATURE_COUNT = 10
PRE_SPREAD_FEATURE_COUNT = 12
# Finger spread: angle between neighbouring fingers in the palm plane, in units of
# SPREAD_DEGREES. -1 means unknown (poses recorded before spread was measured).
SPREAD_DEGREES = 20.0
SPREAD_UNKNOWN = -1.0
SPREAD_APART = 0.6       # >= 12 degrees: a deliberate gap between two fingers
SPREAD_TOGETHER = 0.4    # <= 8 degrees: two fingers held together
SPREAD_MISMATCH = 0.5    # a gap counts as "not apart" for matching below 10 degrees
SPREAD_GATE = 0.5        # split poses: each inner gap must agree within 10 degrees
# Spread values 12-14 are I-M, M-R, R-P. Only the two inner gaps define a split: the
# pinky drifts out on its own in a relaxed hand, so it can't carry a deliberate gap.
SPREAD_GAPS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
SPLIT_GAPS = (0, 1)
MAX_GESTURES = 6
SIDES = ("left", "right", "either")
# With the circle timer off a gesture still needs this brief stable hold, so a single
# noisy tracking frame passing through a pose can't fire it.
INSTANT_HOLD_SECONDS = 0.12

# Curated actions, cycled with the arrow buttons on each gesture row. Anything not
# handled internally is passed to SpatialDesktop.execute_pc_voice_command.
ACTIONS = (
    ("none", "NO ACTION"),
    ("custom_key", "CUSTOM KEY"),
    ("toggle_glass_menu", "GLASS MENU"),
    ("open_search", "SEARCH"),
    ("media_play_pause", "PLAY / PAUSE"),
    ("media_next", "NEXT TRACK"),
    ("media_previous", "PREVIOUS TRACK"),
    ("volume_up", "VOLUME UP"),
    ("volume_down", "VOLUME DOWN"),
    ("volume_mute", "MUTE"),
    ("mouse_right_click", "RIGHT CLICK"),
    ("mouse_middle_click", "MIDDLE CLICK"),
    ("mouse_back", "MOUSE BACK"),
    ("mouse_forward", "MOUSE FORWARD"),
    ("copy", "COPY"),
    ("paste", "PASTE"),
    ("paste_here", "PASTE AT TLI"),
    ("undo", "UNDO"),
    ("redo", "REDO"),
    ("select_all", "SELECT ALL"),
    ("switch_window", "SWITCH WINDOW"),
    ("show_desktop", "SHOW DESKTOP"),
    ("minimize_window", "MINIMISE WINDOW"),
    ("maximize_window", "MAXIMISE WINDOW"),
    ("fullscreen_toggle", "FULLSCREEN"),
    ("snap_left", "SNAP LEFT"),
    ("snap_right", "SNAP RIGHT"),
    ("new_tab", "NEW TAB"),
    ("close_tab", "CLOSE TAB"),
    ("go_back", "PAGE BACK"),
    ("go_forward", "PAGE FORWARD"),
    ("refresh_window", "REFRESH"),
    ("open_file_explorer", "FILE EXPLORER"),
)
ACTION_IDS = tuple(a for a, _ in ACTIONS)
# Columns and headings for the Gestures-page action dropdown. Every ACTIONS entry
# appears exactly once (checked by tests).
ACTION_GROUPS = (
    (("GENERAL", ("none", "custom_key", "toggle_glass_menu", "open_search")),
     ("MEDIA", ("media_play_pause", "media_next", "media_previous",
                "volume_up", "volume_down", "volume_mute"))),
    (("MOUSE", ("mouse_right_click", "mouse_middle_click", "mouse_back", "mouse_forward")),
     ("EDIT", ("copy", "paste", "paste_here", "undo", "redo", "select_all"))),
    (("WINDOWS", ("switch_window", "show_desktop", "minimize_window", "maximize_window",
                  "fullscreen_toggle", "snap_left", "snap_right")),),
    (("BROWSER & FILES", ("new_tab", "close_tab", "go_back", "go_forward",
                          "refresh_window", "open_file_explorer")),),
)
ACTION_LABELS = dict(ACTIONS)


def _pt(lm):
    if isinstance(lm, (tuple, list)):
        x, y = float(lm[0]), float(lm[1])
        z = float(lm[2]) if len(lm) > 2 else 0.0
    else:
        x, y, z = float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))
    return (x*_FRAME_W, y*_FRAME_H, z*_FRAME_W)


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _dist(a, b):
    return _norm(_sub(a, b))


def _angle(u, v):
    nu, nv = _norm(u), _norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) / (nu*nv)
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def hand_features(landmarks):
    """Return a rotation/scale-tolerant feature vector, or None for bad input.

    Layout (12 values):
      0-3  bend of index/middle/ring/pinky: total joint bend in degrees / 180
      4-7  reach of index..pinky: (wrist->tip / wrist->knuckle) - 1
      8    thumb out: thumb tip to middle knuckle / palm size
      9    thumb bend: bend at the thumb's two outer joints / 90
      10   thumb across palm: + over the fingers, - out past the index side (palm widths)
      11   thumb screen direction: -1 pointing up, +1 pointing down. Only compared when
           both poses have the thumb out, so it separates thumbs up from thumbs down
           without making every other pose depend on hand rotation.
      12-14 spread between index-middle, middle-ring, ring-pinky: the angle between the
           knuckle-to-tip directions after projecting them onto the palm plane, in
           SPREAD_DEGREES. Not part of the distance; see split_gate().
    """
    try:
        if landmarks is None or len(landmarks) < 21:
            return None
        p = [_pt(landmarks[i]) for i in range(21)]
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    palm = math.sqrt(max(0.0, _dist(p[0], p[9]) * _dist(p[5], p[17])))
    if not math.isfinite(palm) or palm < 1e-6:
        return None
    bends, reaches = [], []
    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip, tip = FINGERS[name]
        bend = (_angle(_sub(p[mcp], p[0]), _sub(p[pip], p[mcp])) +
                _angle(_sub(p[pip], p[mcp]), _sub(p[dip], p[pip])) +
                _angle(_sub(p[dip], p[pip]), _sub(p[tip], p[dip])))
        bends.append(min(1.6, bend/180.0))
        knuckle = max(1e-6, _dist(p[0], p[mcp]))
        reaches.append(min(1.6, max(-0.5, _dist(p[0], p[tip])/knuckle - 1.0)))
    thumb_out = min(2.0, _dist(p[4], p[9]) / palm)
    thumb_bend = (_angle(_sub(p[2], p[1]), _sub(p[3], p[2])) +
                  _angle(_sub(p[3], p[2]), _sub(p[4], p[3]))) / 90.0
    across_vec = _sub(p[17], p[5])
    width_sq = max(1e-9, across_vec[0]**2 + across_vec[1]**2 + across_vec[2]**2)
    off = _sub(p[4], p[5])
    thumb_across = (off[0]*across_vec[0] + off[1]*across_vec[1] + off[2]*across_vec[2]) / width_sq
    tx, ty = p[4][0]-p[2][0], p[4][1]-p[2][1]
    thumb_vertical = ty / max(1e-6, math.hypot(tx, ty))
    values = bends + reaches + [thumb_out, min(2.0, thumb_bend),
                                min(1.5, max(-1.5, thumb_across)), thumb_vertical]
    values += _finger_spreads(p)
    return values if all(math.isfinite(v) for v in values) else None


def _cross(u, v):
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])


def _finger_spreads(p):
    """Angles between neighbouring fingers in the palm plane (SPREAD_DEGREES units).

    Projecting onto the palm plane drops the noisy depth part of each fingertip, so the
    value follows the sideways gap between fingers even when the hand is turned.
    """
    normal = _cross(_sub(p[5], p[0]), _sub(p[17], p[0]))
    length = _norm(normal)
    if length < 1e-6:
        return [SPREAD_UNKNOWN]*len(SPREAD_GAPS)
    normal = (normal[0]/length, normal[1]/length, normal[2]/length)

    def in_plane(name):
        mcp, _pip, _dip, tip = FINGERS[name]
        d = _sub(p[tip], p[mcp])
        k = d[0]*normal[0] + d[1]*normal[1] + d[2]*normal[2]
        return (d[0]-k*normal[0], d[1]-k*normal[1], d[2]-k*normal[2])

    return [min(2.0, _angle(in_plane(a), in_plane(b)) / SPREAD_DEGREES) for a, b in SPREAD_GAPS]


# Matches gesture_rules.THUMB_OUT_ACROSS_PALM, which decides whether a closed hand is a fist.
THUMB_OUT_ACROSS = -0.35


def thumb_is_out(features):
    return bool(features) and len(features) >= FEATURE_COUNT and (features[8] > 0.95 or features[10] < THUMB_OUT_ACROSS)


def extended_fingers(features):
    """Best-guess set of extended fingers, used for labels and conflict checks."""
    if not features or len(features) < FEATURE_COUNT:
        return set()
    out = set()
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        if features[i] < 0.42 and features[4+i] > 0.55:
            out.add(name)
    if features[8] > 0.95 or features[10] < THUMB_OUT_ACROSS:
        out.add("thumb")
    return out


def _spread_gaps(features, ext):
    """{gap index: spread} for inner gaps whose two fingers are both extended."""
    if not features or len(features) < FEATURE_COUNT:
        return {}
    out = {}
    for i in SPLIT_GAPS:
        a, b = SPREAD_GAPS[i]
        value = features[12+i]
        if a in ext and b in ext and value >= 0.0:
            out[i] = value
    return out


def is_split(features):
    """A deliberate split: one inner finger gap held apart while another is held together."""
    gaps = _spread_gaps(features, extended_fingers(features))
    return (any(v >= SPREAD_APART for v in gaps.values()) and
            any(v <= SPREAD_TOGETHER for v in gaps.values()))


def is_vulcan(features):
    """Vulcan salute: all four fingers out, index+middle together, gap before the ring."""
    ext = extended_fingers(features)
    gaps = _spread_gaps(features, ext)
    return ({"index", "middle", "ring", "pinky"} <= ext and len(gaps) == 2 and
            gaps[1] >= SPREAD_APART and gaps[0] <= SPREAD_TOGETHER)


def split_gate(a, b):
    """False when a split pose is compared with a pose whose finger gaps don't match.

    Only applies when either pose is a split, so ordinary poses match exactly as
    before however the fingers happen to be spread.
    """
    if not (is_split(a) or is_split(b)):
        return True
    gaps_a = _spread_gaps(a, extended_fingers(a))
    gaps_b = _spread_gaps(b, extended_fingers(b))
    for i in set(gaps_a) & set(gaps_b):
        x, y = gaps_a[i], gaps_b[i]
        if abs(x-y) > SPREAD_GATE:
            return False
        if max(x, y) >= SPREAD_APART and min(x, y) < SPREAD_MISMATCH:
            return False
    return True


def describe(features):
    ext = extended_fingers(features)
    if not ext:
        return "FIST"
    if is_vulcan(features):
        return "VULCAN"
    if is_split(features):
        gaps = _spread_gaps(features, ext)
        names = [f for f in FINGER_ORDER if f in ext]
        label = FINGER_SHORT[names[0]]
        for prev, name in zip(names, names[1:]):
            gap = next((i for i, pair in enumerate(SPREAD_GAPS) if pair == (prev, name)), None)
            label += ("|" if gaps.get(gap, 0.0) >= SPREAD_APART else "·") + FINGER_SHORT[name]
        return label
    if len(ext) == 5:
        return "OPEN HAND"
    if ext == {"thumb"}:
        if features[11] < -0.5:
            return "THUMBS UP"
        if features[11] > 0.5:
            return "THUMBS DOWN"
    return "·".join(FINGER_SHORT[f] for f in FINGER_ORDER if f in ext)


def builtin_conflict(features, pinch_ratio=None):
    """Return a message if the pose collides with a built-in gesture, else None."""
    ext = extended_fingers(features)
    if pinch_ratio is not None and pinch_ratio < 0.38:
        return "That looks like a pinch, which is already a built-in gesture."
    if not ext:
        return "That looks like a fist, which is already a built-in gesture."
    if len(ext) == 5 and not is_split(features):
        return "That's an open hand, the pointing pose. Split your fingers (like a Vulcan salute) to use it."
    if ext == {"index", "middle"} or ext == {"index", "middle", "thumb"}:
        return "That looks like the peace sign, which already toggles tracking."
    return None


# Returned by pinch_ratio when the thumb is pinching a finger other than the index.
NOT_AN_INDEX_PINCH = 99.0


def pinch_ratio(landmarks):
    """Thumb-tip to index-tip distance divided by palm size.

    When the thumb tip is nearer the middle, ring or pinky tip than the index tip,
    the pose is not the built-in index pinch, so a large ratio is returned and the
    pose stays available for custom gestures.
    """
    try:
        p = [_pt(landmarks[i]) for i in (0, 4, 5, 8, 9, 17)]
        other_tips = [_pt(landmarks[i]) for i in (12, 16, 20)]
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    palm = math.sqrt(max(0.0, _dist(p[0], p[4]) * _dist(p[2], p[5])))
    if palm <= 1e-6:
        return None
    to_index = _dist(p[1], p[3])
    if any(_dist(p[1], tip) <= to_index for tip in other_tips):
        return NOT_AN_INDEX_PINCH
    return to_index / palm


# The thumb is described by 2 of the 10 values but is often the only difference
# between poses (4 fingers vs open hand, pointing vs an L-shape), so weight it up.
# Finger spread (12-14) is not part of the distance: split_gate() handles it, so
# ordinary poses keep matching the same way however the fingers are spread.
FEATURE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.5, 2.0, 3.0, 0.0, 0.0, 0.0)


def feature_distance(a, b):
    """Weighted root-mean-square difference between two feature vectors.

    The thumb direction (index 11) only counts when both poses have the thumb out.
    """
    if not a or not b or len(a) != len(b):
        return math.inf
    weights = list(FEATURE_WEIGHTS[:len(a)])
    if len(a) >= FEATURE_COUNT and not (thumb_is_out(a) and thumb_is_out(b)):
        weights[11] = 0.0
    total = sum(w*(x-y)**2 for w, x, y in zip(weights, a, b))
    return math.sqrt(total / max(1e-9, sum(weights)))


def mean_features(samples):
    clean = [s for s in samples if s and len(s) == FEATURE_COUNT]
    if not clean:
        return None
    return [sum(s[i] for s in clean)/len(clean) for i in range(FEATURE_COUNT)]


def best_pose_match(features, gestures, side, tolerance):
    """Index of the closest recorded pose for this hand within tolerance, else None."""
    if features is None:
        return None
    best, best_d = None, math.inf
    for index, gesture in enumerate(gestures):
        if gesture.get("kind") != "pose":
            continue
        if gesture.get("side", "either") not in ("either", side):
            continue
        if not split_gate(features, gesture.get("features")):
            continue
        d = feature_distance(features, gesture.get("features"))
        if d < best_d:
            best, best_d = index, d
    return best if best_d <= float(tolerance) else None


class HoldTrigger:
    """Fire once when the same candidate is held for hold_seconds; re-arm on release."""

    def __init__(self):
        self.candidate = None
        self.since = None
        self.fired = False

    def update(self, candidate, now, hold_seconds):
        if candidate != self.candidate:
            self.candidate = candidate
            self.since = now if candidate is not None else None
            self.fired = False
            return None
        if candidate is None or self.fired:
            return None
        if now - self.since >= float(hold_seconds):
            self.fired = True
            return candidate
        return None

    def progress(self, now, hold_seconds):
        if self.candidate is None or self.since is None:
            return 0.0
        if self.fired:
            return 1.0
        return min(1.0, max(0.0, (now-self.since)/max(1e-6, float(hold_seconds))))


def tli_overlap(left_point, right_point, diameter):
    """Whether two TLI circles (given as (x, y)) overlap."""
    if left_point is None or right_point is None:
        return False
    d = math.hypot(left_point[0]-right_point[0], left_point[1]-right_point[1])
    return d <= max(4.0, float(diameter))


def _valid_gesture(item):
    if not isinstance(item, dict):
        return None
    kind = item.get("kind")
    action = item.get("action", "none")
    if action not in ACTION_IDS:
        action = "none"
    timer = bool(item.get("timer", True))
    if kind != "pose":  # includes old "tli_overlap" rows: overlap is now built in
        return None
    features = item.get("features")
    try:
        features = [float(v) for v in features]
    except (TypeError, ValueError):
        return None
    if len(features) == LEGACY_FEATURE_COUNT:
        # Recorded before the thumb-across/direction values existed: neutral padding.
        features = features + [0.0, 0.0]
    if len(features) == PRE_SPREAD_FEATURE_COUNT:
        # Recorded before finger spread was measured: unknown, so never a split.
        features = features + [SPREAD_UNKNOWN]*len(SPREAD_GAPS)
    if len(features) != FEATURE_COUNT or not all(math.isfinite(v) for v in features):
        return None
    side = item.get("side", "either")
    side = side if side in SIDES else "either"
    key_combo = str(item.get("key_combo") or "")[:80]
    return {"kind": "pose", "side": side, "features": features,
            "label": str(item.get("label") or describe(features))[:16], "action": action, "timer": timer,
            "key_combo": key_combo}


def load_custom_gestures(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    items = data.get("gestures", []) if isinstance(data, dict) else []
    out = [g for g in (_valid_gesture(i) for i in items) if g is not None]
    return out[:MAX_GESTURES]


def save_custom_gestures(path, gestures):
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    clean = [g for g in (_valid_gesture(i) for i in gestures) if g is not None][:MAX_GESTURES]
    try:
        temp.write_text(json.dumps({"version": 1, "gestures": clean}, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, target)
        return True
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
