"""Small, dependency-free decisions for Spatial Desktop gesture arbitration."""
from __future__ import annotations

import math

WINDOW_CONTROL_HIT_TESTS = {3, 5, 8, 9, 20, 21}



def scale_hand_point(point, origin, scale):
    """Apply the same origin/scale mapping to any landmark or interaction point."""
    x, y = float(point[0]), float(point[1])
    ox, oy = float(origin[0]), float(origin[1])
    factor = min(1.0, max(0.3, float(scale)))
    return ox + (x - ox) * factor, oy + (y - oy) * factor


def quick_pinch_click_eligible(pinch_started_at, release_started_at,
                               min_seconds=0.06, max_seconds=0.65):
    """Whether one released pinch is a discrete click rather than a menu hold."""
    if pinch_started_at is None or release_started_at is None:
        return False
    held = float(release_started_at) - float(pinch_started_at)
    return max(0.0, float(min_seconds)) <= held <= max(0.0, float(max_seconds))




# How long a released quick pinch may wait for the hand to show whether it opened
# (a click) or closed (a pinch seen in passing while making a thumbs up or a fist).
PINCH_RELEASE_SETTLE_SECONDS = 0.30
# A pinch must last this long before moving it starts a drag (mouse button held).
PINCH_DRAG_MIN_HOLD_SECONDS = 0.15


def pinch_release_click_verdict(release_for, is_open, fingers_closed,
                                settle_seconds=PINCH_RELEASE_SETTLE_SECONDS):
    """'click', 'cancel' or 'wait' for a released quick pinch.

    Closing the hand (into a thumbs up, thumbs down or fist) sweeps the index finger
    past the thumb, which reads as a brief pinch. A real click releases into an open
    hand; a pass-through pinch releases into folded fingers.
    """
    if fingers_closed:
        return "cancel"
    if is_open or float(release_for) >= float(settle_seconds):
        return "click"
    return "wait"


def pinch_drag_should_start(origin, point, movement_threshold=10.0):
    """Start a click-and-hold drag when a held pinch moves beyond tracking jitter.

    The caller must additionally ensure the Pinch Menu timer has not completed and
    that the gesture is not owned by a modifier or UI control.
    """
    if origin is None or point is None:
        return False
    try:
        ox, oy = float(origin[0]), float(origin[1])
        x, y = float(point[0]), float(point[1])
        threshold = max(1.0, float(movement_threshold))
    except (TypeError, ValueError, IndexError):
        return False
    if not all(math.isfinite(value) for value in (ox, oy, x, y, threshold)):
        return False
    return math.hypot(x-ox, y-oy) >= threshold


def align_double_click_point(previous_time, previous_point, now, point,
                             window_seconds=0.45, max_distance=60.0):
    """Snap a second nearby click to the first click's point within the click window."""
    current = (float(point[0]), float(point[1]))
    if previous_time is None or previous_point is None:
        return current, False
    prior = (float(previous_point[0]), float(previous_point[1]))
    elapsed = float(now) - float(previous_time)
    distance = math.hypot(current[0]-prior[0], current[1]-prior[1])
    if 0.0 <= elapsed <= max(0.0, float(window_seconds)) and distance <= max(0.0, float(max_distance)):
        return prior, True
    return current, False


def hand_apparent_size_scale(palm_length, palm_width, reference_size,
                             min_scale=0.45, max_scale=2.25):
    """Return a direct (not inverse) screen-space size factor for all hand strokes."""
    length = max(1.0, float(palm_length))
    width = max(1.0, float(palm_width))
    reference = max(1.0, float(reference_size))
    apparent_size = math.sqrt(length * width)
    return min(float(max_scale), max(float(min_scale), apparent_size / reference))


# MediaPipe hand skeleton bones (landmark index pairs), plus palm edges so that a
# fingertip resting on the palm is measured against the palm, not just its joints.
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    (0, 9), (0, 13), (5, 17),
)


def _point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx-ax, by-ay
    length_sq = dx*dx + dy*dy
    if length_sq <= 1e-12:
        return math.hypot(p[0]-ax, p[1]-ay)
    t = min(1.0, max(0.0, ((p[0]-ax)*dx + (p[1]-ay)*dy)/length_sq))
    return math.hypot(p[0]-(ax+t*dx), p[1]-(ay+t*dy))


def hand_contact_gap(left_points, right_points):
    """Smallest 2D gap between two hands' skeletons, in the points' own units.

    Every landmark (plus the palm centre) of each hand is measured against every bone
    of the other, so fingertip-to-fingertip, fingertip-to-palm and palm-to-palm
    contact are all detected rather than just the nearest fingertip pair.
    Returns None when either hand is incomplete.
    """
    if not left_points or not right_points or len(left_points) < 21 or len(right_points) < 21:
        return None
    hands = []
    for raw in (left_points, right_points):
        pts = [(float(p[0]), float(p[1])) for p in raw[:21]]
        palm = [pts[i] for i in (0, 5, 9, 13, 17)]
        centre = (sum(p[0] for p in palm)/5.0, sum(p[1] for p in palm)/5.0)
        hands.append((pts, pts + [centre]))
    best = math.inf
    for (pts_a, probes_a), (pts_b, _probes_b) in ((hands[0], hands[1]), (hands[1], hands[0])):
        for probe in probes_a:
            for i, j in HAND_BONES:
                d = _point_segment_distance(probe, pts_b[i], pts_b[j])
                if d < best:
                    best = d
    return best if math.isfinite(best) else None


def shared_contact_weight(gap, touch_distance, release_distance):
    """1.0 at/inside touch distance, 0.0 at/beyond release distance, smoothstep between."""
    if gap is None:
        return 0.0
    inner = max(0.0, float(touch_distance))
    outer = max(inner + 1e-6, float(release_distance))
    t = min(1.0, max(0.0, (outer - float(gap)) / (outer - inner)))
    return t*t*(3.0 - 2.0*t)


def shared_contact_origins(wrists, weight):
    """Blend each hand's scale pivot from its own wrist to one shared pivot.

    Scaling both hands about the SAME point preserves every relative position exactly
    (only multiplied by the scale), so real contact stays contact at any hand scale.
    The shared pivot is the midpoint of the wrists, which keeps the pair's centre where
    it was in independent-wrist mode: the hands meet in the middle and nothing drifts.
    With weight 0 each hand keeps its own wrist pivot (the single-hand behaviour).
    """
    origins = {side: (float(p[0]), float(p[1])) for side, p in wrists.items()}
    w = min(1.0, max(0.0, float(weight)))
    if w <= 0.0 or not all(side in origins for side in ("left", "right")):
        return origins
    mx = (origins["left"][0] + origins["right"][0]) * 0.5
    my = (origins["left"][1] + origins["right"][1]) * 0.5
    return {side: (x + (mx - x)*w, y + (my - y)*w) for side, (x, y) in origins.items()}


def ease_toward(current, target, dt, engage_seconds=0.05, release_seconds=0.09):
    """Frame-rate independent exponential easing; engages a little faster than it lets go."""
    tau = engage_seconds if target > current else release_seconds
    dt = max(0.0, float(dt))
    if tau <= 0.0 or dt >= 1.0:
        return float(target)
    alpha = 1.0 - math.exp(-dt / tau)
    return float(current) + (float(target) - float(current)) * alpha


def pinch_timer_elapsed_seconds(sequence_started, accumulated_seconds,
                                pulse_started_at, release_started_at, now,
                                buffer_seconds=0.8):
    """Count only active pinch time after the sequence's confirmation buffer."""
    if sequence_started is None:
        return 0.0
    elapsed = max(0.0, float(accumulated_seconds))
    if pulse_started_at is None:
        return elapsed
    end = float(now) if release_started_at is None else min(float(now), float(release_started_at))
    fill_started = max(float(pulse_started_at), float(sequence_started) + max(0.0, float(buffer_seconds)))
    return elapsed + max(0.0, end - fill_started)


def pinch_timer_pulse_seconds(sequence_started, pulse_started_at,
                              release_started_at, buffer_seconds=0.8):
    """Return one released pulse's eligible timer contribution, excluding buffer time."""
    if sequence_started is None or pulse_started_at is None or release_started_at is None:
        return 0.0
    fill_started = max(float(pulse_started_at), float(sequence_started) + max(0.0, float(buffer_seconds)))
    return max(0.0, float(release_started_at) - fill_started)


def pinch_timer_release_before_buffer(sequence_started, release_started_at,
                                      buffer_seconds=0.8):
    """True when a pinch was released before its initial confirmation buffer elapsed."""
    return (sequence_started is not None and release_started_at is not None and
            float(release_started_at) < float(sequence_started) + max(0.0, float(buffer_seconds)))


def pinch_timer_gap_restarts_buffer(sequence_started, release_started_at, now,
                                   buffer_seconds=0.8, jitter_tolerance=0.0):
    """Restart confirmation after any detected release/re-pinch before the buffer.

    Tracking loss is handled separately by the frame loop. If a hand is still tracked
    but the pinch pose releases, the next pinch must earn a fresh continuous buffer.
    This prevents quick double-pinches from sharing elapsed confirmation time.
    """
    return (pinch_timer_release_before_buffer(sequence_started, release_started_at, buffer_seconds) and
            float(now) - float(release_started_at) >= max(0.0, float(jitter_tolerance)))


def quick_pinch_reentry_eligible(sequence_started, pulse_started_at,
                                 release_started_at, now, click_fired=False,
                                 timer_latched=False, buffer_seconds=0.8,
                                 min_release_gap=0.0, min_pulse_seconds=0.06,
                                 max_pulse_seconds=0.65):
    """Whether a re-pinch should finish the previous short pulse as a click.

    A single observed non-pinch frame is enough
    to separate two gestures. Do not require a 60ms gap: that drops legitimate rapid
    double-pinches. The pulse itself must still be long enough to be intentional.
    """
    if sequence_started is None or release_started_at is None or now is None:
        return False
    pulse_start = (sequence_started if pulse_started_at is None
                   else pulse_started_at)
    return (
        float(now) - float(release_started_at) >= max(0.0, float(min_release_gap))
        and not bool(click_fired)
        and not bool(timer_latched)
        and pinch_timer_release_before_buffer(
            sequence_started, release_started_at, buffer_seconds
        )
        and quick_pinch_click_eligible(
            pulse_start, release_started_at,
            min_pulse_seconds, max_pulse_seconds
        )
    )

def should_arm_shake_drag(*, hit_test, has_target_window):
    """Only arm shake-drag on a real window body, not title-bar menu/buttons."""
    return bool(has_target_window and hit_test not in WINDOW_CONTROL_HIT_TESTS)



def should_dismiss_pinch_menu(*, selection_ready: bool, is_fist: bool,
                               is_pinch: bool, point_inside_popup: bool) -> bool:
    """Dismiss on a fresh grip/pinch outside the popup after opening pose is released."""
    return bool(selection_ready and (is_fist or is_pinch) and not point_inside_popup)



def should_prefer_clear_left_pinch_over_fist(*, is_pinch, normalized_gap):
    """Identify a tight left-hand thumb/index pinch that can clear a stale fist label.

    This exception is only used for the left hand. Right-hand fist priority remains
    absolute because a right fist owns fist-drag page scrolling.
    """
    if not is_pinch:
        return False
    try:
        gap = float(normalized_gap)
    except (TypeError, ValueError):
        return False
    return math.isfinite(gap) and 0.0 <= gap <= 0.40


def thumb_pinches_index(thumb_to_fingertip_distances):
    """True only when the thumb tip is nearer the index fingertip than any other fingertip.

    Distances are thumb-tip to (index, middle, ring, pinky) tips. Thumb-to-middle,
    -ring or -pinky pinches leave the index tip close to the thumb too, so the
    thumb/index gap alone cannot tell them apart; those poses are kept free for
    custom gestures and must never trigger the built-in index pinch.
    """
    try:
        d = [float(value) for value in thumb_to_fingertip_distances]
    except (TypeError, ValueError):
        return False
    if len(d) < 4 or not all(math.isfinite(value) for value in d[:4]):
        return False
    return d[0] < min(d[1:4])


def thumb_middle_pinch_pose(*, normalized_middle_gap, normalized_index_gap,
                            finger_straightness, edge_tolerant=False):
    """Recognise thumb-to-middle fingertip contact, distinct from the menu pinch.

    The middle tip must be close to the thumb relative to palm length and clearly
    closer than the index tip. At least two of index/ring/little should remain
    extended, so a curled fist or an ordinary thumb-index pinch does not qualify.
    ``finger_straightness`` is ordered index, middle, ring, little.
    """
    try:
        middle_gap = float(normalized_middle_gap)
        index_gap = float(normalized_index_gap)
        straight = [float(value) for value in finger_straightness]
    except (TypeError, ValueError):
        return False
    if (len(straight) < 4 or not all(math.isfinite(v) for v in (middle_gap, index_gap, *straight[:4]))):
        return False
    max_gap = 0.57 if edge_tolerant else 0.50
    if middle_gap > max_gap or middle_gap >= index_gap * 0.90:
        return False
    extended_threshold = 0.64 if edge_tolerant else 0.68
    return sum(straight[i] >= extended_threshold for i in (0, 2, 3)) >= 2


def opposite_diagonal_corner_assignment(rect, left_point, right_point,
                                        tolerance_ratio=0.14,
                                        min_tolerance=28.0, max_tolerance=82.0):
    """Return per-hand opposite-corner labels when both points pin a window's diagonal.

    ``rect`` is (left, top, right, bottom); points are global (x, y). Coordinates
    must be near actual corners, rather than merely near any point on the border.
    """
    try:
        left, top, right, bottom = (float(v) for v in rect)
        points = {"left": (float(left_point[0]), float(left_point[1])),
                  "right": (float(right_point[0]), float(right_point[1]))}
    except (TypeError, ValueError, IndexError):
        return None
    width, height = right-left, bottom-top
    if width <= 0 or height <= 0:
        return None
    tol_x = min(float(max_tolerance), max(float(min_tolerance), width*float(tolerance_ratio)))
    tol_y = min(float(max_tolerance), max(float(min_tolerance), height*float(tolerance_ratio)))
    corners = {"tl": (left, top), "tr": (right, top),
               "bl": (left, bottom), "br": (right, bottom)}
    nearest = {}
    for side, (x, y) in points.items():
        ranked = sorted(((math.hypot((x-cx)/tol_x, (y-cy)/tol_y), name, cx, cy)
                         for name, (cx, cy) in corners.items()), key=lambda row: row[0])
        distance, name, cx, cy = ranked[0]
        if distance > math.sqrt(2.0) or abs(x-cx) > tol_x or abs(y-cy) > tol_y:
            return None
        nearest[side] = name
    opposite = {("tl", "br"), ("br", "tl"), ("tr", "bl"), ("bl", "tr")}
    if (nearest["left"], nearest["right"]) not in opposite:
        return None
    return nearest


def classify_middle_pinch_window_action(rect, left_point, right_point,
                                        corner_zone_ratio=0.25):
    """Choose window resize near opposite corners, content zoom elsewhere in a window.

    Coordinates are global screen positions; ``rect`` is (left, top, right, bottom).
    Each corner zone extends inward by 25% of both dimensions by default. Only a pair
    in opposite diagonal corner zones chooses window resizing. Other pairs inside the
    same application window choose content zoom/stretch, independent of application
    name or process type. Returns (action, corner_assignment), where action is
    ``"resize"`` or ``"zoom"`` and the assignment is None for zoom.
    """
    try:
        left, top, right, bottom = (float(v) for v in rect)
        points = {"left": (float(left_point[0]), float(left_point[1])),
                  "right": (float(right_point[0]), float(right_point[1]))}
        ratio = float(corner_zone_ratio)
    except (TypeError, ValueError, IndexError):
        return "zoom", None
    width, height = right-left, bottom-top
    if width <= 0.0 or height <= 0.0 or not math.isfinite(ratio):
        return "zoom", None
    ratio = min(0.49, max(0.01, ratio))
    # The mode is selected only when both pinches begin inside the same window.
    for x, y in points.values():
        if not (left <= x < right and top <= y < bottom):
            return "zoom", None
    corners = opposite_diagonal_corner_assignment(
        rect, points["left"], points["right"],
        tolerance_ratio=ratio, min_tolerance=1.0,
        max_tolerance=max(width, height))
    if corners is not None:
        return "resize", corners
    return "zoom", None


def resize_rect_from_corner_deltas(rect, start_points, current_points, corners,
                                   min_width=260.0, min_height=160.0):
    """Calculate a native-window rect by dragging its two pinned diagonal corners."""
    try:
        left, top, right, bottom = (float(v) for v in rect)
        new_edges = {"l": left, "t": top, "r": right, "b": bottom}
        for side in ("left", "right"):
            sx, sy = (float(v) for v in start_points[side])
            cx, cy = (float(v) for v in current_points[side])
            dx, dy = cx-sx, cy-sy
            corner = corners[side]
            if corner not in ("tl", "tr", "bl", "br"):
                return None
            # Normal corner names are two characters (tl/tr/bl/br), with axis order
            # y then x; apply each hand's movement to the two edges it controls.
            if corner[1] == "l":
                new_edges["l"] = left + dx
            else:
                new_edges["r"] = right + dx
            if corner[0] == "t":
                new_edges["t"] = top + dy
            else:
                new_edges["b"] = bottom + dy
        # If a moved edge reaches the opposite edge, preserve the opposite edge and
        # clamp the moved one, so the window never collapses or reverses its geometry.
        if new_edges["r"]-new_edges["l"] < float(min_width):
            if new_edges["l"] != left:
                new_edges["l"] = new_edges["r"]-float(min_width)
            else:
                new_edges["r"] = new_edges["l"]+float(min_width)
        if new_edges["b"]-new_edges["t"] < float(min_height):
            if new_edges["t"] != top:
                new_edges["t"] = new_edges["b"]-float(min_height)
            else:
                new_edges["b"] = new_edges["t"]+float(min_height)
        return (new_edges["l"], new_edges["t"], new_edges["r"], new_edges["b"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def resize_rect_from_distance_scale(rect, start_distance, current_distance,
                                    min_width=320.0, min_height=200.0,
                                    min_scale=0.35, max_scale=3.0):
    """Scale a window around its centre as two hand anchors move apart/together.

    This is the forgiving fallback when a user pinches over an application but does
    not start precisely on opposite corners. A relative distance scale makes it
    possible to enlarge ordinary application windows without first finding their
    exact border pixels.
    """
    try:
        left, top, right, bottom = (float(v) for v in rect)
        initial = float(start_distance)
        current = float(current_distance)
        if not all(math.isfinite(v) for v in (left, top, right, bottom, initial, current)):
            return None
        width, height = right-left, bottom-top
        if width <= 0.0 or height <= 0.0 or initial <= 1.0 or current < 0.0:
            return None
        scale = min(float(max_scale), max(float(min_scale), current/initial))
        new_width = max(float(min_width), width*scale)
        new_height = max(float(min_height), height*scale)
        centre_x = (left+right)*0.5
        centre_y = (top+bottom)*0.5
        return (centre_x-new_width*0.5, centre_y-new_height*0.5,
                centre_x+new_width*0.5, centre_y+new_height*0.5)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def should_recognize_menu_pinch(*, normalized_gap, finger_straightness, edge_tolerant=False):
    """Recognise a natural thumb/index pinch even if the index bends slightly.

    At least two of the three non-index long fingers should be extended. This is
    intentionally more tolerant than requiring every finger to be perfectly straight,
    while a closed fist (three curled fingers) does not qualify.
    """
    try:
        gap = float(normalized_gap)
        straight = [float(value) for value in finger_straightness]
    except (TypeError, ValueError):
        return False
    if len(straight) < 4 or not math.isfinite(gap):
        return False
    if not all(math.isfinite(value) for value in straight[:4]):
        return False
    # MediaPipe jitter and slight wrist rotation can otherwise miss an intentional pinch.
    # Keep the non-index finger requirement so a closed fist still cannot qualify.
    max_gap = 0.59 if edge_tolerant else 0.54
    extended_threshold = 0.68 if edge_tolerant else 0.72
    extended_non_index = sum(value >= extended_threshold for value in straight[1:4])
    return 0.0 <= gap <= max_gap and extended_non_index >= 2


def detect_fist_shake(samples, *, now=None, window_seconds=0.95, min_horizontal_span=42.0,
                      min_reversals=3, min_step=7.5, max_vertical_span=68.0,
                      settle_seconds=0.16):
    """Recognise an intentional side-to-side fist shake from (time, x, y) samples.

    Requiring several large direction changes within a short window avoids arming a
    window drag from ordinary pointer jitter or a single large cursor movement.
    Coordinates may be unbounded screen-mapped positions so edge clamping doesn't
    hide a shake near the display edge.
    """
    if not samples:
        return False
    latest = float(samples[-1][0] if now is None else now)
    recent = [sample for sample in samples
              if latest-float(sample[0]) <= window_seconds and latest >= float(sample[0])]
    if len(recent) < 5:
        return False
    xs = [float(sample[1]) for sample in recent]
    ys = [float(sample[2]) for sample in recent]
    if max(xs)-min(xs) < min_horizontal_span or max(ys)-min(ys) > max_vertical_span:
        return False
    directions = []
    anchor_x = xs[0]
    last_significant_move_at = float(recent[0][0])
    for sample in recent[1:]:
        x = float(sample[1])
        dx = x-anchor_x
        if abs(dx) >= min_step:
            direction = 1 if dx > 0 else -1
            if not directions or directions[-1] != direction:
                directions.append(direction)
            anchor_x = x
            last_significant_move_at = float(sample[0])
    reversals = sum(1 for a, b in zip(directions, directions[1:]) if a != b)
    # Confirm that the shake has actually finished. Arming on the third reversal would
    # make the remaining half of the shake drag the window, contrary to the user's intent.
    return reversals >= min_reversals and latest-last_significant_move_at >= settle_seconds


def should_start_scroll_action(*, held_seconds, confirmation_delay, dx, dy,
                               already_armed, scroll_mode_enabled, hit_test=None,
                               movement_threshold=100.0):
    """Start scrolling only for vertical movement in client/scrollbar content.

    A title-bar/non-client gesture is never misclassified as a page scroll. The user
    explicitly selects drag by shaking a fist side-to-side.
    """
    if already_armed or held_seconds < confirmation_delay:
        return False
    try:
        dx, dy = float(dx), float(dy)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return False
    if hit_test not in (None, 0, 1, 6, 7):
        return False
    # A scrollbar can respond to shorter movement; content scrolling needs a decisive
    # vertical trajectory to avoid confusing a drag with scrolling.
    threshold = 34.0 if hit_test in (6, 7) else (52.0 if scroll_mode_enabled else movement_threshold)
    if abs(dy) < threshold or abs(dy) < max(34.0, abs(dx)*1.20):
        return False
    return True


def should_start_right_fist_scroll(*, side, is_fist, left_fist_active, modifier_active,
                                   already_armed, dx, dy, hit_test=None,
                                   movement_threshold=8.0):
    """Start a right-hand-only scroll from deliberate vertical fist movement.

    There is intentionally no hold timer: the left fist is reserved for modifier /
    two-hand drag, and the right fist scrolls once vertical movement is clear. The
    target must be page content or a scrollbar, never a title-bar/window control.
    """
    if (side != "right" or not is_fist or left_fist_active or modifier_active or
            already_armed or hit_test not in (None, 0, 1, 6, 7)):
        return False
    try:
        dx, dy, threshold = float(dx), float(dy), float(movement_threshold)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (dx, dy, threshold)):
        return False
    threshold = max(6.0, threshold)
    # Require modest but intentional vertical travel. Horizontal motion remains a
    # separate window-drag gesture and tiny tracking noise cannot start scrolling.
    return abs(dy) >= threshold and abs(dy) >= max(6.0, abs(dx) * 0.9)


def active_fist_scroll_release_confirmed(*, is_open, is_fist, is_pinch,
                                         fingers_closed, open_duration,
                                         release_seconds=0.48):
    """Only release one-hand fist scrolling after a sustained, unambiguous open pose.

    Hand landmark classifiers can briefly label a curled grip as open while the fist
    anchor is moving. Requiring both an open hand and non-folded finger landmarks for
    a short continuous interval preserves the active scroll through those flickers.
    """
    if not is_open or is_fist or is_pinch or fingers_closed:
        return False
    try:
        elapsed = float(open_duration)
        required = max(0.10, float(release_seconds))
    except (TypeError, ValueError):
        return False
    return math.isfinite(elapsed) and math.isfinite(required) and elapsed >= required


def progressive_scroll_gain(total_travel, *, slow_gain=0.40, ramp_distance=260.0, max_extra_gain=2.4):
    """Return a deliberately gentle, bounded gain for fist-controlled scrolling.

    Scrolling starts slowly and builds over sustained travel instead of quickly
    accelerating into large, jerky wheel events.
    """
    try:
        distance = float(total_travel)
    except (TypeError, ValueError):
        distance = 0.0
    if not math.isfinite(distance):
        distance = 0.0
    distance = max(0.0, distance)
    return max(0.05, float(slow_gain)) + min(max(0.0, float(max_extra_gain)),
                                            distance / max(1.0, float(ramp_distance)))


def drag_scroll_rate_from_offset(vertical_offset, *, deadzone=5.0, base_rate=1.2,
                                  ramp_distance=20.0, max_rate=20.0):
    """Return a signed scroll rate based on vertical distance from a drag anchor.

    Close to the anchor scrolling pauses. Moving farther away increases the rate,
    and the sign always follows the side of the anchor the hand occupies.
    """
    try:
        offset = float(vertical_offset)
        deadzone = max(0.0, float(deadzone))
        base_rate = max(0.0, float(base_rate))
        ramp_distance = max(1.0, float(ramp_distance))
        max_rate = max(base_rate, float(max_rate))
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in
               (offset, deadzone, base_rate, ramp_distance, max_rate)):
        return 0.0
    distance = abs(offset)
    if distance <= deadzone:
        return 0.0
    rate = min(max_rate, base_rate + (distance-deadzone)/ramp_distance)
    return rate if offset > 0 else -rate


def smooth_scroll_delta(previous, current, *, previous_weight=0.65, max_delta=24.0):
    """Clamp frame spikes and smooth vertical hand movement before scroll accumulation."""
    try:
        prev = float(previous)
        cur = float(current)
        weight = float(previous_weight)
        limit = abs(float(max_delta))
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in (prev, cur, weight, limit)):
        return 0.0
    weight = min(0.95, max(0.0, weight))
    limit = max(1.0, limit)
    prev = min(limit, max(-limit, prev))
    cur = min(limit, max(-limit, cur))
    return prev*weight + cur*(1.0-weight)


def is_plausible_hand_landmarks(landmarks, score, *, min_score=0.60, min_width=0.045, min_height=0.055):
    """Reject phantom detections while retaining clearly shaped side-on hands.

    Most detections must have a normal 2-D hand footprint. A narrow edge-on footprint
    is accepted only when the other image axis is substantial, the landmarks span real
    depth, the wrist-to-palm distance is non-degenerate, and handedness confidence is
    high. This avoids treating arbitrary thin blobs or background noise as hands.
    Accepts MediaPipe landmark objects or (x, y[, z]) sequences for offline tests.
    """
    try:
        confidence = float(score)
        if (not math.isfinite(confidence) or confidence < min_score or
                landmarks is None or len(landmarks) < 21):
            return False
        xy = []
        xyz = []
        for point in landmarks[:21]:
            if hasattr(point, "x") and hasattr(point, "y"):
                x, y = float(point.x), float(point.y)
                z = float(getattr(point, "z", 0.0))
            else:
                x, y = float(point[0]), float(point[1])
                z = float(point[2]) if len(point) > 2 else 0.0
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return False
            # A modest off-frame tolerance allows real hands at screen edges but rejects
            # wildly misplaced landmarks that often accompany failed inference.
            if x < -0.12 or x > 1.12 or y < -0.12 or y > 1.12:
                return False
            xy.append((x, y))
            xyz.append((x, y, z))
        wrist_x, wrist_y = xy[0]
        if not (-0.08 <= wrist_x <= 1.08 and -0.08 <= wrist_y <= 1.08):
            return False
        xs = [point[0] for point in xy]
        ys = [point[1] for point in xy]
        zs = [point[2] for point in xyz]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        z_span = max(zs) - min(zs)
        if width > 0.92 or height > 0.98:
            return False

        palm_2d = math.hypot(xy[0][0] - xy[9][0], xy[0][1] - xy[9][1])
        palm_3d = math.sqrt(sum((xyz[0][axis] - xyz[9][axis]) ** 2 for axis in range(3)))
        normal_footprint = width >= min_width and height >= min_height and palm_2d >= 0.022

        # Side-on exception: only one image axis may collapse, while the other remains
        # clearly hand-sized. Require reliable handedness, landmark depth, and a full
        # wrist-to-middle-MCP span before relaxing either 2-D dimension.
        side_on_x = (0.012 <= width < min_width and height >= 0.10 and
                     z_span >= 0.025 and palm_3d >= 0.05 and confidence >= 0.66)
        side_on_y = (0.012 <= height < min_height and width >= 0.10 and
                     z_span >= 0.025 and palm_3d >= 0.05 and confidence >= 0.66)
        if not (normal_footprint or side_on_x or side_on_y):
            return False
        return palm_3d >= 0.035
    except (TypeError, ValueError, IndexError, AttributeError):
        return False


# Fist thumb check. Measured across the palm (index knuckle -> little-finger knuckle),
# in palm widths from the index knuckle: a thumb tucked over the fingers sits around
# +0.3, one resting along the side of the index finger near -0.15, and a thumbs-up /
# thumbs-down thumb well past the index side, around -0.6 to -0.9.
THUMB_OUT_ACROSS_PALM = -0.35


def thumb_across_palm(landmarks):
    """Thumb tip position across the palm, in palm widths from the index knuckle.

    Positive = over the fingers (tucked), negative = out past the index-finger side.
    Uses 640x360 camera-pixel scaling so x and y are comparable. Returns None on bad input.
    """
    try:
        if landmarks is None or len(landmarks) < 21:
            return None
        def pt(i):
            lm = landmarks[i]
            if isinstance(lm, (tuple, list)):
                x, y, z = float(lm[0]), float(lm[1]), float(lm[2]) if len(lm) > 2 else 0.0
            else:
                x, y, z = float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))
            return (x*640.0, y*360.0, z*640.0)
        index_mcp, pinky_mcp, tip = pt(5), pt(17), pt(4)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    across = tuple(p - q for p, q in zip(pinky_mcp, index_mcp))
    width = math.sqrt(sum(c*c for c in across))
    if not math.isfinite(width) or width < 1e-6:
        return None
    offset = tuple(p - q for p, q in zip(tip, index_mcp))
    value = sum(o*a for o, a in zip(offset, across)) / (width*width)
    return value if math.isfinite(value) else None


def thumb_axis_is_vertical_up(landmarks, max_angle_degrees=25.0, min_rise_pixels=8.0):
    """Require the thumb base-to-tip axis to point nearly straight up in the image.

    MediaPipe x/y are normalized independently, so convert to the camera's 640x360
    reference pixels before judging angle. This deliberately rejects diagonal thumbs
    that previously slipped through the thumbs-up timer detector.
    """
    try:
        if landmarks is None or len(landmarks) < 5:
            return False

        def xy(index):
            point = landmarks[index]
            if isinstance(point, (tuple, list)):
                return float(point[0]), float(point[1])
            return float(point.x), float(point.y)

        base_x, base_y = xy(2)
        tip_x, tip_y = xy(4)
        dx = (tip_x - base_x) * 640.0
        dy = (tip_y - base_y) * 360.0
        angle_limit = math.tan(math.radians(max(1.0, min(89.0, float(max_angle_degrees)))))
        rise = -dy
        return (math.isfinite(dx) and math.isfinite(dy) and
                rise >= max(0.0, float(min_rise_pixels)) and
                abs(dx) <= rise * angle_limit)
    except (TypeError, ValueError, IndexError, AttributeError):
        return False


def thumb_sticking_out(landmarks, threshold=THUMB_OUT_ACROSS_PALM):
    """True when the thumb points clearly away from a closed hand (thumbs up/down)."""
    value = thumb_across_palm(landmarks)
    return value is not None and value < float(threshold)
