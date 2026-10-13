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


def thumb_sticking_out(landmarks, threshold=THUMB_OUT_ACROSS_PALM):
    """True when the thumb points clearly away from a closed hand (thumbs up/down)."""
    value = thumb_across_palm(landmarks)
    return value is not None and value < float(threshold)
