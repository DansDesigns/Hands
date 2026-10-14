#!/usr/bin/env python3
"""Spatial Desktop — hand input, offline voice, local search and optional face welcome (v61).

The webcam supplies hand landmarks. TLI calibration is hand-relative (not camera-depth
calibration): it stores a pinch point in a palm-local coordinate frame. Clicks, drag/scroll,
right-click, and menu toggle are recognised by pose transitions and timed holds.
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import threading
import queue
import urllib.request
import re
import ctypes
import subprocess
import getpass
import shutil
from datetime import datetime
from ctypes import wintypes
from pathlib import Path

from gesture_rules import (
    should_dismiss_pinch_menu,
    detect_fist_shake,
    should_start_right_fist_scroll,
    should_arm_shake_drag,
    progressive_scroll_gain,
    drag_scroll_rate_from_offset,
    smooth_scroll_delta,
    should_recognize_menu_pinch,
    thumb_pinches_index,
    should_prefer_clear_left_pinch_over_fist,
    is_plausible_hand_landmarks,
    scale_hand_point,
    pinch_timer_elapsed_seconds,
    pinch_timer_pulse_seconds,
    pinch_timer_release_before_buffer,
    pinch_timer_gap_restarts_buffer,
    quick_pinch_click_eligible,
    pinch_release_click_verdict,
    PINCH_DRAG_MIN_HOLD_SECONDS,
    quick_pinch_reentry_eligible,
    align_double_click_point,
    hand_apparent_size_scale,
    hand_contact_gap,
    shared_contact_weight,
    shared_contact_origins,
    ease_toward,
    pinch_drag_should_start,
    thumb_sticking_out,
)
from voice_services import Speech, KokoroTTS
from hand_occlusion import TuckSmoother
from hand_contact import contact_offsets as hand_contact_offsets
from custom_gestures import (
    ACTION_IDS as CUSTOM_GESTURE_ACTION_IDS, ACTION_LABELS as CUSTOM_GESTURE_ACTION_LABELS,
    ACTION_GROUPS as CUSTOM_GESTURE_ACTION_GROUPS,
    MAX_GESTURES as CUSTOM_GESTURE_MAX, INSTANT_HOLD_SECONDS as CUSTOM_GESTURE_INSTANT_HOLD, HoldTrigger, hand_features, mean_features, describe as describe_pose,
    builtin_conflict, pinch_ratio as custom_pinch_ratio, best_pose_match, tli_overlap,
    load_custom_gestures, save_custom_gestures,
)
from voice_pc_commands import match_pc_command, aliases_for_help
from voice_assistant_commands import live_answer, file_query
from d1_decision import (classify_voice_command, start_loading as start_d1_loading,
                         selected_model as selected_d1_model, select_model as select_d1_model,
                         status as d1_status, MODELS as D1_MODELS, stop_server as stop_d1_server,
                         start_loading as start_d1_server)
from search_launcher import SearchLauncherDialog
from face_welcome import FaceWelcomeService
from tli_calibration import encode_relative_tli, resolve_relative_tli, mean_relative_tli
from updater import VERSION_FILE, compare_versions, read_version, fetch_remote_version, download_update, install_update
from hand_calibration import (
    load_hand_calibration, save_hand_calibration, palm_measurements,
    fingertip_contact_ratio, mean_finite, median_finite, screen_touch_offset,
    relative_palm_projection_scale, proximity_thresholds, default_calibration,
)

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as exc:
    print("MediaPipe is required:", exc)
    raise SystemExit(1)

from PySide6.QtCore import Qt, QTimer, QPoint, QPointF, QRectF, QObject, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetricsF, QCursor, QLinearGradient, QImage, QPainterPath, QPainterPathStroker, QRadialGradient, QPolygonF
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QInputDialog

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

try:
    import vosk
    import sounddevice as sd
except Exception:
    vosk = None
    sd = None

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
HAND_MODEL = MODELS / "hand_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
VOSK_MODEL = MODELS / "vosk-model-small-en-us-0.15"
GESTURES_FILE = ROOT / "gestures.txt"
UI_SETTINGS_FILE = ROOT / "ui_settings.json"
TLI_CALIBRATION_FILE = ROOT / "tli_calibration.json"
HAND_CALIBRATION_FILE = ROOT / "hand_calibration.json"
CUSTOM_GESTURES_FILE = ROOT / "custom_gestures.json"
NOTIFICATION_POSITIONS = ("top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right")
# Hand render styles, in the order shown by the Settings -> Hands switch.
HAND_RENDER_STYLES = ("skeleton", "flat", "3d")
HAND_RENDER_STYLE_LABELS = {"skeleton": "SKELETON", "flat": "FLAT", "3d": "3D"}
# Settings -> Notify rows: (ui preference key, label, default position).
NOTIFICATION_SETTING_ROWS = (
    ("paused_notice_position", "GESTURE LEGEND LOCATION", "bottom-center"),
    ("heard_bar_position", "HEARD / TRANSCRIPT BAR", "bottom-center"),
    ("left_status_position", "LEFT HAND STATUS", "bottom-left"),
    ("right_status_position", "RIGHT HAND STATUS", "bottom-right"),
    ("dictation_indicator_position", "DICTATION INDICATOR", "top-left"),
    ("stt_indicator_position", "STT INDICATOR", "top-right"),
)
TTS_VOICE_OPTIONS = (
    "vits-piper-en_GB-alan-medium",
    "vits-piper-en_GB-alba-medium",
    "vits-piper-en_GB-cori-medium",
    "vits-piper-en_GB-jenny_dioco-medium",
    "vits-piper-en_US-amy-low",
    "vits-piper-en_US-ryan-medium",
)

def load_ui_preferences() -> dict:
    defaults = {
        "paused_notice_position": "bottom-center",
        "heard_bar_position": "bottom-center",
        "left_status_position": "bottom-left",
        "right_status_position": "bottom-right",
        "dictation_indicator_position": "top-left",
        "stt_indicator_position": "top-right",
        "ui_font_scale": 1.2,
    }
    try:
        supplied = json.loads(UI_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return defaults.copy()
    result = defaults.copy()
    if isinstance(supplied, dict):
        for key in result:
            value = supplied.get(key)
            if key == "ui_font_scale":
                try:
                    result[key] = min(1.4, max(0.7, float(value)))
                except (TypeError, ValueError):
                    pass
            elif value in NOTIFICATION_POSITIONS:
                result[key] = value
    return result

def save_ui_preferences(values: dict) -> None:
    current = load_ui_preferences()
    for key in current:
        value = values.get(key, current[key])
        if key == "ui_font_scale":
            try:
                current[key] = min(1.4, max(0.7, float(value)))
            except (TypeError, ValueError):
                pass
        elif value in NOTIFICATION_POSITIONS:
            current[key] = value
    try:
        UI_SETTINGS_FILE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print("Could not save UI preferences:", exc)


def load_tli_calibrations() -> dict:
    """Load optional palm-relative TLI coordinates for each physical hand."""
    result = {"left": None, "right": None}
    try:
        supplied = json.loads(TLI_CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return result
    if not isinstance(supplied, dict):
        return result
    for side in result:
        value = supplied.get(side)
        if not isinstance(value, dict):
            continue
        try:
            coords = {key: float(value[key]) for key in ("forward", "lateral", "depth")}
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(number) and abs(number) <= 4.0 for number in coords.values()):
            result[side] = coords
    return result


def save_tli_calibrations(values: dict) -> bool:
    """Persist valid per-hand local coordinates atomically."""
    clean = {"left": None, "right": None}
    for side in clean:
        value = values.get(side) if isinstance(values, dict) else None
        if not isinstance(value, dict):
            continue
        try:
            coords = {key: float(value[key]) for key in ("forward", "lateral", "depth")}
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(number) and abs(number) <= 4.0 for number in coords.values()):
            clean[side] = coords
    temporary = TLI_CALIBRATION_FILE.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, TLI_CALIBRATION_FILE)
        return True
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        print("Could not save TLI calibration:", exc)
        return False


def save_gesture_settings(values: dict) -> None:
    """Persist settings used by the shape-and-timing hand input system."""
    def clamped(key, low, high, default):
        try:
            number = float(values.get(key, default))
            if not math.isfinite(number):
                number = default
        except (ValueError, TypeError):
            number = default
        return min(high, max(low, number))

    colors = {}
    for key, default in (("left_click_color", "#ff5555"), ("right_click_color", "#5091ff")):
        value = str(values.get(key, default))
        colors[key] = value.lower() if re.fullmatch(r"#[0-9a-fA-F]{6}", value) else default

    lines = [
        "# SPATIAL DESKTOP — HAND GESTURE SETTINGS",
        "# Lines beginning with # are comments. Use one setting_name=value per line.",
        "# Do not add quotes. Colours must be #RRGGBB. Numeric values may use decimals.",
        "# Settings panel changes are saved here automatically; restart to apply manual edits.",
        "#",
        "# ACTIVE GESTURES — no depth calibration or forward-push activation is used.",
        "# POINTER/TLI: saved palm-relative calibration point when calibrated; otherwise thumb/index fingertip midpoint.",
        "# QUICK CLICK: quick fist/grip -> open OR quick thumb/index pinch -> release clicks at the starting TLI point.",
        "# LEFT MODIFIER: hold LEFT fist; quick RIGHT pinch/release = right-click/context menu; held RIGHT grip = drag a native window (OS) or virtual canvas/app (3D).",
        "# Modifier gestures suppress the regular Pinch Menu timer; open RIGHT hand to release a modifier drag.",
        "# DOUBLE-CLICK: two quick thumb/index pinches near the same point are snapped to one target for a native double-click.",
        "# PINCH MENU: hold thumb/index pinch continuously for pinch_buffer_seconds (default 0.8s); then the circle fills for the hold timer.",
        "# PINCH MENU: while open, a fresh pinch/grab outside the popup closes it without clicking through.",
        "# FIST MOVEMENT: a moving RIGHT-only fist starts smooth vertical page scrolling without a fist timer; left fist remains the modifier, and two-fist window drag remains available.",
        "# TIMERS: only Pinch Menu and peace-sign gestures show countdowns. Right-fist scrolling starts on deliberate vertical movement and has no circle timer.",
        "# A release before 0.8s cancels that timer sequence. Quick/double pinches start a fresh buffer; after confirmation only held-pinch time counts.",
        "# A fist/grip has priority over pinch recognition; a fist alone cannot start the Pinch Menu timer.",
        "# PEACE SIGN: hold an index+middle V-sign for 1.5 seconds to detach/reattach that hand from native OS mouse movement; glass and Pinch Menu gestures and live TLI tracking remain active.",
        "# A detached hand cannot drag-scroll; reattach it to scroll with a fist.",
        "# FIST: fingers closed AND thumb tucked in. With the thumb sticking out it is not a fist (thumbs up/down are free for custom gestures).",
        "# TLI OVERLAP: bring both TLIs together and hold for the gesture hold time to toggle the glass menu.",
        "# MODES: select Right-click or Scroll Window in the Pinch Menu; a timed pinch while a mode is active returns to normal.",
        "# FIST: a quick fist-open clicks. A RIGHT-only fist starts scrolling when moved vertically over page content; make both fists for the separate two-hand window drag.",
        "# SCROLL: RIGHT-only fist + deliberate vertical movement over page content starts gentle scrolling immediately; there is no right-fist circle timer.",
        "# EDGE INPUT: gesture recognition is slightly more tolerant near screen edges; skeleton joints are not individually clamped.",
        "# RELEASE COOLDOWN: active drag/scroll stops after an open-hand pose remains stable for 0.20 seconds; timed pinches need a continuous 0.8-second confirmation hold.",
        "# SLIDER SNAP: fist OR thumb/index pinch near a settings slider snaps the TLI to its knob and starts dragging; no hold timer.",
        "# SLIDER STABILITY: isolated large cursor jumps are ignored; a new slider cursor position must persist for three frames.",
        "# HAND DETECTION: conservative MediaPipe confidence, three consistent detections, no cached-landmark display; side-on geometry requires high confidence and depth support.",
        "# PHYSICAL SIDES: camera handedness labels are normalized at the tracker boundary before cursor priority and colour assignment.",
        "# GLASS UI: menu and Settings panels use stronger Gaussian backdrop blur, a darker frosted tint, and soft halo edges.",
        "# PINCH MENU ITEMS: RIGHT-CLICK MODE, SCROLL WINDOW, and GLASS MENU. Grab or pinch an item to activate it.",
        "# RIGHT-CLICK MODE makes quick fist-open grabs right-click at the original TLI point.",
        "# DRAG WINDOW: left fist + right grip drags a native window in OS mode or the virtual canvas/app in 3D mode; open the right hand to release.",
        "# NORMAL TWO-HAND DRAG: make both hands fists; keep one still and move the other. Opening either releases.",
        "# SCROLL WINDOW mode makes a held fist scroll vertically inside the target window. A timed pinch exits the mode.",
        "# CAMERA OVERSCAN: the central 60% × 68% camera area maps to the full screen; outside it clamps to the nearest edge.",
        "# All gesture recognition uses hand shape and time. Relative camera depth is not used for activation.",
        "#",
        "# SETTINGS",
        "# hand_visual_scale: skeleton/TLI scale, 0.1 to 1.0.",
        "# hand_contact_band_ratio: palm attraction distance beyond touch, in palm widths; 0.0 to 0.9; default 0.02.",
        "# palm_attract_acceleration: speed multiplier for the virtual hands blending toward shared contact alignment; 0.2 to 5.0x; default 0.33x.",
        "# skin_tone_darkness: 0 to 100 percent extra warm darkening of the skin-like hand mesh.",
        "# finger_thickness: rounded finger segment thickness multiplier, 0.8 to 10.0; default 4.24.",
        "# palm_width_scale: palm contour width multiplier, 0.65 to 1.80; default 1.26.",
        "# hand_render_style: skeleton (lines), flat (painted 2D hand) or 3d (rigged 3D hand mesh).",
        "# hand_skin_effect: kept for older versions; true unless hand_render_style is skeleton.",
        "# TLI calibration: Hand Settings -> Calibrate Left/Right TLI; the live hand and transformed TLI are redrawn above the target and hit-tested at the same point.",
        "# Each TLI calibration stores the pinch midpoint in normalized palm-local 3D coordinates in tli_calibration.json.",
        "# HAND ONBOARDING: on first run, extend an arm and touch below the camera, then capture a palm-facing pose and a two-hand contact pose.",
        "# Hand scale/proximity calibration is saved in hand_calibration.json and can be repeated from Settings -> Hands.",
        "# Scroll input is injected as native wheel input at the live cursor so Explorer/Office controls receive it, not a posted event to the wrong top-level window.",
        "# pinch_buffer_seconds: Pinch Menu confirmation buffer before the circle starts, 0.3 to 2.0 seconds; default 0.8. Releasing early cancels the sequence (or clicks, if quick).",
        "# hand_tracking_sensitivity: drag movement gain and gentle scroll gain, 0.5 to 1.5; higher moves faster.",
        "# custom_gesture_hold_seconds: how long a recorded custom gesture must be held to fire, 0.2 to 1.5 s; default 0.5.",
        "# custom_gesture_tolerance: how loosely a live pose may match a recorded one, 0.08 to 0.40; default 0.18.",
        "# Custom gestures themselves are recorded from Settings -> Gestures and saved in custom_gestures.json.",
        "# detached_notice_opacity: opacity the persistent 'mouse detached' notice fades to 5 s after the glass menu closes, 0 to 100 percent; default 30.",
        "# scroll_deadzone: grip/fist scroll dead zone around the grab point, 4 to 120 pixels; default 70.",
        "# scroll_acceleration: scroll-rate increase with distance from the drag anchor, 0.5 to 10.0x; default 0.33x.",
        "# cursor_smoothing: adaptive jitter damping, 0.0 to 0.95; default 0.90.",
        "# tli_glow_intensity: TLI halo intensity multiplier, 0.0 to 2.5.",
        "# tli_colour_shift_degrees: hue shift applied to both TLI colours, 0 to 360 degrees.",
        "# tli_border_thickness: TLI outline thickness, 1.5 to 8.0 pixels.",
        "# tli_diameter: TLI outer diameter, 12 to 96 pixels; default 30 pixels.",
        "# skeleton_line_thickness: skeleton stroke multiplier, 0.5 to 4.0; default 1.5.",
        "# click_debounce_seconds: minimum fist duration accepted as a click, 0.04 to 0.25 seconds.",
        "# default_gesture_hold_seconds: shared activation duration for peace sign, thumbs up, and Pinch Menu, 0.5 to 5.0 seconds; default 0.5.",
        "# invert_scroll: true reverses vertical scroll direction; false is normal.",
        "# swap_drag_scroll_notches: true swaps the drag-scroll Back/Forward navigation notch directions.",
        "# cursor_lock_priority: right or left; selected physical hand gets first refusal for new gestures when both are visible.",
        "# Quick pinch clicks require a discrete short pulse before the 0.8s timer buffer; two quick pinches at one point form a native double-click.",
        "# Settings slider drag uses a reduced gain (0.42x) for finer control; initial pickup preserves the value.",
        "# Default TLI colours: LEFT red, RIGHT blue. Swatches can still cycle independently.",
        f"hand_visual_scale={clamped('hand_visual_scale', 0.1, 1.0, 0.3):.2f}",
        f"hand_recognition_threshold={clamped('hand_recognition_threshold', 0.30, 0.90, 0.60):.2f}",
        f"hand_contact_band_ratio={clamped('hand_contact_band_ratio', 0.0, 0.9, 0.02):.2f}",
        f"palm_attract_acceleration={clamped('palm_attract_acceleration', 0.2, 5.0, 0.33):.2f}",
        f"skin_tone_darkness={clamped('skin_tone_darkness', 0.0, 100.0, 0.0):.1f}",
        f"finger_thickness={clamped('finger_thickness', 0.8, 10.0, 4.24):.2f}",
        f"palm_width_scale={clamped('palm_width_scale', 0.65, 1.8, 1.26):.2f}",
        f"hand_skin_effect={str(bool(values.get('hand_skin_effect', True))).lower()}",
        f"hand_render_style={values.get('hand_render_style') if values.get('hand_render_style') in HAND_RENDER_STYLES else ('flat' if values.get('hand_skin_effect', True) else 'skeleton')}",
        f"skin_use_skeleton_color={str(bool(values.get('skin_use_skeleton_color', False))).lower()}",
        f"hand_transparency={clamped('hand_transparency', 10.0, 100.0, 75.0):.1f}",
        f"tli_glow_enabled={str(bool(values.get('tli_glow_enabled', False))).lower()}",
        f"hand_tracking_sensitivity={clamped('hand_tracking_sensitivity', 0.5, 1.5, 1.5):.2f}",
        f"scroll_acceleration={clamped('scroll_acceleration', 0.5, 10.0, 1.0):.2f}",
        f"scroll_deadzone={clamped('scroll_deadzone', 4.0, 120.0, 70.0):.1f}",
        f"scroll_backdrop_opacity={clamped('scroll_backdrop_opacity', 0.0, 100.0, 45.0):.1f}",
        f"scroll_backdrop_hue={clamped('scroll_backdrop_hue', 0.0, 255.0, 145.0):.1f}",
        f"scroll_notch_hue={clamped('scroll_notch_hue', 0.0, 360.0, 120.0):.1f}",
        f"detached_notice_opacity={clamped('detached_notice_opacity', 0.0, 100.0, 30.0):.1f}",
        f"custom_gesture_hold_seconds={clamped('custom_gesture_hold_seconds', 0.2, 1.5, 0.5):.2f}",
        f"custom_gesture_tolerance={clamped('custom_gesture_tolerance', 0.08, 0.40, 0.18):.3f}",
        f"cursor_smoothing={clamped('cursor_smoothing', 0.0, 0.95, 0.9):.2f}",
        f"tli_glow_intensity={clamped('tli_glow_intensity', 0.0, 2.5, 0.0):.2f}",
        f"tli_colour_shift_degrees={clamped('tli_colour_shift_degrees', 0.0, 360.0, 0.0):.1f}",
        f"tli_border_thickness={clamped('tli_border_thickness', 1.0, 8.0, 1.0):.2f}",
        f"tli_diameter={clamped('tli_diameter', 12.0, 96.0, 30.0):.1f}",
        f"skeleton_line_thickness={clamped('skeleton_line_thickness', 0.5, 4.0, 1.5):.2f}",
        f"click_debounce_seconds={clamped('click_debounce_seconds', 0.04, 0.25, 0.085):.3f}",
        f"menu_pinch_hold_seconds={clamped('menu_pinch_hold_seconds', 0.5, 5.0, 0.6):.2f}",
        f"default_gesture_hold_seconds={clamped('default_gesture_hold_seconds', 0.5, 5.0, 0.5):.2f}",
        f"pinch_buffer_seconds={clamped('pinch_buffer_seconds', 0.3, 2.0, 0.8):.2f}",
        f"invert_scroll={str(bool(values.get('invert_scroll', True))).lower()}",
        f"swap_drag_scroll_notches={str(bool(values.get('swap_drag_scroll_notches', False))).lower()}",
        f"cursor_lock_priority={values.get('cursor_lock_priority', 'right') if values.get('cursor_lock_priority', 'right') in ('right', 'left') else 'right'}",
        f"left_click_color={colors['left_click_color']}",
        f"right_click_color={colors['right_click_color']}",
    ]
    try:
        GESTURES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        print("Could not save gestures.txt:", exc)


def load_gesture_settings() -> dict:
    """Load settings for cursor mapping, smoothing, timing, and hand colours."""
    values = {
        "hand_visual_scale": 0.3,
        "hand_recognition_threshold": 0.60,
        "hand_contact_band_ratio": 0.02,
        "palm_attract_acceleration": 0.33,
        "skin_tone_darkness": 0.0,
        "finger_thickness": 4.24,
        "palm_width_scale": 1.26,
        "hand_skin_effect": True,
        # New installations default to the rigged 3D mesh. Legacy files that
        # only specify hand_skin_effect continue to use their previous style.
        "hand_render_style": "3d",
        "skin_use_skeleton_color": False,
        "hand_transparency": 75.0,
        "tli_glow_enabled": False,
        "hand_tracking_sensitivity": 1.5,
        "scroll_acceleration": 1.0,
        "scroll_deadzone": 70.0,
        "scroll_backdrop_opacity": 45.0,
        "scroll_backdrop_hue": 145.0,
        "scroll_notch_hue": 120.0,
        "detached_notice_opacity": 30.0,
        "custom_gesture_hold_seconds": 0.5,
        "custom_gesture_tolerance": 0.18,
        "cursor_smoothing": 0.9,
        "tli_glow_intensity": 0.0,
        "tli_colour_shift_degrees": 0.0,
        "tli_border_thickness": 1.0,
        "tli_diameter": 30.0,
        "skeleton_line_thickness": 1.5,
        "click_debounce_seconds": 0.085,
        "menu_pinch_hold_seconds": 0.6,
        "default_gesture_hold_seconds": 0.5,
        "pinch_buffer_seconds": 0.8,
        "invert_scroll": True,
        "swap_drag_scroll_notches": False,
        "cursor_lock_priority": "right",
        "left_click_color": "#ff5555",
        "right_click_color": "#5091ff",
    }
    ranges = {
        "hand_visual_scale": (0.1, 1.0),
        "hand_recognition_threshold": (0.30, 0.90),
        "hand_contact_band_ratio": (0.0, 0.9),
        "palm_attract_acceleration": (0.2, 5.0),
        "skin_tone_darkness": (0.0, 100.0),
        "finger_thickness": (0.8, 10.0),
        "palm_width_scale": (0.65, 1.8),
        "hand_tracking_sensitivity": (0.5, 1.5),
        "scroll_acceleration": (0.5, 10.0),
        "scroll_deadzone": (4.0, 120.0),
        "scroll_backdrop_opacity": (0.0, 100.0),
        "scroll_backdrop_hue": (0.0, 255.0),
        "scroll_notch_hue": (0.0, 360.0),
        "detached_notice_opacity": (0.0, 100.0),
        "custom_gesture_hold_seconds": (0.2, 1.5),
        "custom_gesture_tolerance": (0.08, 0.40),
        "cursor_smoothing": (0.0, 0.95),
        "tli_glow_intensity": (0.0, 2.5),
        "tli_colour_shift_degrees": (0.0, 360.0),
        "tli_border_thickness": (1.0, 8.0),
        "tli_diameter": (12.0, 96.0),
        "skeleton_line_thickness": (0.5, 4.0),
        "click_debounce_seconds": (0.04, 0.25),
        "menu_pinch_hold_seconds": (0.5, 5.0),
        "default_gesture_hold_seconds": (0.5, 5.0),
        "pinch_buffer_seconds": (0.3, 2.0),
    }
    try:
        content = GESTURES_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    explicit_hand_render_style = False
    legacy_hand_skin_effect = False
    for raw in content:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in ranges:
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                lo, hi = ranges[key]
                values[key] = min(hi, max(lo, number))
        # Old floating_device_mode entries are intentionally ignored; gesture
        # activation no longer depends on estimated depth or device distance.
        elif key in ("invert_scroll", "swap_drag_scroll_notches"):
            values[key] = value.lower() in ("true", "1", "yes", "on")
        elif key == "hand_skin_effect":
            values[key] = value.lower() in ("true", "1", "yes", "on")
            legacy_hand_skin_effect = True
        elif key == "hand_render_style" and value.lower() in HAND_RENDER_STYLES:
            values[key] = value.lower()
            explicit_hand_render_style = True
        elif key == "cursor_lock_priority" and value.lower() in ("left", "right"):
            values[key] = value.lower()
        elif key in ("left_click_color", "right_click_color") and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            values[key] = value.lower()
    # A legacy config explicitly using hand_skin_effect but lacking the newer
    # three-way option keeps its old flat/skeleton interpretation.
    if legacy_hand_skin_effect and not explicit_hand_render_style:
        values.pop("hand_render_style", None)
    return values


def download(url: str, target: Path) -> None:
    print(f"Downloading {url}")
    temp_path = str(target) + ".part"
    urllib.request.urlretrieve(url, temp_path)
    os.replace(temp_path, target)


def ensure_models() -> None:
    if not HAND_MODEL.exists():
        download(HAND_URL, HAND_MODEL)


def lm_xyz(landmarks, index):
    point = landmarks[index]
    return float(point.x), float(point.y), float(getattr(point, "z", 0.0))


def distance3d(landmarks, first, second):
    ax, ay, az = lm_xyz(landmarks, first)
    bx, by, bz = lm_xyz(landmarks, second)
    return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)


def finger_straightness(landmarks, mcp, pip, dip, tip):
    """1 means almost straight; smaller values mean the finger is curled."""
    direct = distance3d(landmarks, mcp, tip)
    chain = distance3d(landmarks, mcp, pip) + distance3d(landmarks, pip, dip) + distance3d(landmarks, dip, tip)
    return direct / max(1e-6, chain)


def _finger_straightness_values(landmarks):
    return [finger_straightness(landmarks, *chain) for chain in
            ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))]


def is_fist_or_grip_pose(landmarks, edge_tolerant=False):
    """Fist/grip pose; relax one landmark threshold near screen edges."""
    if landmarks is None or len(landmarks) < 21:
        return False
    straight = _finger_straightness_values(landmarks)
    # Keep a pointing index finger from being mistaken for a grip. Allow one noisy
    # landmark chain among the remaining fingers because hand tracking is imperfect.
    # Slightly wider margins tolerate ordinary half-curled knuckles and the extra
    # landmark noise seen in a diagonal/side-on grip, without accepting a straight index.
    index_limit = 0.87 if edge_tolerant else 0.83
    other_limit = 0.90 if edge_tolerant else 0.86
    fingers_closed = straight[0] <= index_limit and sum(value <= other_limit for value in straight[1:]) >= 2
    # A thumb merely lying alongside the fist is common in a grip. Reject only a
    # clearly extended thumb (as in thumbs-up/down), not a moderate sideways offset.
    return fingers_closed and not thumb_sticking_out(landmarks, threshold=-0.55)


def fingers_closed_pose(landmarks, edge_tolerant=False):
    """The four fingers folded, whatever the thumb does (fist, thumbs up/down)."""
    if landmarks is None or len(landmarks) < 21:
        return False
    straight = _finger_straightness_values(landmarks)
    index_limit = 0.81 if edge_tolerant else 0.79
    other_limit = 0.87 if edge_tolerant else 0.83
    return straight[0] <= index_limit and sum(value <= other_limit for value in straight[1:]) >= 2


def is_open_hand_pose(landmarks, edge_tolerant=False):
    """Release pose used to complete the quick fist-then-open left click."""
    if landmarks is None or len(landmarks) < 21:
        return False
    straight = _finger_straightness_values(landmarks)
    threshold = 0.77 if edge_tolerant else 0.81
    return sum(value >= threshold for value in straight) >= 3


def is_thumb_index_menu_pinch(landmarks, edge_tolerant=False):
    """Robust thumb/index pinch (index only) with at least two other fingers extended.

    The index finger naturally bends while making a pinch, so its straightness must not
    be a prerequisite. A closed fist still fails because it has fewer than two extended
    non-index fingers. This makes the timed Pinch Menu much less likely to keep restarting.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    # Only the index finger makes the built-in pinch: thumb-to-middle/ring/pinky
    # pinches are reserved for custom gestures.
    if not thumb_pinches_index([distance3d(landmarks, 4, tip) for tip in (8, 12, 16, 20)]):
        return False
    palm = max(1e-5, distance3d(landmarks, 0, 9))
    pinch_gap = distance3d(landmarks, 4, 8) / palm
    straight = _finger_straightness_values(landmarks)
    return should_recognize_menu_pinch(normalized_gap=pinch_gap,
                                       finger_straightness=straight,
                                       edge_tolerant=edge_tolerant)


class Tracker:
    """Low-latency, hand-only MediaPipe tracker. No body-pose model is loaded."""
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            # Do not lower MediaPipe's global thresholds to solve the side-on case:
            # that was allowing background false positives to become phantom hands.
            # Side-on views are handled by a narrow, high-confidence geometry exception.
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.58,
            min_tracking_confidence=0.55,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.camera_index = 0
        self.cap = self._open_camera(self.camera_index)
        if self.cap is None:
            self.hands.close()
            raise RuntimeError("Could not open camera. Check camera permissions and the camera index.")
        self._reopen_failures = 0
        self.running = True
        self.hand_tracking_enabled = True
        self.hands_data = []
        self._candidate_streak = {"left": 0, "right": 0}
        self._candidate_center = {"left": None, "right": None}
        self._candidate_misses = {"left": 0, "right": 0}
        # Never draw cached landmarks after MediaPipe has lost a hand: stale landmarks
        # were the source of visible phantom hands. A valid side-on hand is handled by
        # the geometry exception below and must reconfirm after a dropout.
        self.frame_id = 0
        self.timestamp_ms = 0
        self.latest_frame = None  # mirrored 480x270 BGR frame shared read-only with optional face welcome
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # A camera that went through a system sleep often never delivers again on the old
    # handle: grab() fails forever, blocks for its driver timeout, or keeps returning the
    # last (frozen) frame. These limits decide when the handle is released and reopened.
    CAMERA_STALL_SECONDS = 2.0        # failed or frozen frames for this long -> reopen
    CAMERA_RESUME_GAP_SECONDS = 3.0   # one loop pass this long (sleep/resume) -> reopen
    CAMERA_REOPEN_MAX_DELAY = 30.0

    @staticmethod
    def _open_camera(index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _clear_hand_state(self):
        with self.lock:
            self.hands_data = []
        self._candidate_streak = {"left": 0, "right": 0}
        self._candidate_center = {"left": None, "right": None}
        self._candidate_misses = {"left": 0, "right": 0}

    def _reopen_camera(self, reason):
        """Release the camera and open it again; True once it delivers a frame."""
        print(f"Camera {reason}; reopening it.", flush=True)
        self._clear_hand_state()
        old, self.cap = self.cap, None
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        time.sleep(0.5)  # give the driver a moment to settle / re-enumerate after resume
        # The original camera first. Only after repeated failures try the next few
        # indexes too, because a device that re-enumerated can come back under another.
        indexes = [self.camera_index]
        if self._reopen_failures >= 2:
            indexes += [i for i in range(4) if i != self.camera_index]
        for index in indexes:
            if not self.running:
                return False
            cap = self._open_camera(index)
            if cap is None:
                continue
            delivered = False
            for _ in range(15):  # the first frames after a resume can still fail briefly
                if cap.grab():
                    delivered = True
                    break
                time.sleep(0.1)
            if delivered and self.running:
                self.cap = cap
                self._reopen_failures = 0
                print(f"Camera {index} is delivering frames again.", flush=True)
                return True
            cap.release()
        self._reopen_failures += 1
        return False

    def _loop(self):
        last_frame_time = 0.0
        frame_interval = 1.0 / 18.0
        # Run inference at a modest active cadence and downshift further when no hands
        # are present. This substantially reduces sustained CPU use on software-rendered
        # Linux desktops while staying responsive enough for gesture interaction.
        idle_frame_interval = 1.0 / 8.0
        # Paused hand tracking only needs occasional frames for the optional face welcome.
        paused_frame_interval = 1.0 / 2.0
        idle_after_seconds = 1.0
        last_hand_seen = 0.0
        last_wall, last_mono = time.time(), time.monotonic()
        failing_since = None        # first failed grab/retrieve of the current run
        frozen_since = None         # first repeat of an identical frame
        last_sample = None
        reopen_delay, next_reopen_at = 1.0, 0.0
        gap_check_armed = False

        def reopen(reason, force=False):
            nonlocal failing_since, frozen_since, last_sample, reopen_delay, next_reopen_at
            nonlocal last_wall, last_mono, last_frame_time
            if not force and time.monotonic() < next_reopen_at:
                return
            if self._reopen_camera(reason):
                reopen_delay = 1.0
            else:
                reopen_delay = min(self.CAMERA_REOPEN_MAX_DELAY, reopen_delay*2.0)
            next_reopen_at = time.monotonic() + reopen_delay
            failing_since = frozen_since = last_sample = None
            last_frame_time = 0.0
            last_wall, last_mono = time.time(), time.monotonic()

        while self.running:
            # Active inference is capped at 18 FPS to reduce sustained CPU load. A much
            # longer pass means
            # the PC slept (the wall clock jumps across a suspend; the monotonic clock may
            # or may not, depending on the OS) or the driver stalled in grab(): either way
            # the old camera handle is not trustworthy any more.
            wall, mono = time.time(), time.monotonic()
            # Not armed until a frame has been processed: the very first pass includes
            # model start-up, which can be slow on a modest machine.
            if gap_check_armed and (wall - last_wall > self.CAMERA_RESUME_GAP_SECONDS or
                                    mono - last_mono > self.CAMERA_RESUME_GAP_SECONDS):
                reopen("paused for %.0f s (sleep/resume?)" % max(wall - last_wall, mono - last_mono), force=True)
                continue
            last_wall, last_mono = wall, mono
            if self.cap is None:
                # Still waiting for the camera to come back after a failed reopen.
                reopen("is unavailable")
                time.sleep(0.2)
                continue
            # grab() waits for the next camera frame without decoding it; frames that
            # are skipped below are never decoded/converted, which saves CPU.
            ok = self.cap.grab()
            frame = None
            now = time.monotonic()
            if ok:
                if not self.hand_tracking_enabled:
                    interval = paused_frame_interval
                elif now - last_hand_seen > idle_after_seconds:
                    interval = idle_frame_interval
                else:
                    interval = frame_interval
                elapsed = now - last_frame_time
                if elapsed < interval:
                    # Do not busy-spin while waiting for the next inference slot.
                    # Camera reads can return much faster than the inference cadence;
                    # without yielding here this loop can consume a full CPU core even
                    # when gesture handling is disabled. Use a longer bounded sleep to
                    # avoid a high-frequency grab/sleep spin on cameras whose grab() does
                    # not block for a fresh frame; shutdown remains responsive.
                    time.sleep(min(interval - elapsed, 0.02))
                    continue
                ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                # A camera read failure must not leave the previous frame's hands on screen.
                self._clear_hand_state()
                if failing_since is None:
                    failing_since = now
                elif now - failing_since > self.CAMERA_STALL_SECONDS:
                    reopen("stopped delivering frames")
                    continue
                time.sleep(0.02)
                continue
            failing_since = None
            # Real camera frames always carry sensor noise; the exact same picture for
            # seconds means the driver is replaying its last buffer (common after resume).
            sample = frame[::24, ::24]
            if last_sample is not None and np.array_equal(sample, last_sample):
                if frozen_since is None:
                    frozen_since = now
                elif now - frozen_since > self.CAMERA_STALL_SECONDS:
                    reopen("is returning a frozen picture")
                    continue
            else:
                frozen_since = None
                if reopen_delay > 1.0 and now >= next_reopen_at:
                    reopen_delay = 1.0
            last_sample = sample.copy()
            last_frame_time = now
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            try:
                found = []
                # Speech commands can pause hand inference while the camera feed remains
                # available for the optional local face-welcome feature. This saves CPU and
                # prevents any hand-driven input while paused.
                if self.hand_tracking_enabled:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                    # MediaPipe VIDEO timestamps must be monotonic and should reflect
                    # the actual frame cadence; a fixed 33 ms step becomes inaccurate
                    # now that inference deliberately runs below 30 FPS.
                    self.timestamp_ms = max(self.timestamp_ms + 1, int(now * 1000.0))
                    result = self.hands.detect_for_video(image, self.timestamp_ms)
                    raw_seen = set()
                    for index, landmarks in enumerate(result.hand_landmarks):
                        model_label, score = "right", 0.0
                        if result.handedness and index < len(result.handedness) and result.handedness[index]:
                            model_label = result.handedness[index][0].category_name.lower()
                            score = float(result.handedness[index][0].score)
                        # On this mirrored webcam path MediaPipe's handedness label has been
                        # observed opposite to the physical hand. Normalize once here so
                        # priority, colour and gesture owner all refer to physical hands.
                        label = {"left": "right", "right": "left"}.get(model_label, model_label)
                        points = list(landmarks)
                        if label not in ("left", "right") or not is_plausible_hand_landmarks(points, score):
                            continue
                        raw_seen.add(label)
                        xs = [float(point.x) for point in points[:21]]
                        ys = [float(point.y) for point in points[:21]]
                        center = ((min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5)
                        previous = self._candidate_center.get(label)
                        if (previous is not None and self._candidate_misses[label] <= 1 and
                                math.hypot(center[0]-previous[0], center[1]-previous[1]) <= 0.12):
                            self._candidate_streak[label] = min(3, self._candidate_streak[label]+1)
                        else:
                            self._candidate_streak[label] = 1
                        self._candidate_center[label] = center
                        self._candidate_misses[label] = 0
                        # Three consistent detections are required before a skeleton can
                        # appear. A single jittering/background hit must never be rendered.
                        if self._candidate_streak[label] >= 3:
                            found.append((label, points, score))
                    if raw_seen:
                        last_hand_seen = time.monotonic()
                    for side in ("left", "right"):
                        if side not in raw_seen:
                            # Remember the detection history across one dropped frame, but
                            # never reuse its landmarks for display; this smooths side-on
                            # reacquisition without showing a ghost hand during the gap.
                            self._candidate_misses[side] += 1
                            if self._candidate_misses[side] > 1:
                                self._candidate_streak[side] = 0
                                self._candidate_center[side] = None
                                self._candidate_misses[side] = 0
                with self.lock:
                    self.hands_data = found
                    self.latest_frame = frame.copy()
                    self.frame_id += 1
                if not gap_check_armed:
                    gap_check_armed = True
                    last_wall, last_mono = time.time(), time.monotonic()
            except Exception as exc:
                # Keep the camera thread alive, but never preserve visible hands from a
                # failed inference frame. Clear candidate history so recovery is fresh.
                with self.lock:
                    self.hands_data = []
                self._candidate_streak = {"left": 0, "right": 0}
                self._candidate_center = {"left": None, "right": None}
                self._candidate_misses = {"left": 0, "right": 0}
                print("Hand tracking frame failed:", exc)
                time.sleep(0.005)

    def set_hand_tracking_enabled(self, enabled: bool):
        """Enable/disable hand inference without stopping camera frames for face welcome."""
        self.hand_tracking_enabled = bool(enabled)
        if not enabled:
            # Clear pending candidate history too; resuming must not inherit a pre-pause
            # detection streak or briefly revive the last skeleton.
            self._candidate_streak = {"left": 0, "right": 0}
            self._candidate_center = {"left": None, "right": None}
            self._candidate_misses = {"left": 0, "right": 0}
            with self.lock:
                self.hands_data = []

    def snapshot(self):
        with self.lock:
            return list(self.hands_data), self.frame_id

    def snapshot_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.2)
        if self.cap is not None:
            self.cap.release()
        self.hands.close()


class SpeechEvents(QObject):
    """Qt signals safely bridge worker threads to the GUI thread."""
    final = Signal(str)
    partial = Signal(str)
    status = Signal(str)
    face_status = Signal(str)


class UpdateCheckSignals(QObject):
    """Thread-safe progress/completion signals for background updater operations."""
    finished = Signal(str, str)
    action_finished = Signal(str, str)
    progress = Signal(int, int)


class SearchHandOverlay(QWidget):
    """Click-through overlay that keeps hands/TLI visible above the search launcher."""
    def __init__(self, owner):
        super().__init__(None)
        self.owner = owner
        self.setWindowTitle("Spatial Desktop Hand Indicators")
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._last_raise = 0.0
        self._last_native_raise = 0.0
        self._native_style_ready = False
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(24)
        self.sync_timer.timeout.connect(self.sync_to_launcher)
        self.sync_timer.start()

    def sync_to_launcher(self):
        owner = self.owner
        dialog = owner.search_dialog
        should_show = bool(dialog is not None and dialog.isVisible() and
                           not owner.hand_tracking_paused)
        # Poll slowly while Search is closed; the launcher also triggers an immediate
        # sync when it opens, so the fast cadence resumes without a visible delay.
        wanted_interval = 24 if should_show else 150
        if self.sync_timer.interval() != wanted_interval:
            self.sync_timer.setInterval(wanted_interval)
        if not should_show:
            if self.isVisible():
                self.hide()
            return
        target_geometry = owner.geometry()
        if self.geometry() != target_geometry:
            self.setGeometry(target_geometry)
        if not self.isVisible():
            self.show()
        now = time.monotonic()
        # Qt's topmost dialogs can be raised after this transparent Tool window. Reassert
        # z-order frequently without activating the overlay, so the skeleton/TLI/menu stay
        # visible over the search launcher while all input still passes through to it.
        if now-self._last_raise > 0.12:
            self.raise_()
            self._last_raise = now
        if sys.platform.startswith("win") and now-self._last_native_raise > 0.12:
            try:
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                if not self._native_style_ready:
                    style = user32.GetWindowLongW(hwnd, -20)
                    style |= 0x00000020 | 0x00080000 | 0x00000080 | 0x08000000  # transparent/layered/tool/noactivate
                    user32.SetWindowLongW(hwnd, -20, style)
                    self._native_style_ready = True
                # HWND_TOPMOST (-1), SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_SHOWWINDOW.
                user32.SetWindowPos(wintypes.HWND(hwnd), wintypes.HWND(-1), 0, 0, 0, 0,
                                    0x0001 | 0x0002 | 0x0010 | 0x0040)
                self._last_native_raise = now
            except Exception:
                pass
        self.update()

    def paintEvent(self, event):
        """Paint one frame; the painter is always released, even if drawing fails.

        If an exception escaped mid-frame, the QPainter stayed active on the window,
        and every later repaint then failed ("Painter not active" / "one painter at
        a time"), leaving stale frames stacked on screen. Re-entrant paints (possible
        while native windows are being moved) are skipped for the same reason.
        """
        if getattr(self, "_in_paint_event", False):
            return
        self._in_paint_event = True
        self._frame_painter = None
        try:
            self._paint_frame(event)
        except Exception:
            if not getattr(self, "_paint_error_reported", False):
                self._paint_error_reported = True
                import traceback
                print("Spatial Desktop: a frame failed to paint (later failures are not repeated):")
                traceback.print_exc()
        finally:
            painter = self._frame_painter
            self._frame_painter = None
            if painter is not None and painter.isActive():
                painter.end()
            self._in_paint_event = False

    def _paint_frame(self, _event):
        owner = self.owner
        if not (owner.search_dialog is not None and owner.search_dialog.isVisible()):
            return
        # Native video windows can temporarily invalidate the transparent overlay surface
        # while Windows is moving/resizing them. Skip that frame instead of issuing
        # follow-on drawing calls against an inactive painter.
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter()
        self._frame_painter = painter
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        now = time.monotonic()
        # The completion ring fades out while the Pinch Menu fades in/out. Paint the
        # menu before the hands so the hand and TLI both appear on top of it.
        for side in ("right", "left"):
            owner._paint_gesture_timer_fade(painter, side, now)
            owner._paint_pinch_menu(painter, side, now)
        # This overlay is the sole hand-visual owner while Search is visible; the main
        # window skips the identical layer to avoid double-painted ghost silhouettes.
        for side in ("left", "right"):
            points = owner.hand_positions.get(side, [])
            if not points:
                continue
            color = QColor(owner.left_click_color if side == "left" else owner.right_click_color)
            owner._paint_hand_skeleton(painter, points, side)
            cursor = owner.side_cursor_positions.get(side)
            if cursor is None:
                # The current frame has landmarks but no eligible cursor (for example,
                # a paused hand). Do not synthesize a second/fallback TLI in this overlay.
                continue
            cursor = owner._visual_tli_position(side, cursor, points)
            # TLI is drawn once, in the final loop below, so the Search overlay never
            # stacks duplicate rings while a hold/pinch timer is active.
            # Mirror only the Pinch Menu timer while Search covers the main overlay.
            # Right-fist movement scrolls directly and intentionally has no fist timer ring.
            started = owner.menu_pinch_started.get(side) if not owner.fist_action_menu_open.get(side) else None
            mode = "pinch"
            if owner.menu_pinch_release_candidate_since.get(side) is not None:
                started = None
            peace_started = owner.peace_hold_started.get(side)
            if peace_started is not None and not owner.peace_toggle_latched.get(side, False):
                ring = QRectF(cursor.x()-31, cursor.y()-31, 62, 62)
                elapsed = max(0.0, now-peace_started)
                progress = min(1.0, max(0.0, elapsed/max(0.01, owner.peace_toggle_hold_seconds)))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(255, 220, 110, 100), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(255, 220, 110, 245), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(255, 242, 205, 245))
                painter.setFont(QFont("Sans", 8, QFont.DemiBold))
                remaining = max(0.0, owner.peace_toggle_hold_seconds-elapsed)
                painter.drawText(QRectF(cursor.x()-22, cursor.y()+29, 44, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")
            if started is not None:
                elapsed = owner._menu_pinch_elapsed_seconds(side, now)
                delay = owner.pinch_buffer_seconds
                duration = owner.menu_pinch_hold_seconds
                total_duration = delay + duration
                # The confirmation buffer is deliberately unfilled; the ring advances
                # only once the pinch buffer has elapsed.
                progress = min(1.0, max(0.0, elapsed/max(0.01, duration)))
                buffer_remaining = max(0.0, delay-(now-started))
                remaining = buffer_remaining if buffer_remaining > 0.0 else max(0.0, duration-elapsed)
                ring = QRectF(cursor.x()-41, cursor.y()-41, 82, 82)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 95), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 250), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(245, 251, 255, 245))
                painter.setFont(QFont("Sans", 8, QFont.DemiBold))
                painter.drawText(QRectF(cursor.x()-22, cursor.y()+23, 44, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")
        # Repaint the TLI last so it remains visible over both launcher and menu buttons.
        for side in ("right", "left"):
            points = owner.hand_positions.get(side, [])
            if not points:
                continue
            cursor = owner.side_cursor_positions.get(side)
            if cursor is None:
                continue
            cursor = owner._visual_tli_position(side, cursor, points)
            color = QColor(owner.left_click_color if side == "left" else owner.right_click_color)
            if owner.fist_action_menu_open.get(side):
                original = owner.fist_origin_points.get(side)
                if original is not None:
                    painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 235), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(original, 13, 13)
                    painter.drawLine(QPointF(original.x()-6, original.y()), QPointF(original.x()+6, original.y()))
                    painter.drawLine(QPointF(original.x(), original.y()-6), QPointF(original.x(), original.y()+6))
                # The common bright TLI remains the single visible/interactive cursor.
            engaged = side in owner.active_push_sides or (owner.click_side == side and now < owner.click_flash_until)
            owner._paint_tli(painter, cursor, side, engaged)
        painter.end()


class TliCalibrationWindow(QWidget):
    """Dedicated blank target for TLI calibration; gesture input is modal at the app level."""
    cancel_requested = Signal()

    def __init__(self, side):
        super().__init__(None)
        self.side = side
        self.progress = 0
        self.sample_count = 12
        self.phase_text = "Release the pinch used to open this window"
        self.setWindowTitle(f"{side.title()} TLI Calibration")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.live_overlay = {"hands": [], "tli": []}
        self.setFixedSize(440, 330)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRectF(0, 0, 1280, 800).toRect()
        self.move(available.center().x()-self.width()//2,
                  available.center().y()-self.height()//2)

    def pinch_area_global_rect(self):
        """The large, empty interior area that the thumb/index pinch must occupy."""
        area = QRectF(26.0, 76.0, self.width()-52.0, self.height()-116.0)
        top_left = self.mapToGlobal(QPoint(round(area.left()), round(area.top())))
        return QRectF(float(top_left.x()), float(top_left.y()), area.width(), area.height())

    def set_phase(self, text, progress=None):
        self.phase_text = str(text)
        if progress is not None:
            self.progress = min(self.sample_count, max(0, int(progress)))
        self.update()

    def set_hand_overlay(self, hands, tli_points):
        """Paint live tracked hands above this separate topmost calibration window."""
        self.live_overlay = {"hands": list(hands or []), "tli": list(tli_points or [])}
        self.update()

    def paintEvent(self, event):
        """Paint one frame; the painter is always released, even if drawing fails.

        If an exception escaped mid-frame, the QPainter stayed active on the window,
        and every later repaint then failed ("Painter not active" / "one painter at
        a time"), leaving stale frames stacked on screen. Re-entrant paints (possible
        while native windows are being moved) are skipped for the same reason.
        """
        if getattr(self, "_in_paint_event", False):
            return
        self._in_paint_event = True
        self._frame_painter = None
        try:
            self._paint_frame(event)
        except Exception:
            if not getattr(self, "_paint_error_reported", False):
                self._paint_error_reported = True
                import traceback
                print("Spatial Desktop: a frame failed to paint (later failures are not repeated):")
                traceback.print_exc()
        finally:
            painter = self._frame_painter
            self._frame_painter = None
            if painter is not None and painter.isActive():
                painter.end()
            self._in_paint_event = False

    def _paint_frame(self, _event):
        # Native video windows can temporarily invalidate the transparent overlay surface
        # while Windows is moving/resizing them. Skip that frame instead of issuing
        # follow-on drawing calls against an inactive painter.
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter()
        self._frame_painter = painter
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        whole = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        # Draw only the outline plus compact header/footer plates. The target interior
        # stays transparent, allowing the live skeleton/skin hand overlay beneath this
        # topmost window to remain visible while the user positions the pinch inside it.
        painter.setPen(QPen(QColor(90, 235, 255, 245), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(whole, 22, 22)
        painter.setPen(QPen(QColor(105, 240, 255, 170), 1.0))
        painter.setBrush(QBrush(QColor(4, 15, 31, 216)))
        painter.drawRoundedRect(QRectF(4, 4, self.width()-8, 67), 17, 17)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(4, 15, 31, 216)))
        painter.drawRoundedRect(QRectF(4, self.height()-42, self.width()-8, 38), 13, 13)

        painter.setPen(QColor(235, 252, 255, 250))
        painter.setFont(QFont("Sans", 14, QFont.Bold))
        painter.drawText(QRectF(24, 17, self.width()-48, 27), Qt.AlignCenter,
                         f"{self.side.upper()} TLI CALIBRATION")
        painter.setPen(QColor(170, 220, 235, 235))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(QRectF(20, 45, self.width()-40, 23), Qt.AlignCenter,
                         "Pinch thumb + index together inside this window")

        # Leave the majority of the window visually blank to provide a clear target,
        # without extra controls that can accidentally be activated by hand gestures.
        area = QRectF(26, 76, self.width()-52, self.height()-116)
        painter.setPen(QPen(QColor(80, 225, 245, 205), 1.8, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(area, 14, 14)
        # A faint target crosshair gives a visual centre without obscuring the hand.
        target_centre = area.center()
        painter.setPen(QPen(QColor(100, 240, 245, 105), 1.0, Qt.DotLine))
        painter.drawLine(QPointF(target_centre.x()-17, target_centre.y()),
                         QPointF(target_centre.x()+17, target_centre.y()))
        painter.drawLine(QPointF(target_centre.x(), target_centre.y()-17),
                         QPointF(target_centre.x(), target_centre.y()+17))

        # The target is a separate topmost window, so explicitly redraw the current
        # tracked hand inside it. This avoids the hand/TLI being hidden beneath the target
        # border on platforms that do not composite sibling translucent windows together.
        edges = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                 (13,17),(17,18),(18,19),(19,20),(0,17),(5,9),(9,13),(13,17))
        for points, color in self.live_overlay.get("hands", []):
            if len(points) < 21:
                continue
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 225),
                                3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for a, b in edges:
                painter.drawLine(points[a], points[b])
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 245)))
            for point in points:
                painter.drawEllipse(point, 3.2, 3.2)
        for point, color in self.live_overlay.get("tli", []):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
            painter.drawEllipse(point, 25, 25)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 100)))
            painter.drawEllipse(point, 16, 16)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(245, 255, 255, 255), 2.6))
            painter.drawEllipse(point, 11, 11)
            painter.drawLine(QPointF(point.x()-5, point.y()), QPointF(point.x()+5, point.y()))
            painter.drawLine(QPointF(point.x(), point.y()-5), QPointF(point.x(), point.y()+5))

        painter.setPen(QColor(192, 222, 232, 230))
        painter.setFont(QFont("Sans", 8))
        painter.drawText(QRectF(20, self.height()-33, self.width()-40, 15), Qt.AlignCenter,
                         self.phase_text)
        track = QRectF(82, self.height()-14, self.width()-164, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(45, 82, 101, 240)))
        painter.drawRoundedRect(track, 2.5, 2.5)
        ratio = self.progress / max(1, self.sample_count)
        progress_rect = QRectF(track.left(), track.top(), track.width()*ratio, track.height())
        painter.setBrush(QBrush(QColor(65, 245, 195, 245)))
        painter.drawRoundedRect(progress_rect, 2.5, 2.5)
        painter.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancel_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class HandCalibrationWizard(QWidget):
    """First-run onboarding and camera/palm/proximity calibration surface.

    The centre is refreshed with live tracked landmarks by the owner, so the user's hands
    remain visible above the wizard card rather than being hidden behind a separate window.
    """
    action_requested = Signal(str)

    STEPS = {
        0: ("HAND TRACKING SETUP", "Calibrate the screen anchor, hand size, and two-hand proximity.",
            "You can rerun this guide at any time from Settings → Hands."),
        1: ("1 · SCREEN TOUCH ANCHOR", "Extend an arm and gently touch your index fingertip", 
            "to the screen just below the camera, then capture the pose."),
        2: ("2 · PALM SIZE REFERENCE", "Keep one or both hands in front of the camera.",
            "Tilt your wrist up so the palm faces the camera, then capture."),
        3: ("3 · TWO-HAND PROXIMITY", "Show both hands and gently bring fingertips together.",
            "Hold them near contact and capture their relative distance."),
        4: ("CALIBRATION COMPLETE", "Your hand and proximity references have been saved.",
            "You can rerun this calibration from Settings → Hands."),
    }

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Spatial Desktop · Hand Setup")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedSize(700, 500)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRectF(0, 0, 1280, 800).toRect()
        self.move(available.center().x()-self.width()//2,
                  available.center().y()-self.height()//2)
        self.stage = 0
        self.progress = 0
        self.sample_count = 20
        self.capturing = False
        self.live_overlay = {"hands": [], "tli": []}
        self._primary_rect = QRectF(0, 0, 0, 0)
        self._secondary_rect = QRectF(0, 0, 0, 0)

    def set_stage(self, stage):
        self.stage = max(0, min(4, int(stage)))
        self.progress = 0
        self.capturing = False
        self.update()

    def set_capture_progress(self, progress, capturing=True):
        self.progress = min(self.sample_count, max(0, int(progress)))
        self.capturing = bool(capturing)
        self.update()

    def set_live_hand_overlay(self, hands, tli_points):
        self.live_overlay = {"hands": list(hands or []), "tli": list(tli_points or [])}
        self.update()

    def _button_rectangles(self):
        y = self.height()-72
        self._primary_rect = QRectF(self.width()/2-174, y, 224, 44)
        self._secondary_rect = QRectF(self.width()/2+64, y, 110, 44)
        return self._primary_rect, self._secondary_rect

    def paintEvent(self, event):
        """Paint one frame; the painter is always released, even if drawing fails.

        If an exception escaped mid-frame, the QPainter stayed active on the window,
        and every later repaint then failed ("Painter not active" / "one painter at
        a time"), leaving stale frames stacked on screen. Re-entrant paints (possible
        while native windows are being moved) are skipped for the same reason.
        """
        if getattr(self, "_in_paint_event", False):
            return
        self._in_paint_event = True
        self._frame_painter = None
        try:
            self._paint_frame(event)
        except Exception:
            if not getattr(self, "_paint_error_reported", False):
                self._paint_error_reported = True
                import traceback
                print("Spatial Desktop: a frame failed to paint (later failures are not repeated):")
                traceback.print_exc()
        finally:
            painter = self._frame_painter
            self._frame_painter = None
            if painter is not None and painter.isActive():
                painter.end()
            self._in_paint_event = False

    def _paint_frame(self, _event):
        # Native video windows can temporarily invalidate the transparent overlay surface
        # while Windows is moving/resizing them. Skip that frame instead of issuing
        # follow-on drawing calls against an inactive painter.
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter()
        self._frame_painter = painter
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        whole = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        painter.setPen(QPen(QColor(80, 235, 250, 235), 2.3))
        painter.setBrush(QBrush(QColor(4, 15, 28, 178)))
        painter.drawRoundedRect(whole, 24, 24)
        painter.setPen(QPen(QColor(115, 235, 250, 170), 1.1))
        painter.setBrush(QBrush(QColor(4, 17, 32, 218)))
        painter.drawRoundedRect(QRectF(8, 8, self.width()-16, 124), 18, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(4, 17, 32, 230)))
        painter.drawRoundedRect(QRectF(8, self.height()-91, self.width()-16, 80), 17, 17)

        title, line1, line2 = self.STEPS[self.stage]
        painter.setPen(QColor(130, 247, 255, 250))
        painter.setFont(QFont("Sans", 15, QFont.Bold))
        painter.drawText(QRectF(25, 20, self.width()-50, 30), Qt.AlignCenter, title)
        painter.setPen(QColor(235, 249, 255, 250))
        painter.setFont(QFont("Sans", 10))
        painter.drawText(QRectF(28, 58, self.width()-56, 26), Qt.AlignCenter, line1)
        painter.setPen(QColor(188, 224, 236, 245))
        painter.drawText(QRectF(28, 86, self.width()-56, 24), Qt.AlignCenter, line2)

        target = QRectF(30, 150, self.width()-60, self.height()-250)
        painter.setPen(QPen(QColor(75, 228, 246, 210), 1.7, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(target, 18, 18)
        centre = target.center()
        painter.setPen(QPen(QColor(100, 245, 248, 95), 1.2, Qt.DotLine))
        painter.drawLine(QPointF(centre.x()-19, centre.y()), QPointF(centre.x()+19, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y()-19), QPointF(centre.x(), centre.y()+19))

        # Explicitly paint the latest hand landmarks into this topmost window, keeping them
        # visible even when the platform stacks separate translucent windows unexpectedly.
        edges = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                 (13,17),(17,18),(18,19),(19,20),(0,17),(5,9),(9,13),(13,17))
        for points, color in self.live_overlay.get("hands", []):
            if len(points) < 21:
                continue
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220),
                                3.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for a, b in edges:
                painter.drawLine(points[a], points[b])
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 240)))
            for point in points:
                painter.drawEllipse(point, 3.3, 3.3)
        for point, color in self.live_overlay.get("tli", []):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
            painter.drawEllipse(point, 27, 27)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 90)))
            painter.drawEllipse(point, 18, 18)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(240, 255, 255, 250), 2.4))
            painter.drawEllipse(point, 12, 12)
            painter.drawLine(QPointF(point.x()-5, point.y()), QPointF(point.x()+5, point.y()))
            painter.drawLine(QPointF(point.x(), point.y()-5), QPointF(point.x(), point.y()+5))

        if self.stage in (1, 2, 3):
            painter.setPen(QColor(180, 220, 232, 245))
            painter.setFont(QFont("Sans", 9))
            status = (f"Sampling {self.progress}/{self.sample_count} · keep still"
                      if self.capturing else "Press CAPTURE POSE when you are ready")
            painter.drawText(QRectF(25, self.height()-113, self.width()-50, 18), Qt.AlignCenter, status)
            track = QRectF(150, self.height()-96, self.width()-300, 5)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(40, 70, 88, 235)))
            painter.drawRoundedRect(track, 2.5, 2.5)
            painter.setBrush(QBrush(QColor(70, 245, 195, 245)))
            painter.drawRoundedRect(QRectF(track.left(), track.top(), track.width()*self.progress/self.sample_count,
                                           track.height()), 2.5, 2.5)

        primary, secondary = self._button_rectangles()
        if self.stage == 0:
            primary_label, secondary_label = "START CALIBRATION", "SKIP"
        elif self.stage == 4:
            primary_label, secondary_label = "FINISH", "RERUN LATER"
        else:
            primary_label = (f"SAMPLING {self.progress}/{self.sample_count}" if self.capturing
                             else "CAPTURE POSE")
            secondary_label = "CANCEL"
        for rect, label, primary_button in ((primary, primary_label, True), (secondary, secondary_label, False)):
            active = not (primary_button and self.capturing)
            fill = QColor(22, 115, 115, 238) if primary_button else QColor(19, 51, 71, 238)
            if not active:
                fill = QColor(18, 65, 72, 238)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(115, 242, 250, 225), 1.5))
            painter.drawRoundedRect(rect, 10, 10)
            painter.setPen(QColor(232, 253, 255, 245))
            painter.setFont(QFont("Sans", 9, QFont.DemiBold))
            painter.drawText(rect, Qt.AlignCenter, label)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            position = event.position()
            primary, secondary = self._button_rectangles()
            if primary.contains(position) and not self.capturing:
                action = {0: "start", 1: "capture", 2: "capture", 3: "capture", 4: "finish"}.get(self.stage)
                if action:
                    self.action_requested.emit(action)
                event.accept()
                return
            if secondary.contains(position):
                self.action_requested.emit("skip" if self.stage == 0 else "cancel")
                event.accept()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.action_requested.emit("cancel")
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and not self.capturing:
            action = {0: "start", 1: "capture", 2: "capture", 3: "capture", 4: "finish"}.get(self.stage)
            if action:
                self.action_requested.emit(action)
                event.accept()
                return
        super().keyPressEvent(event)


class VoiceReplySignals(QObject):
    finished = Signal(str, str)


class SpatialDesktop(QWidget):
    def __init__(self):
        super().__init__()
        self.voice_reply_signals = VoiceReplySignals(self)
        self.voice_reply_signals.finished.connect(self._on_general_voice_answer)
        self.setWindowTitle("Spatial Desktop")
        # Consistent tactile feedback for native buttons across settings and dialogs.
        self.setStyleSheet(self.styleSheet() + " QPushButton:pressed, QToolButton:pressed { background-color: rgba(75, 205, 235, 210); border: 2px solid #b9f5ff; padding-top: 2px; padding-left: 2px; }")
        # Let X11/XLibre's window manager own the fullscreen overlay's stacking
        # level. With override-redirect, an app opened later (notably osm-launcher,
        # which is also WindowStaysOnTopHint) can stay above the hands indefinitely.
        # A WM-managed, mouse-transparent, always-on-top Tool remains click-through
        # while sharing the same topmost layer as native launchers.
        overlay_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        self.setWindowFlags(overlay_flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        settings = load_gesture_settings()
        self.hand_visual_scale = settings["hand_visual_scale"]
        self.hand_recognition_threshold = float(settings.get("hand_recognition_threshold", 0.60))
        self.hand_contact_band_ratio = settings.get("hand_contact_band_ratio", 0.02)
        self.palm_attract_acceleration = settings.get("palm_attract_acceleration", 0.33)
        self.skin_tone_darkness = settings.get("skin_tone_darkness", -50.0)
        self.finger_thickness = settings.get("finger_thickness", 4.24)
        self.palm_width_scale = settings.get("palm_width_scale", 1.26)
        self.hand_skin_effect_enabled = bool(settings.get("hand_skin_effect", True))
        # Skeleton, flat (painted 2D) or 3d (rigged mesh). Older settings files only
        # have hand_skin_effect, which maps to flat/skeleton.
        style = settings.get("hand_render_style")
        self.hand_render_style = (style if style in HAND_RENDER_STYLES else
                                  ("flat" if self.hand_skin_effect_enabled else "skeleton"))
        self._hand_mesh_renderer = None
        self._hand_mesh_frame_cache = {}
        self._hand_mesh_notice_shown = False
        self.skin_use_skeleton_color = bool(settings.get("skin_use_skeleton_color", False))
        self.hand_transparency = float(settings.get("hand_transparency", 75.0))
        self.tli_glow_enabled = bool(settings.get("tli_glow_enabled", False))
        self.hand_tracking_sensitivity = settings["hand_tracking_sensitivity"]
        self.scroll_acceleration = settings.get("scroll_acceleration", 1.0)
        self.scroll_deadzone = settings.get("scroll_deadzone", 70.0)
        self.scroll_backdrop_opacity = float(settings.get("scroll_backdrop_opacity", 45.0))
        self.scroll_backdrop_hue = float(settings.get("scroll_backdrop_hue", 145.0))
        self.scroll_notch_hue = float(settings.get("scroll_notch_hue", 120.0))
        # Persistent "mouse detached" notice: full while the glass menu is open and for
        # 5 s after it closes or after the notification stack changes, then it fades.
        self.detached_notice_opacity = settings.get("detached_notice_opacity", 30.0)
        self.notice_fade_delay_seconds = 2.0
        # Long enough to feel gentle, while the status cards remain on screen at a
        # low resting opacity instead of disappearing completely.
        self.notice_fade_duration_seconds = 1.6
        self._notice_signature = None
        self._notice_full_since = 0.0
        self._notification_stack_full_since = 0.0
        self.custom_gesture_hold_seconds = settings.get("custom_gesture_hold_seconds", 0.5)
        self.custom_gesture_tolerance = settings.get("custom_gesture_tolerance", 0.18)
        # Recorded custom gestures (Settings -> Gestures).
        self.custom_gestures = load_custom_gestures(CUSTOM_GESTURES_FILE)
        self.custom_gesture_recording = None
        self.custom_gesture_message = ""
        self.custom_gesture_rings = {}
        self.custom_gesture_dropdown = None  # row index whose action list is open
        self.custom_gesture_key_capture = None  # gesture row awaiting a physical keyboard shortcut
        self.custom_gesture_triggers = {"left": HoldTrigger(), "right": HoldTrigger(), "overlap": HoldTrigger()}
        self.custom_gesture_live = {}
        self.cursor_smoothing = settings["cursor_smoothing"]
        self.tli_glow_intensity = settings.get("tli_glow_intensity", 1.0)
        self.tli_colour_shift_degrees = settings.get("tli_colour_shift_degrees", 0.0)
        self.tli_border_thickness = settings.get("tli_border_thickness", 5.5)
        self.tli_diameter = settings.get("tli_diameter", 30.0)
        self.skeleton_line_thickness = settings.get("skeleton_line_thickness", 1.5)
        self.click_debounce_seconds = settings["click_debounce_seconds"]
        self.default_gesture_hold_seconds = settings.get("default_gesture_hold_seconds", settings["menu_pinch_hold_seconds"])
        self.menu_pinch_hold_seconds = self.default_gesture_hold_seconds
        self.invert_scroll = settings.get("invert_scroll", True)
        self.swap_drag_scroll_notches = settings.get("swap_drag_scroll_notches", False)
        self.cursor_lock_priority = settings.get("cursor_lock_priority", "right")
        self.ui_preferences = load_ui_preferences()
        self.ui_font_scale = float(self.ui_preferences.get("ui_font_scale", 1.2))
        self.d1_ui_log = ["Engine panel ready."]
        self._last_d1_log_signature = None
        # Slider interaction intentionally moves at less than half the cursor speed for finer control.
        self.slider_drag_gain = 0.42
        self.left_click_color = QColor(settings["left_click_color"])
        self.right_click_color = QColor(settings["right_click_color"])
        # A double-click is two ordinary quick clicks at nearby points. Keep the first
        # target for the second click to compensate for natural hand drift.
        self.quick_grasp_double_click_window = 0.75
        if sys.platform.startswith("win"):
            try:
                self.quick_grasp_double_click_window = max(
                    0.25, float(ctypes.windll.user32.GetDoubleClickTime()) / 1000.0
                )
            except Exception:
                pass
        self.last_quick_grasp = {"left": None, "right": None}
        # Pinches have a separate click-pair tracker so two quick pinches can produce a
        # native double-click without relying on fist/grip click state.
        self.quick_pinch_double_click_window = self.quick_grasp_double_click_window
        self.quick_pinch_click_min_seconds = 0.06
        self.last_quick_pinch_click = {"left": None, "right": None}
        self.right_click_mode_enabled = False
        self.scroll_mode_enabled = False

        self.tracker = Tracker()
        self.speech_events = SpeechEvents()
        self.speech_events.final.connect(self.on_speech)
        self.speech_events.partial.connect(self.on_speech_partial)
        self.speech_events.status.connect(self.on_voice_status)
        self.speech = Speech(
            self.speech_events.final.emit,
            partial_callback=self.speech_events.partial.emit,
            status_callback=self.speech_events.status.emit,
        )
        self.tts = KokoroTTS(status_callback=self.speech_events.status.emit, activity_callback=self.speech.set_paused)
        self.face_enrollment_status = "Face welcome is opt-in"
        self.face_events_status = self.speech_events.face_status
        self.face_events_status.connect(self.on_face_status)
        self.face_welcome = FaceWelcomeService(
            frame_provider=self.tracker.snapshot_frame,
            status_callback=self.speech_events.face_status.emit,
            welcome_callback=self.tts.speak,
        )
        self.face_welcome.start()
        self.last_external_foreground_hwnd = None
        self.search_return_hwnd = None
        self.last_external_window_id = None  # X11 window ID for Linux when xdotool is present
        self.search_return_window_id = None
        self.stt_enabled = False
        # Voice input remains running while hand inference is paused, so speech can resume it.
        self.hand_tracking_paused = False
        self.speech_partial = ""
        self.last_speech = ""
        # Heard-bar feedback state: successful commands dismiss immediately; unknown
        # commands get a brief red shake, then a clear message and timed dismissal.
        self.heard_popup_mode = "transcript"
        self.heard_popup_message = ""
        self.heard_popup_generation = 0
        self.heard_popup_shake_until = 0.0
        self.search_dialog = None
        # Focused-field dictation writes stable Vosk words into the currently focused OS field.
        self.focus_dictation_enabled = False
        self.focus_dictation_paused = False
        self.focus_dictation_committed_words = []
        self.focus_dictation_last_partial = ""
        # Voice-driven selection modes: text selection uses a real held mouse button;
        # image lasso copies a screen rectangle into the system clipboard as an image.
        self.voice_selection_mode = None  # None | "text" | "image"
        self.voice_lasso_start = None     # global screen QPointF
        self.voice_selection_target = None
        self.status = f"Ready — offline Vosk STT · Kokoro TTS: {self.tts.status}"

        self.desktop_mode = "os"  # start in OS Desktop mode; the menu can switch to virtual/3D mode
        self._applied_desktop_mode = None
        self._applied_menu_state = None
        self._last_native_topmost_reassert = 0.0
        self._linux_x11_lib = None
        self._linux_x11_display = None
        self.menu_open = True
        self.hub_visible = True
        self.settings_open = False
        self.settings_page = "interaction"
        self.update_status = f"Installed version: {read_version(VERSION_FILE)}\nSelect UPDATE to check GitHub and download a newer release."
        self.update_check_running = False
        self.update_progress_current = 0
        self.update_progress_total = 0
        self.update_available = False
        self.update_download_path = None
        self.update_action_running = False
        self.update_action_mode = ""
        self.update_installed = False
        self.update_check_signals = UpdateCheckSignals(self)
        self.update_check_signals.finished.connect(self._on_update_check_finished)
        self.update_check_signals.action_finished.connect(self._on_update_action_finished)
        self.update_check_signals.progress.connect(self._on_update_progress)
        self._update_check_thread = None
        self.hub_pos = QPointF(0.5, 0.46)
        self.hub_scale = 1.0
        self.object_depth_scale = 1.0
        self.depth = 0.0
        self.target_depth = 0.0
        self.canvas_offset = QPointF(0.0, 0.0)
        self.menu_transition_started = 0.0
        self.menu_transition_duration = 0.22
        self.menu_input_block_until = 0.0
        # Settings-page transitions keep the glass panel mounted while its shape
        # gently flexes and the page contents cross-fade.
        self.settings_page_transition_started = 0.0
        self.settings_page_transition_duration = 0.78
        self.settings_page_transition_from_size = None
        self.settings_page_pending = None

        self.hand_positions = {"left": [], "right": []}
        self.tli_calibrations = load_tli_calibrations()
        self.tli_calibration_persisted = {side: self.tli_calibrations.get(side) is not None
                                          for side in ("left", "right")}
        self.hand_calibration = load_hand_calibration(HAND_CALIBRATION_FILE)
        self.hand_calibration_active = False
        self.hand_calibration_window = None
        self.hand_calibration_stage = 0
        self.hand_calibration_capture_requested = False
        self.hand_calibration_capture_samples = []
        self.hand_calibration_capture_count = 20
        self.hand_calibration_draft = default_calibration()
        self.tli_calibration_pending = {"left": False, "right": False}
        self.tli_calibration_wait_release = {"left": False, "right": False}
        self.tli_calibration_samples = {"left": [], "right": []}
        self.tli_calibration_sample_count = 12
        self.tli_calibration_active_side = None
        self.tli_calibration_window = None
        # Per-hand visual scale pivots: own wrist when apart, one shared pivot when the
        # hands touch, so touching hands stay together even when the meshes are shrunk.
        self.hand_scale_origins = {}
        # Shared contact adjustment: when the real hands are REALLY close, both virtual
        # hands are scaled about one shared pivot so physical contact stays contact.
        # Full effect at the calibrated touch gap; fades out over this many palm widths
        # beyond it. Smaller = only kicks in nearer contact (snappier); larger = gentler.
        # Loaded from gestures.txt above; set from Settings -> Hands -> CONTACT RANGE.
        self.hand_contact_weight = 0.0
        self.hand_contact_last_update = None
        self.mouse_detached_by_side = {"left": True, "right": False}
        self.peace_hold_started = {"left": None, "right": None}
        self.peace_toggle_latched = {"left": False, "right": False}
        # Per-hand gesture lock: a sustained thumbs-up toggles only the hand making it.
        # Camera tracking and the other hand remain active while one hand is disabled.
        self.gesture_tracking_enabled = {"left": True, "right": True}
        self.thumbs_up_hold_started = {"left": None, "right": None}
        self.thumbs_up_last_seen = {"left": None, "right": None}
        self.thumbs_up_toggle_latched = {"left": False, "right": False}
        self.gesture_tracking_toggle_hold_seconds = self.default_gesture_hold_seconds
        # The left hand starts detached. Show its banner immediately on startup;
        # _hand_tracking_notice_opacity applies the usual delay and fade timeout.
        self.hand_tracking_notice_text = "Left mouse detached"
        self.hand_tracking_notice_until = float("inf")
        # Shorter hold, with a visible countdown ring so the toggle feels responsive.
        self.peace_toggle_hold_seconds = self.default_gesture_hold_seconds
        self.hand_last_seen = {"left": 0.0, "right": 0.0}
        self.hand_missing_since = {"left": None, "right": None}
        self.tracking_loss_grace_seconds = 1.5
        # Release handling is separate from the Pinch Menu's continuous 0.8s confirmation hold.
        # Require a short stable open pose so scrolling stops promptly without reacting
        # to a single bad landmark frame.
        self.active_release_cooldown_seconds = 0.20
        self.right_click_context_active = {"left": False, "right": False}
        self.side_cursor_positions = {"left": None, "right": None}
        # Screen offset applied when drawing each hand so it moves with the smoothed TLI.
        self.hand_draw_offset = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        # Drawn-hand landmarks with fingertips the camera can't see folded in (see
        # hand_occlusion). Gestures and the TLI keep using hand_positions.
        self.hand_display_positions = {"left": None, "right": None}
        self._hand_tucker = TuckSmoother()
        # Keep raw, smoothed camera-mapped anchors separate from scaled interaction points.
        # This avoids feeding transformed coordinates back into the smoothing filter.
        self.smoothed_side_cursor_positions = {"left": None, "right": None}
        self.cursor_updated_at = {"left": 0.0, "right": 0.0}
        self.pointer_side = None
        self.last_os_cursor_pos = None
        self.active_push_sides = set()
        self.tli_overrides = {}
        self.click_flash_until = 0.0
        self.click_side = None
        self._glass_pressed_point = None
        self._glass_pressed_at = 0.0

        # Mutually exclusive per-hand gesture states; no depth-calibration state exists.
        sides = ("left", "right")
        self.gesture_timer_delay_seconds = 0.8  # FIXED soft gesture buffer: do not remove or shorten
        # Pinch Menu confirmation buffer before the circle timer starts; adjustable in
        # Settings -> Interaction (PINCH BUFFER). Fist holds keep the fixed buffer above.
        self.pinch_buffer_seconds = float(settings.get("pinch_buffer_seconds", 0.8))
        self.fist_quick_click_max_seconds = 0.65
        self.fist_started_at = {side: None for side in sides}
        self.fist_release_candidate_since = {side: None for side in sides}
        self.fist_open_candidate_since = {side: None for side in sides}
        self.fist_origin_points = {side: None for side in sides}
        self.fist_arm_points = {side: None for side in sides}
        self.fist_hold_armed = {side: False for side in sides}
        self.fist_action = {side: None for side in sides}
        self.fist_last_points = {side: None for side in sides}
        self.fist_start_controls = {side: None for side in sides}
        self.fist_slider_offset_x = {side: 0.0 for side in sides}
        self.slider_last_good_cursor = {side: None for side in sides}
        self.slider_cursor_jump_candidate = {side: None for side in sides}
        self.fist_scroll_remainder = {side: 0.0 for side in sides}
        # Stationary origin marker for recognised fist-drag scrolling.
        self.fist_scroll_markers = {side: None for side in sides}
        self.fist_mouse_down = {side: False for side in sides}
        self.fist_action_menu_open = {side: False for side in sides}
        self.fist_hovered_action = {side: None for side in sides}
        # Pinch Menu fades independently of the main glass menu. Closing snapshots its
        # geometry so it can fade out even after gesture state has been reset.
        self.pinch_menu_animation = {
            side: {"started_at": 0.0, "from_alpha": 1.0, "to_alpha": 1.0,
                   "duration": 0.20, "snapshot": None}
            for side in sides
        }
        self.gesture_timer_fade = {side: None for side in sides}
        # The opening timed pinch must be released before a second grab/pinch can select an item.
        self.pin_menu_ready_for_selection = {side: False for side in sides}
        self.pin_menu_selection_started_at = {side: None for side in sides}
        self.pin_menu_selection_kind = {side: None for side in sides}
        self.pin_menu_selection_option = {side: None for side in sides}
        self.pin_menu_selection_debounce_seconds = 0.10
        # Prevent the selection grip from immediately dismissing the native OS context menu.
        self.pinch_menu_dismiss_block_until = 0.0
        self.fist_target_window = {side: None for side in sides}
        self.fist_target_window_resolved = {side: False for side in sides}
        self.fist_started_on_taskbar = {side: False for side in sides}
        self.fist_start_hit_test = {side: None for side in sides}
        self.fist_shake_samples = {side: [] for side in sides}
        self.fist_hold_ready_notified = {side: False for side in sides}
        # Active window drag/scroll is owned by the hand that held the fist to start it.
        self.active_action = {side: None for side in sides}
        self.action_stop_wait_open = {side: False for side in sides}
        self.action_stop_open_since = {side: None for side in sides}
        self.menu_pinch_started = {side: None for side in sides}
        # Accumulate only time spent actually pinching; short release gaps and duplicate
        # pinch detections must not advance the circle timer.
        self.menu_pinch_accumulated_seconds = {side: 0.0 for side in sides}
        self.menu_pinch_pulse_started_at = {side: None for side in sides}
        self.menu_pinch_origin_points = {side: None for side in sides}
        self.menu_pinch_release_candidate_since = {side: None for side in sides}
        self.menu_pinch_click_fired = {side: False for side in sides}
        # Native click-and-hold drag armed when a pinch moves before the timer completes.
        self.pinch_drag_active = {side: False for side in sides}
        self.pinch_drag_origin_points = {side: None for side in sides}
        self.pinch_drag_last_points = {side: None for side in sides}
        self.pinch_drag_release_candidate_since = {side: None for side in sides}
        self.pinch_drag_start_distance = 10.0
        self.menu_pinch_latched = set()
        # After a fist, require a clearly open (non-pinched) hand before another timed
        # pinch can arm. This prevents fist/grip landmark flicker opening the Pinch Menu.
        self.fist_release_required_for_pinch = {side: False for side in sides}
        self.fist_pose_confirmation_frames = {side: 0 for side in sides}
        self.pinch_recovery_frames = {side: 0 for side in sides}
        self.fist_release_buffer_started_at = {side: None for side in sides}
        self.menu_pinch_sequence_gap_seconds = 0.80  # FIXED 0.8s soft gap for repeated pinch/re-arm
        # LEFT fist acts like a keyboard modifier: RIGHT pinch/grip is a click-only input
        # while held, independent of the Pinch Menu's repeated-pinch timer sequence.
        self.left_click_modifier_active = False
        # Per-hand status cards brighten for five seconds when their displayed state changes.
        self._status_card_signatures = {"left": None, "right": None}
        self._status_card_full_until = {"left": 0.0, "right": 0.0}
        self.left_modifier_fist_frames = 0
        self.left_modifier_open_frames = 0
        self.left_modifier_last_fist_at = None
        self.left_modifier_release_candidate_since = None
        self.modifier_click_started_at = {side: None for side in sides}
        self.modifier_click_origin = {side: None for side in sides}
        self.modifier_click_kind = {side: None for side in sides}  # "pinch_right_click" | "grip_drag"
        self.modifier_click_release_since = {side: None for side in sides}
        self.modifier_click_last_pose_at = {side: None for side in sides}
        self.modifier_click_armed = {side: True for side in sides}
        # Background snapshots are blurred once when the glass menu is opened in OS mode;
        # virtual mode gets a blurred version of its own grid canvas.
        self._glass_backdrop_image = None
        self._glass_virtual_backdrop = None
        self._glass_virtual_backdrop_size = None
        self._capturing_glass_backdrop = False
        self.gesture_owner = None
        self.canvas_active_side = None
        self.active_slider = None
        self.active_slider_side = None
        self.last_active = time.monotonic()


        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.timeout.connect(self.persist_gesture_settings)
        self.tts_settings_save_timer = QTimer(self)
        self.tts_settings_save_timer.setSingleShot(True)
        self.tts_settings_save_timer.timeout.connect(self.tts.save_settings)
        # A mouse-transparent topmost visual layer keeps the skeleton, TLI and gesture
        # progress visible over the search launcher without blocking its controls.
        self.search_hand_overlay = SearchHandOverlay(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(50)
        QTimer.singleShot(1200, self._maybe_start_hand_onboarding)
        QTimer.singleShot(2400, self.speak_startup_greeting)

    def schedule_gesture_settings_save(self):
        self.settings_save_timer.start(450)

    def persist_gesture_settings(self):
        save_gesture_settings({
            "hand_visual_scale": self.hand_visual_scale,
            "hand_recognition_threshold": self.hand_recognition_threshold,
            "hand_contact_band_ratio": self.hand_contact_band_ratio,
            "palm_attract_acceleration": self.palm_attract_acceleration,
            "skin_tone_darkness": self.skin_tone_darkness,
            "finger_thickness": self.finger_thickness,
            "palm_width_scale": self.palm_width_scale,
            "hand_skin_effect": self.hand_render_style != "skeleton",
            "hand_render_style": self.hand_render_style,
            "skin_use_skeleton_color": self.skin_use_skeleton_color,
            "hand_transparency": self.hand_transparency,
            "tli_glow_enabled": self.tli_glow_enabled,
            "hand_tracking_sensitivity": self.hand_tracking_sensitivity,
            "scroll_acceleration": self.scroll_acceleration,
            "scroll_deadzone": self.scroll_deadzone,
            "scroll_backdrop_opacity": self.scroll_backdrop_opacity,
            "scroll_backdrop_hue": self.scroll_backdrop_hue,
            "scroll_notch_hue": self.scroll_notch_hue,
            "detached_notice_opacity": self.detached_notice_opacity,
            "custom_gesture_hold_seconds": self.custom_gesture_hold_seconds,
            "custom_gesture_tolerance": self.custom_gesture_tolerance,
            "cursor_smoothing": self.cursor_smoothing,
            "tli_glow_intensity": self.tli_glow_intensity,
            "tli_colour_shift_degrees": self.tli_colour_shift_degrees,
            "tli_border_thickness": self.tli_border_thickness,
            "tli_diameter": self.tli_diameter,
            "skeleton_line_thickness": self.skeleton_line_thickness,
            "click_debounce_seconds": self.click_debounce_seconds,
            "menu_pinch_hold_seconds": self.menu_pinch_hold_seconds,
            "default_gesture_hold_seconds": self.default_gesture_hold_seconds,
            "pinch_buffer_seconds": self.pinch_buffer_seconds,
            "invert_scroll": self.invert_scroll,
            "swap_drag_scroll_notches": self.swap_drag_scroll_notches,
            "cursor_lock_priority": self.cursor_lock_priority,
            "left_click_color": self.left_click_color.name(),
            "right_click_color": self.right_click_color.name(),
        })
        save_ui_preferences(self.ui_preferences)

    def _ui_font(self, size, weight=QFont.Normal):
        """Shared font sizing for glass menu, settings, status notices and modifier UI."""
        scale = min(1.4, max(0.7, float(getattr(self, "ui_font_scale", 1.0))))
        return QFont("Sans", max(6, int(round(float(size) * scale))), weight)

    def _log_d1_ui(self, message):
        message = str(message).strip()
        if not message:
            return
        if not hasattr(self, "d1_ui_log"):
            self.d1_ui_log = []
        stamp = time.strftime("%H:%M:%S")
        self.d1_ui_log.append(f"{stamp}  {message}")
        self.d1_ui_log = self.d1_ui_log[-80:]
        self.update()

    def _blur_glass_image(self, image, sigma=32.0):
        """Return a softly blurred, opaque RGB image for a localized frosted-glass panel."""
        if image is None or image.isNull():
            return None
        try:
            rgb = image.convertToFormat(QImage.Format.Format_RGB32)
            width, height = rgb.width(), rgb.height()
            byte_count = rgb.sizeInBytes()
            raw = np.frombuffer(rgb.bits(), dtype=np.uint8, count=byte_count)
            rows = raw.reshape((height, rgb.bytesPerLine()))
            pixels = rows[:, :width*4].reshape((height, width, 4))[:, :, :3].copy()
            # Keep the sigma expressed in full-resolution screen pixels, but work at a
            # smaller resolution to make stronger blur affordable without delaying gestures.
            scale = min(1.0, 1100.0/max(width, height))
            small_width = max(1, int(round(width*scale)))
            small_height = max(1, int(round(height*scale)))
            if scale < 0.999:
                small = cv2.resize(pixels, (small_width, small_height), interpolation=cv2.INTER_AREA)
            else:
                small = pixels
            blurred_small = cv2.GaussianBlur(
                small, (0, 0), sigmaX=max(0.5, sigma*scale), sigmaY=max(0.5, sigma*scale)
            )
            blurred = (cv2.resize(blurred_small, (width, height), interpolation=cv2.INTER_LINEAR)
                       if scale < 0.999 else blurred_small)
            result = np.empty((height, width, 4), dtype=np.uint8)
            result[:, :, :3] = blurred
            result[:, :, 3] = 255
            return QImage(result.data, width, height, width*4, QImage.Format.Format_RGB32).copy()
        except Exception as exc:
            print("Glass backdrop blur unavailable:", exc)
            return image

    def _virtual_glass_backdrop(self):
        """Make a blurred version of the virtual desktop's background grid."""
        size = (max(1, self.width()), max(1, self.height()))
        if self._glass_virtual_backdrop is not None and self._glass_virtual_backdrop_size == size:
            return self._glass_virtual_backdrop
        width, height = size
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor(8, 12, 22))
        grid = QPainter(image)
        grid.setRenderHint(QPainter.Antialiasing, True)
        for x in range(0, width, 64):
            grid.setPen(QPen(QColor(60, 105, 155, 28), 1))
            grid.drawLine(x, 0, x, height)
        for y in range(0, height, 64):
            grid.setPen(QPen(QColor(60, 105, 155, 28), 1))
            grid.drawLine(0, y, width, y)
        grid.end()
        self._glass_virtual_backdrop = self._blur_glass_image(image, sigma=36.0)
        self._glass_virtual_backdrop_size = size
        return self._glass_virtual_backdrop

    def _capture_glass_backdrop(self):
        """Capture the desktop without our overlay, then cache its blurred pixels for glass panels."""
        if self.desktop_mode != "os" or self._capturing_glass_backdrop:
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        was_visible = self.isVisible()
        was_fullscreen = self.isFullScreen()
        hand_overlay = getattr(self, "search_hand_overlay", None)
        hand_overlay_was_visible = bool(hand_overlay is not None and hand_overlay.isVisible())
        hand_overlay_timer = getattr(hand_overlay, "sync_timer", None) if hand_overlay is not None else None
        hand_overlay_timer_was_active = bool(hand_overlay_timer is not None and hand_overlay_timer.isActive())
        self._capturing_glass_backdrop = True
        try:
            if hand_overlay_timer_was_active:
                hand_overlay_timer.stop()
            if hand_overlay_was_visible:
                hand_overlay.hide()
            if was_visible:
                self.hide()
                QApplication.processEvents()
            captured = screen.grabWindow(0)
            image = captured.toImage()
            if image.isNull():
                return
            if image.size() != self.size():
                image = image.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            blurred = self._blur_glass_image(image, sigma=40.0)
            if blurred is not None and not blurred.isNull():
                self._glass_backdrop_image = blurred
        except Exception as exc:
            print("Could not capture glass backdrop:", exc)
        finally:
            if was_visible:
                if was_fullscreen:
                    self.showFullScreen()
                else:
                    self.show()
                self.raise_()
                self.apply_desktop_mode()
            if hand_overlay_was_visible and hand_overlay is not None:
                hand_overlay.show()
                hand_overlay.sync_to_launcher()
            if hand_overlay_timer_was_active and hand_overlay_timer is not None:
                hand_overlay_timer.start()
            self._capturing_glass_backdrop = False
            self.update()

    def _paint_frosted_backdrop(self, painter, rect, radius, opacity=0.15):
        """Draw only the blurred image inside a rounded glass panel's bounds."""
        image = self._glass_backdrop_image if self.desktop_mode == "os" else self._virtual_glass_backdrop()
        # Opening the floating Pinch Menu must be immediate. If the desktop snapshot has not
        # been captured yet, draw the pre-blurred virtual backdrop rather than blocking the
        # UI thread for a screen capture/blur at the end of the gesture timer.
        if image is None or image.isNull():
            image = self._virtual_glass_backdrop()
        if image is None or image.isNull():
            return
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.save()
        painter.setClipPath(path)
        painter.setOpacity(painter.opacity()*min(1.0, max(0.0, float(opacity))))
        # Only blit the part of the full-screen image behind the panel. Drawing the
        # whole screen-sized image every frame and relying on the clip made each
        # repaint far more expensive than the visible glass area requires.
        target = rect.intersected(QRectF(0, 0, self.width(), self.height()))
        if not target.isEmpty():
            sx = image.width()/max(1.0, float(self.width()))
            sy = image.height()/max(1.0, float(self.height()))
            source = QRectF(target.x()*sx, target.y()*sy, target.width()*sx, target.height()*sy)
            painter.drawImage(target, image, source)
        painter.restore()

    def _main_menu_close_rect(self, cx=None, cy=None, scale=None):
        """Wide Close button near the bottom edge of the main glass menu."""
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        _panel_width, panel_height = self._menu_panel_dimensions()
        return QRectF(cx-175.0*scale, cy+(panel_height*0.5-54.0)*scale,
                      350.0*scale, 36.0*scale)

    def _tli_calibration_button_rects(self, cx=None, cy=None, scale=None):
        """Left/right TLI calibration buttons, taken from the shared Hands-page layout."""
        rects = self._settings_control_rects(cx, cy, scale, page="hands")
        return {"left": rects["calibrate_left"], "right": rects["calibrate_right"]}

    # ---- Settings page layout -------------------------------------------------
    # One geometry source for every Settings page. Painting, click hit-testing,
    # hover glow and pressed glow all read these rectangles, so what is drawn is
    # exactly what can be pressed. Units are logical (before menu scaling) and
    # relative to the panel centre. Buttons sit in the left column, sliders in the
    # right column, and the panel height is derived from the taller column.
    SETTINGS_LEFT_X = -320.0
    SETTINGS_RIGHT_X = 20.0
    SETTINGS_COLUMN_W = 300.0
    SETTINGS_BUTTON_H = 38.0
    SETTINGS_BUTTON_GAP = 10.0
    SETTINGS_GROUP_GAP = 12.0
    SETTINGS_SLIDER_PITCH = 52.0
    SETTINGS_CONTENT_TOP = 85.0     # distance from panel top: clears the tab strip
    SETTINGS_CONTENT_BOTTOM = 62.0  # distance from panel bottom: clears Back
    SETTINGS_INFO_CONTENT_TOP = 78.0
    SETTINGS_MIN_HEIGHT = 400.0
    SETTINGS_INFO_PAGE_HEIGHT = 500.0

    def _settings_layout(self, page=None):
        """Logical layout of one Settings page (see the constants above)."""
        page = self.settings_page if page is None else page
        L, R, W = self.SETTINGS_LEFT_X, self.SETTINGS_RIGHT_X, self.SETTINGS_COLUMN_W
        bh, gap, group = self.SETTINGS_BUTTON_H, self.SETTINGS_BUTTON_GAP, self.SETTINGS_GROUP_GAP
        buttons, texts = {}, {}
        y = 0.0

        def stack(name, h=bh, after=gap):
            nonlocal y
            buttons[name] = (L, y, W, h)
            y += h + after

        right_h = 0.0
        if page == "interaction":
            stack("scroll_toggle")
            stack("swap_drag_scroll_toggle")
            stack("priority_toggle")
        elif page == "hands":
            stack("render_style")
            stack("skin_tint")
            y += group
            stack("calibrate_left")
            stack("calibrate_right")
            y += group
            stack("hand_setup")
        elif page == "tli":
            stack("glow_toggle")
            y += 6.0
            half = (W-gap)*0.5
            buttons["swatch_left"] = (L, y, half, 70.0)
            buttons["swatch_right"] = (L+half+gap, y, half, 70.0)
            y += 70.0 + gap
        elif page == "notifications":
            for key, _label, _default in NOTIFICATION_SETTING_ROWS:
                stack("notify:" + key)
            texts["help"] = (L, y, W, 40.0)
            y += 40.0 + gap
        elif page == "voice":
            stack("tts_toggle")
            stack("voice_cycle")
            y += group
            stack("test_voice")
            stack("face_welcome")
            y += group
            stack("streamtts_config", h=42.0)
            texts["status"] = (L, y, W, 48.0)
            y += 48.0 + gap
        elif page == "update":
            # The single UPDATE / INSTALL / RESTART button is centred in the panel.
            buttons["update_action"] = (-W*0.5, y, W, 46.0)
            y += 46.0 + 14.0
            # Status and description span both columns: this page has no sliders.
            texts["status"] = (L, y, 2*W+(R-(L+W)), 72.0)
            y += 80.0
            texts["about"] = (L, y, 2*W+(R-(L+W)), 44.0)
            y += 44.0 + gap
        elif page == "d1":
            stack("d1_model", h=44.0)
            half = (W-gap)*0.5
            buttons["d1_start"] = (L, y, half, 42.0)
            buttons["d1_stop"] = (L+half+gap, y, half, 42.0)
            y += 42.0 + gap
            stack("d1_retry", h=42.0)
            texts["state"] = (R, 0.0, W, 22.0)
            texts["about"] = (R, 26.0, W, 40.0)
            texts["log"] = (R, 72.0, W, 122.0)
            right_h = 72.0 + 122.0
        elif page == "gestures":
            y = 394.0  # see _custom_gesture_layout
        left_h = max(0.0, y - gap)

        sliders = []
        for index, row in enumerate(self._settings_slider_rows(page)):
            name = row[0]
            row_top = index*self.SETTINGS_SLIDER_PITCH
            track_left, track_right = R+12.0, R+W-12.0
            if name == "scroll_notch_hue":
                track_right -= 30.0  # room for the colour swatch at the track end
            sliders.append({"name": name, "y": row_top+12.0, "left": track_left, "right": track_right,
                            "label": (R, row_top+22.0, W, 18.0)})
        if sliders:
            right_h = max(right_h, len(sliders)*self.SETTINGS_SLIDER_PITCH - 12.0)

        if page in ("gesture_map", "speech_commands"):
            height = self.SETTINGS_INFO_PAGE_HEIGHT
            top = -height*0.5 + self.SETTINGS_INFO_CONTENT_TOP
        else:
            height = max(self.SETTINGS_MIN_HEIGHT,
                         self.SETTINGS_CONTENT_TOP + max(left_h, right_h) + self.SETTINGS_CONTENT_BOTTOM)
            top = -height*0.5 + self.SETTINGS_CONTENT_TOP
        return {"page": page, "height": height, "top": top,
                "buttons": buttons, "texts": texts, "sliders": sliders}

    def _settings_control_rects(self, cx=None, cy=None, scale=None, page=None, kind="buttons"):
        """Scaled overlay rectangles for a page's buttons (or text areas)."""
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        layout = self._settings_layout(page)
        top = layout["top"]
        return {name: QRectF(cx+x*scale, cy+(top+y)*scale, w*scale, h*scale)
                for name, (x, y, w, h) in layout[kind].items()}

    # Share of the Hands-page render switch taken by its "RENDER" caption.
    RENDER_SWITCH_CAPTION = 0.30

    def _render_style_segments(self, rect, scale):
        """(style, rect) for each segment of the SKELETON | FLAT | 3D switch (draw + click)."""
        area = rect.adjusted(rect.width()*self.RENDER_SWITCH_CAPTION, 3*scale, -3*scale, -3*scale)
        width = area.width()/len(HAND_RENDER_STYLES)
        return [(style, QRectF(area.left()+i*width, area.top(), width, area.height()))
                for i, style in enumerate(HAND_RENDER_STYLES)]

    def _settings_text_rects(self, cx=None, cy=None, scale=None, page=None):
        return self._settings_control_rects(cx, cy, scale, page=page, kind="texts")

    def _settings_back_rect(self, cx=None, cy=None, scale=None):
        """One shared bottom-centred Back button geometry for every settings page."""
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        _panel_width, panel_height = self._menu_panel_dimensions()
        return QRectF(cx-50.0*scale, cy+(panel_height*0.5-44.0)*scale,
                      100.0*scale, 38.0*scale)

    def _settings_tab_rects(self, cx=None, cy=None, scale=None):
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        # Give the two longest labels extra width while keeping the overall tab
        # strip the same width and centred in the panel. Hit-testing uses these
        # same rectangles, so the visible and clickable areas stay aligned.
        tab_gap = 4.5*scale
        tab_height = 40.0*scale
        widths = [94.0, 70.0, 70.0, 70.0, 94.0, 70.0, 70.0, 70.0]
        total = sum(widths)*scale + tab_gap*7.0
        left = cx-total*0.5
        _panel_width, panel_height = self._menu_panel_dimensions()
        y = cy-(panel_height*0.5-25.0)*scale
        names = ("interaction", "hands", "tli", "notifications", "gestures", "voice", "d1", "update")
        rects = {}
        x = left
        for name, width in zip(names, widths):
            rects[name] = QRectF(x, y, width*scale, tab_height)
            x += (width*scale) + tab_gap
        return rects

    def _notification_rect(self, position, width, height, margin=18):
        """Position a notification in one of six screen anchors."""
        width = min(max(1.0, float(width)), max(1.0, self.width()-2*margin))
        height = min(max(1.0, float(height)), max(1.0, self.height()-2*margin))
        position = position if position in NOTIFICATION_POSITIONS else "bottom-center"
        if position.endswith("left"):
            x = float(margin)
        elif position.endswith("right"):
            x = float(self.width()-width-margin)
        else:
            x = float((self.width()-width)*0.5)
        # Leave extra room above the Windows taskbar for bottom-anchored notices.
        bottom_margin = margin + 26.0
        y = float(margin if position.startswith("top") else self.height()-height-bottom_margin)
        return QRectF(x, y, width, height)

    def _stacked_notification_rect(self, position, width, height, stack_offsets, margin=18, gap=8):
        """Place same-anchor notifications in a vertical stack, never on top of each other."""
        rect = self._notification_rect(position, width, height, margin)
        offset = float(stack_offsets.get(position, 0.0))
        rect.translate(0.0, offset if position.startswith("top") else -offset)
        stack_offsets[position] = offset + float(height) + float(gap)
        # Keep unusually long stacks inside the screen even on compact displays.
        if rect.top() < margin:
            rect.moveTop(float(margin))
        bottom_margin = margin if position.startswith("top") else margin + 26.0
        if rect.bottom() > self.height()-bottom_margin:
            rect.moveBottom(float(self.height()-bottom_margin))
        return rect

    def _cycle_notification_position(self, key):
        current = self.ui_preferences.get(key, NOTIFICATION_POSITIONS[0])
        try:
            index = NOTIFICATION_POSITIONS.index(current)
        except ValueError:
            index = 0
        self.ui_preferences[key] = NOTIFICATION_POSITIONS[(index+1) % len(NOTIFICATION_POSITIONS)]
        save_ui_preferences(self.ui_preferences)
        self.status = f"{key.replace('_', ' ').title()}: {self.ui_preferences[key]}"
        self.update()

    def _toggle_context_menu_from_voice(self):
        """Voice-accessible toggle for the hand-anchored Pinch Menu."""
        opened = [side for side in ("right", "left") if self.fist_action_menu_open.get(side)]
        if opened:
            for side in opened:
                self._set_pinch_menu_open(side, False)
                self.pin_menu_ready_for_selection[side] = False
                self.fist_hovered_action[side] = None
            self.gesture_owner = None
            self.status = "Pinch Menu closed by voice command"
            self.update()
            return
        side = "right" if self.side_cursor_positions.get("right") is not None else ("left" if self.side_cursor_positions.get("left") is not None else "right")
        anchor = self.side_cursor_positions.get(side)
        if anchor is None:
            global_pos = QCursor.pos()
            mapped = self.mapFromGlobal(global_pos)
            anchor = QPointF(mapped.x(), mapped.y())
        anchor = QPointF(min(max(10, anchor.x()), self.width()-10), min(max(10, anchor.y()), self.height()-10))
        self._set_pinch_menu_open(side, True)
        # Voice-opened context menus also require a neutral/open hand before a new
        # grab or pinch may select anything, preventing the command's existing pose
        # from immediately activating the hovered item.
        self.action_stop_wait_open[side] = True
        self.action_stop_open_since[side] = None
        self.fist_origin_points[side] = QPointF(anchor)
        self.fist_arm_points[side] = QPointF(anchor)
        self.fist_hovered_action[side] = None
        self.pin_menu_ready_for_selection[side] = True
        self.pin_menu_selection_started_at[side] = None
        self.pin_menu_selection_kind[side] = None
        self.pin_menu_selection_option[side] = None
        self.canvas_active_side = side
        self.gesture_owner = side
        self.status = "Pinch Menu opened by voice — hover an option and grab or pinch to select"
        self.update()

    def _toggle_glass_menu_from_voice(self):
        visible = not (self.menu_open and self.hub_visible)
        self._set_menu_open(visible, reset_settings=not visible)
        self.status = "Glass menu " + ("shown" if visible else "hidden")
        self.update()

    def _begin_menu_transition(self, now=None):
        now = time.monotonic() if now is None else now
        self.menu_transition_started = now
        self.menu_input_block_until = max(self.menu_input_block_until,
                                          now + self.menu_transition_duration)
        self.update()

    def _set_menu_open(self, visible, now=None, reset_settings=False):
        visible = bool(visible)
        changed = (self.menu_open != visible or self.hub_visible != visible)
        self.menu_open = visible
        self.hub_visible = visible
        if not visible:
            self.custom_gesture_dropdown = None
            self.custom_gesture_key_capture = None
            # The search launcher is a companion to the glass menu. Never leave it
            # floating over the desktop after a gesture closes the glass menu.
            search_dialog = getattr(self, "search_dialog", None)
            if search_dialog is not None and search_dialog.isVisible():
                search_dialog.close()
        if reset_settings:
            self.settings_open = False
        if changed:
            if visible and self.desktop_mode == "os":
                self._capture_glass_backdrop()
            self._begin_menu_transition(now)
        return changed

    def _restart_spatial_desktop(self):
        """Launch a fresh instance, then close this instance."""
        try:
            command = [sys.executable, *sys.argv]
            options = {"cwd": str(ROOT), "close_fds": True}
            if os.name == "nt":
                options["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                                             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
            else:
                options["start_new_session"] = True
            subprocess.Popen(command, **options)
            self.status = "Restarting Spatial Desktop…"
            self.update_status = self.status
            self.update()
            QTimer.singleShot(250, QApplication.instance().quit)
        except Exception as exc:
            self.status = f"Could not restart automatically: {str(exc)[:120]}"
            self.update_status = self.status
            self.update()

    def _start_update_check(self):
        """One-button update flow: check version, then automatically download if newer."""
        if self.update_check_running or self.update_action_running:
            return
        self.update_check_running = True
        self.update_available = False
        self.update_download_path = None
        self.update_progress_current = 0
        self.update_progress_total = 0
        self.update_status = "Checking GitHub for a newer version…"
        self.status = self.update_status
        self.update()

        def worker():
            try:
                local_version = read_version(VERSION_FILE)
                remote_version = fetch_remote_version()
                comparison = compare_versions(remote_version, local_version)
                if comparison > 0:
                    message = f"Update v{remote_version} found. Downloading…"
                    self.update_check_signals.finished.emit(message, remote_version)
                elif comparison == 0:
                    self.update_check_signals.finished.emit(
                        f"You're up to date: v{local_version}.", "")
                else:
                    self.update_check_signals.finished.emit(
                        f"Installed v{local_version} is newer than GitHub v{remote_version}.", "")
            except Exception as exc:
                self.update_check_signals.finished.emit(
                    f"Could not check GitHub. Check your connection. {str(exc)[:100]}", "")

        self._update_check_thread = threading.Thread(target=worker, name="SpatialDesktopUpdateCheck", daemon=True)
        self._update_check_thread.start()

    def _on_update_check_finished(self, message, remote_version):
        self.update_check_running = False
        self.update_status = message
        self.update_available = False
        if remote_version:
            try:
                self.update_available = compare_versions(remote_version, read_version(VERSION_FILE)) > 0
            except Exception:
                self.update_available = False
        self.status = message.replace("\n", " · ")
        self.update()
        if self.update_available:
            self._start_update_download()

    def _on_update_progress(self, current, total):
        self.update_progress_current = max(0, int(current))
        self.update_progress_total = max(0, int(total))
        if total > 0:
            self.update_status = f"Downloading update… {min(100, int(current * 100 / total))}%"
        else:
            self.update_status = f"Downloading update… {current / (1024 * 1024):.1f} MB"
        self.update()

    def _start_update_download(self):
        if self.update_action_running or not self.update_available:
            return
        self.update_action_running = True
        self.update_action_mode = "download"
        self.update_progress_current = 0
        self.update_progress_total = 0
        self.update_status = "Downloading update from GitHub…"
        self.status = self.update_status
        self.update()

        def worker():
            try:
                source_root = download_update(progress_callback=lambda current, total:
                    self.update_check_signals.progress.emit(int(current), int(total)))
                staged_version = read_version(Path(source_root) / "version.txt")
                current_version = read_version(VERSION_FILE)
                if compare_versions(staged_version, current_version) <= 0:
                    raise ValueError(f"Downloaded version v{staged_version} is not newer than installed v{current_version}")
                self.update_check_signals.action_finished.emit(
                    f"Update v{staged_version} downloaded and verified.", str(source_root))
            except Exception as exc:
                self.update_check_signals.action_finished.emit(
                    f"Update download failed. Check your connection and try again. {str(exc)[:120]}", "")

        threading.Thread(target=worker, name="SpatialDesktopUpdateDownload", daemon=True).start()

    def _start_update_install(self):
        if self.update_action_running or not self.update_download_path:
            return
        self.update_action_running = True
        self.update_action_mode = "install"
        self.update_status = "Installing update and backing up replaced files…"
        self.status = self.update_status
        self.update()
        source_root = self.update_download_path

        def worker():
            try:
                backup_root = install_update(source_root, ROOT)
                self.update_check_signals.action_finished.emit(
                    f"Update installed successfully.\nRestart Spatial Desktop to run the new version.\nBackup: {backup_root}", "installed")
            except Exception as exc:
                self.update_check_signals.action_finished.emit(
                    f"Update installation failed.\nYour settings and calibration were preserved.\n{str(exc)[:140]}", "")

        threading.Thread(target=worker, name="SpatialDesktopUpdateInstall", daemon=True).start()

    def _on_update_action_finished(self, message, result):
        finished_mode = self.update_action_mode
        self.update_action_running = False
        self.update_action_mode = ""
        self.update_status = message
        if result == "installed":
            self.update_download_path = None
            self.update_available = False
            self.update_installed = True
        elif result:
            self.update_download_path = result
        elif finished_mode == "download":
            self.update_download_path = None
        self.status = message.replace("\n", " · ")
        self.update()

    def _begin_settings_page_transition(self, previous_size, now=None):
        """Animate only the page contents and panel shape; never hide the panel."""
        self.settings_page_transition_from_size = tuple(previous_size)
        self.settings_page_transition_started = time.monotonic() if now is None else now
        self.update()

    def _settings_page_transition_progress(self, now=None):
        if not self.settings_page_transition_started:
            return 1.0
        now = time.monotonic() if now is None else now
        return max(0.0, min(1.0, (now-self.settings_page_transition_started) /
                            max(0.01, self.settings_page_transition_duration)))

    def _menu_panel_dimensions(self, page=None):
        """Return logical menu size for a settings page, before display scaling."""
        if not self.settings_open:
            return 490.0, 430.0
        # Heights follow each page's laid-out content, so tabs and Back remain
        # inside the glass while avoiding large blank bands.
        return 720.0, self._settings_layout(page)["height"]

    def menu_ui_scale(self):
        base = max(0.4, self.hub_scale*self.object_depth_scale)
        if self.settings_open:
            panel_width, panel_height = self._menu_panel_dimensions()
            fit_h = max(0.4, (self.height()-36.0)/panel_height)
            fit_w = max(0.4, (self.width()-36.0)/panel_width)
            return min(base, fit_h, fit_w)
        return base

    def _canvas_menu_panel_contains(self, point):
        if point is None or not (self.menu_open and self.hub_visible):
            return False
        cx, cy = self.hub_pos.x()*self.width(), self.hub_pos.y()*self.height()
        scale = self.menu_ui_scale()
        panel_width, panel_height = self._menu_panel_dimensions()
        half_width = panel_width*0.5*scale
        half_height = panel_height*0.5*scale
        return (cx-half_width <= point.x() <= cx+half_width and
                cy-half_height <= point.y() <= cy+half_height)

    def _settings_slider_rows(self, page=None):
        """Slider values, ranges and labels per page, in top-to-bottom order."""
        page = self.settings_page if page is None else page
        if page == "voice":
            return [("tts_speed", float(self.tts.settings["speed"]), 0.7, 1.5, "SPEECH SPEED")]
        if page == "interaction":
            return [
                ("sensitivity", self.hand_tracking_sensitivity, 0.5, 1.5, "DRAG/SCROLL SENSITIVITY"),
                ("smoothing", self.cursor_smoothing, 0.0, 0.95, "CURSOR SMOOTHING"),
                ("default_gesture_hold_seconds", self.default_gesture_hold_seconds, 0.5, 5.0, "DEFAULT GESTURE HOLD TIMER"),
                ("pinch_buffer_seconds", self.pinch_buffer_seconds, 0.3, 2.0, "PINCH BUFFER (BEFORE HOLD TIMER)"),
                ("scroll_acceleration", self.scroll_acceleration, 0.5, 10.0, "SCROLL ACCELERATION"),
                ("scroll_deadzone", self.scroll_deadzone, 4.0, 120.0, "SCROLL DEAD ZONE"),
                ("scroll_backdrop_opacity", self.scroll_backdrop_opacity, 0.0, 100.0, "SCROLL CIRCLE TRANSPARENCY"),
                ("scroll_backdrop_hue", self.scroll_backdrop_hue, 0.0, 255.0, "SCROLL CIRCLE GREY SHADE"),
                ("scroll_notch_hue", self.scroll_notch_hue, 0.0, 360.0, "SCROLL NOTCH COLOUR"),
            ]
        if page == "tli":
            return [
                ("tli_glow_intensity", self.tli_glow_intensity, 0.0, 2.5, "TLI GLOW INTENSITY"),
                ("tli_colour_shift_degrees", self.tli_colour_shift_degrees, 0.0, 360.0, "TLI COLOUR / HUE"),
                ("tli_border_thickness", self.tli_border_thickness, 1.5, 8.0, "TLI BORDER THICKNESS"),
                ("tli_diameter", self.tli_diameter, 12.0, 96.0, "TLI DIAMETER"),
            ]
        if page == "notifications":
            return [("detached_notice_opacity", self.detached_notice_opacity, 0.0, 100.0,
                     "DETACHED NOTICE FADES TO"),
                    ("ui_font_scale", self.ui_font_scale * 100.0, 70.0, 140.0,
                     "UI FONT SIZE")]
        if page == "gestures":
            return [
                ("custom_gesture_hold_seconds", self.custom_gesture_hold_seconds, 0.2, 1.5, "HOLD TIME"),
                ("custom_gesture_tolerance", self.custom_gesture_tolerance, 0.08, 0.40, "MATCH TOLERANCE"),
            ]
        if page == "hands":
            return [
                ("hand_scale", self.hand_visual_scale, 0.1, 1.0, "HAND SCALE"),
                ("hand_recognition_threshold", self.hand_recognition_threshold, 0.30, 0.90, "HAND RECOGNITION THRESHOLD"),
                ("finger_thickness", self.finger_thickness, 0.8, 10.0, "FINGER THICKNESS"),
                ("palm_width_scale", self.palm_width_scale, 0.65, 1.8, "PALM WIDTH"),
                ("skin_tone_darkness", self.skin_tone_darkness, 0.0, 100.0, "SKIN TONE DARKNESS"),
                ("skeleton_line_thickness", self.skeleton_line_thickness, 0.5, 4.0, "SKELETON LINE THICKNESS"),
                ("hand_transparency", self.hand_transparency, 10.0, 100.0, "HAND OPACITY"),
                ("hand_contact_band_ratio", self.hand_contact_band_ratio, 0.0, 0.9, "PALM ATTRACT DISTANCE"),
                ("palm_attract_acceleration", self.palm_attract_acceleration, 0.2, 5.0, "PALM ATTRACT ACCELERATION"),
            ]
        return []

    def _settings_slider_specs(self):
        """Single source of truth for slider values, ranges, labels and hit-test rows.

        Each entry is (name, y_offset, value, minimum, maximum, label); y_offset is the
        track centre relative to the panel centre, taken from the shared page layout.
        """
        layout = self._settings_layout()
        top = layout["top"]
        offsets = {slider["name"]: top+slider["y"] for slider in layout["sliders"]}
        return [(name, offsets.get(name, 0.0), value, minimum, maximum, label)
                for name, value, minimum, maximum, label in self._settings_slider_rows()]

    def _settings_slider_layouts(self, cx=None, cy=None, scale=None):
        """Scaled slider geometry for the current page: name -> (left, right, y, label_rect)."""
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        layout = self._settings_layout()
        top = layout["top"]
        result = {}
        for slider in layout["sliders"]:
            lx, ly, lw, lh = slider["label"]
            result[slider["name"]] = (cx+slider["left"]*scale, cx+slider["right"]*scale,
                                      cy+(top+slider["y"])*scale,
                                      QRectF(cx+lx*scale, cy+(top+ly)*scale, lw*scale, lh*scale))
        return result

    def _canvas_slider_geometry(self, name):
        """Return slider track geometry and the current knob point in overlay coordinates."""
        if not self.settings_open or not self.menu_open or not self.hub_visible:
            return None
        definition = next((item for item in self._settings_slider_specs() if item[0] == name), None)
        geometry = self._settings_slider_layouts().get(name)
        if definition is None or geometry is None:
            return None
        _, _offset, value, minimum, maximum, _label = definition
        left, right, y, _label_rect = geometry
        ratio = min(1.0, max(0.0, (value-minimum)/max(1e-6, maximum-minimum)))
        knob = QPointF(left+ratio*(right-left), y)
        return left, right, y, knob

    def _canvas_slider_name_at(self, point, for_grab=False):
        """Hit-test slider tracks; grab hits have a wider snap-to-slider capture band."""
        if point is None or not self.settings_open or not self.menu_open or not self.hub_visible:
            return None
        if self.settings_page == "gestures" and self.custom_gesture_dropdown is not None:
            return None  # the open action list covers the sliders
        scale = self.menu_ui_scale()
        x_margin = (28 if for_grab else 14)*scale
        y_limit = (30 if for_grab else 26)*scale
        candidates = [(abs(point.y()-y), name)
                      for name, (left, right, y, _label_rect) in self._settings_slider_layouts().items()
                      if left-x_margin <= point.x() <= right+x_margin]
        if not candidates:
            return None
        distance, name = min(candidates)
        return name if distance <= y_limit else None

    def _action_menu_item_names(self):
        return ("right_click", "glass_menu", "dictation_toggle", "stt_toggle")

    def _action_menu_labels(self):
        search_dictation = bool(self.search_dialog is not None and self.search_dialog.isVisible()
                                and self.search_dialog.dictation_enabled)
        dictation_on = self.focus_dictation_enabled or search_dictation
        return {
            "right_click": "RIGHT-CLICK MENU" if not self.right_click_mode_enabled else "RIGHT-CLICK MENU  ·  ON",
            "glass_menu": ("GLASS MENU  ·  HIDE" if self.menu_open and self.hub_visible else "GLASS MENU  ·  SHOW"),
            "dictation_toggle": "DICTATE  ·  " + ("ON" if dictation_on else "OFF"),
            "stt_toggle": "STT  ·  " + ("ON" if self.speech.running else "OFF"),
        }

    def _announce_voice_input_state(self, enabled):
        """Use the same spoken state confirmation for all voice-input toggles."""
        self.tts.speak("Voice input is " + ("on." if enabled else "off."))

    def _toggle_focused_dictation_from_menu(self):
        dialog = self.search_dialog
        if dialog is not None and dialog.isVisible():
            enabled = not dialog.dictation_enabled
            dialog.set_dictation_enabled(enabled, notify=True)
            self.status = "Search dictation " + ("enabled" if dialog.dictation_enabled else "disabled")
        elif self.focus_dictation_enabled:
            self.end_focused_dictation("")
        else:
            self.begin_focused_dictation()
        dialog = self.search_dialog
        dictation_active = bool(self.speech.running and (
            self.focus_dictation_enabled or
            (dialog is not None and dialog.isVisible() and dialog.dictation_enabled)))
        self._announce_voice_input_state(dictation_active)

    def _toggle_stt_from_menu(self):
        dialog = self.search_dialog
        if self.speech.running:
            if dialog is not None and dialog.isVisible() and dialog.dictation_enabled:
                dialog.set_dictation_enabled(False, notify=True)
            self.focus_dictation_enabled = False
            self.focus_dictation_paused = False
            self.focus_dictation_committed_words = []
            self.focus_dictation_last_partial = ""
            self.speech.stop()
            self.stt_enabled = False
            self.status = "Vosk speech recognition stopped"
        else:
            self.stt_enabled = bool(self.speech.start())
            self.status = "Vosk speech recognition " + ("enabled" if self.stt_enabled else "could not start")
        self._announce_voice_input_state(self.stt_enabled)

    def _action_menu_geometry(self, side):
        """Return panel and option rectangles for the thumb/index Pinch Menu."""
        anchor = self.fist_arm_points.get(side)
        if anchor is None:
            anchor = self.fist_origin_points.get(side)
        if anchor is None:
            return None, {}
        scale = min(1.0, max(0.72, min(self.width()/1240.0, self.height()/850.0)))
        panel_w = 308.0*scale
        header_h = 34.0*scale
        item_h = 54.0*scale
        gap = 14.0*scale
        pad = 20.0*scale
        menu_items = self._action_menu_item_names()
        # Compact, titleless Pinch Menu: primary actions full width, voice controls half-width.
        header_h = 0.0
        panel_h = pad + 9.0*scale + item_h*3 + 2*gap + pad
        if anchor.x() + panel_w + 34*scale <= self.width()-8:
            x = anchor.x() + 34*scale
        else:
            x = anchor.x() - panel_w - 34*scale
        x = min(max(8.0, x), max(8.0, self.width()-panel_w-8.0))
        y = min(max(8.0, anchor.y()-panel_h*0.5), max(8.0, self.height()-panel_h-8.0))
        panel = QRectF(x, y, panel_w, panel_h)
        item_x = x+pad
        item_w = panel_w-2*pad
        top = y+pad+9.0*scale
        items = {
            "right_click": QRectF(item_x, top, item_w, item_h),
            "glass_menu": QRectF(item_x, top+item_h+gap, item_w, item_h),
            "dictation_toggle": QRectF(item_x, top+2*(item_h+gap), (item_w-gap)/2, item_h),
            "stt_toggle": QRectF(item_x+(item_w+gap)/2, top+2*(item_h+gap), (item_w-gap)/2, item_h),
        }
        return panel, items

    def _action_menu_option_at(self, side, point):
        _panel, items = self._action_menu_geometry(side)
        for name, rect in items.items():
            if rect.contains(point):
                return name
        return None

    def _find_native_window_at(self, point, prefer_smallest=False):
        """Find a native window under a point, ignoring the Spatial Desktop overlay.

        Windows uses Win32 enumeration. Linux/X11 uses wmctrl geometry and excludes
        windows owned by this process so the transparent tracking overlay is not chosen.
        """
        if point is None:
            return None
        if sys.platform.startswith("linux"):
            try:
                global_point = self._virtual_point_to_global(point)
                px, py = int(global_point.x()), int(global_point.y())
                own_pid = os.getpid()
                try:
                    listing = subprocess.check_output(
                        ["wmctrl", "-lG"], text=True, stderr=subprocess.DEVNULL, timeout=0.4)
                except Exception:
                    return None
                matches = []
                for line in listing.splitlines():
                    fields = line.split(None, 7)
                    if len(fields) < 6:
                        continue
                    try:
                        wid = int(fields[0], 16)
                        x, y, width, height = map(int, fields[2:6])
                    except (ValueError, IndexError):
                        continue
                    try:
                        if wid == int(self.winId()):
                            continue
                    except Exception:
                        pass
                    if width <= 1 or height <= 1 or not (x <= px < x + width and y <= py < y + height):
                        continue
                    try:
                        pid_text = subprocess.check_output(
                            ["xdotool", "getwindowpid", hex(wid)], text=True,
                            stderr=subprocess.DEVNULL, timeout=0.25).strip()
                        if pid_text and int(pid_text) == own_pid:
                            continue
                    except Exception:
                        pass
                    matches.append((wid, width * height))
                if not matches:
                    return None
                # wmctrl does not consistently expose stacking order across WMs;
                # choosing the smallest containing client avoids desktop-sized surfaces.
                return min(matches, key=lambda item: item[1])[0]
            except Exception:
                return None
        if not sys.platform.startswith("win"):
            return None
        try:
            user32 = ctypes.windll.user32
            global_point = self._virtual_point_to_global(point)
            px, py = int(global_point.x()), int(global_point.y())
            overlay_hwnd = int(self.winId())
            rect = wintypes.RECT()
            class_buffer = ctypes.create_unicode_buffer(256)
            found = {"hwnd": None, "matches": []}
            enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            @enum_type
            def enum_window(hwnd, _lparam):
                try:
                    handle = int(hwnd)
                except (TypeError, ValueError):
                    handle = int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0)
                if handle == overlay_hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                    return True
                if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return True
                if not (rect.left <= px < rect.right and rect.top <= py < rect.bottom):
                    return True
                class_buffer.value = ""
                user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))
                if class_buffer.value in ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"):
                    return True
                # Ignore cloaked UWP windows where DWM reports that they are not visible.
                try:
                    cloaked = wintypes.DWORD(0)
                    dwm = ctypes.windll.dwmapi
                    if dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0 and cloaked.value:
                        return True
                except Exception:
                    pass
                found["matches"].append((handle, int(rect.right-rect.left) * int(rect.bottom-rect.top)))
                if not prefer_smallest:
                    found["hwnd"] = handle
                    return False
                return True

            user32.EnumWindows(enum_window, 0)
            if prefer_smallest and found["matches"]:
                # Prefer a real app-sized window over a desktop-sized surface.
                return min(found["matches"], key=lambda item: item[1])[0]
            return found["hwnd"]
        except Exception as exc:
            self.status = f"Window detection unavailable: {exc}"
            return None

    def _native_window_rect(self, hwnd):
        if not hwnd:
            return None
        if sys.platform.startswith("linux"):
            try:
                wid = hex(int(hwnd))
                output = subprocess.check_output(
                    ["wmctrl", "-lG"], text=True, stderr=subprocess.DEVNULL, timeout=0.4)
                for line in output.splitlines():
                    fields = line.split(None, 7)
                    if len(fields) >= 6 and int(fields[0], 16) == int(hwnd):
                        x, y, width, height = map(int, fields[2:6])
                        return (x, y, x + width, y + height)
            except Exception:
                return None
            return None
        if not sys.platform.startswith("win"):
            return None
        try:
            rect = wintypes.RECT()
            if ctypes.windll.user32.GetWindowRect(wintypes.HWND(int(hwnd)), ctypes.byref(rect)):
                return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception:
            pass
        return None

    def _prepare_native_window_drag(self, hwnd):
        """Prepare Linux/X11 clients for direct manipulation in tiling WMs such as Qtile."""
        if not hwnd or not sys.platform.startswith("linux"):
            return
        try:
            wid = hex(int(hwnd))
            subprocess.run(["wmctrl", "-ia", wid], check=False, timeout=0.3,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        # A tiled client cannot be freely positioned with EWMH moveresize requests.
        # Qtile's enable_floating is idempotent, unlike toggle_floating, so existing
        # floating windows remain floating. Other window managers simply skip this.
        try:
            subprocess.run(["qtile", "cmd-obj", "-o", "window", "-f", "enable_floating"],
                           check=False, timeout=0.45, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _move_native_window(self, hwnd, left, top):
        """Move a native window by its outer rect; no title-bar mouse-down is needed."""
        if not hwnd:
            return False
        if sys.platform.startswith("linux"):
            # wmctrl uses the EWMH moveresize request, which works with X11 WMs such
            # as Qtile. -1 keeps the current size; only the top-left position changes.
            try:
                subprocess.run(["wmctrl", "-ir", hex(int(hwnd)), "-e",
                                f"0,{int(round(left))},{int(round(top))},-1,-1"],
                               check=True, timeout=0.35, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except Exception as exc:
                self.status = f"Linux window move failed (install wmctrl / use X11): {exc}"
                return False
        if not sys.platform.startswith("win"):
            return False
        try:
            user32 = ctypes.windll.user32
            flags = 0x0001 | 0x0004 | 0x0010  # NOSIZE | NOZORDER | NOACTIVATE
            return bool(user32.SetWindowPos(wintypes.HWND(int(hwnd)), None, int(round(left)),
                                            int(round(top)), 0, 0, flags))
        except Exception as exc:
            self.status = f"Window move failed: {exc}"
            return False

    def _is_native_taskbar_point(self, point):
        """Avoid treating the Windows taskbar as a draggable application window."""
        if not sys.platform.startswith("win") or point is None:
            return False
        try:
            user32 = ctypes.windll.user32
            global_point = self._virtual_point_to_global(point)
            px, py = int(global_point.x()), int(global_point.y())
            found = {"inside": False}
            enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            @enum_type
            def enum_window(hwnd, _lparam):
                class_name = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(hwnd, class_name, len(class_name))
                if class_name.value not in ("Shell_TrayWnd", "Shell_SecondaryTrayWnd"):
                    return True
                rect = wintypes.RECT()
                if (user32.GetWindowRect(hwnd, ctypes.byref(rect)) and
                        rect.left <= px < rect.right and rect.top <= py < rect.bottom):
                    found["inside"] = True
                    return False
                return True

            user32.EnumWindows(enum_type(enum_window), 0)
            return bool(found["inside"])
        except Exception:
            return False

    def _native_nonclient_hit_test(self, hwnd, point):
        """Return Win32 WM_NCHITTEST at a screen point, or None if unavailable."""
        if not sys.platform.startswith("win") or not hwnd or point is None:
            return None
        try:
            user32 = ctypes.windll.user32
            global_point = self._virtual_point_to_global(point)
            sx, sy = int(global_point.x()), int(global_point.y())
            lparam = ((sy & 0xFFFF) << 16) | (sx & 0xFFFF)
            # A hung target application must not stall the camera/gesture loop.
            result = ctypes.c_size_t(0)
            send_timeout = user32.SendMessageTimeoutW
            send_timeout.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                     ctypes.c_ssize_t, wintypes.UINT, wintypes.UINT,
                                     ctypes.POINTER(ctypes.c_size_t)]
            send_timeout.restype = ctypes.c_size_t
            ok = send_timeout(wintypes.HWND(int(hwnd)), 0x0084, 0, lparam,
                              0x0002, 60, ctypes.byref(result))  # SMTO_ABORTIFHUNG, 60ms
            return int(result.value) if ok else None
        except Exception:
            return None

    def _arm_fist_hold_action(self, side, cursor, now, automatic_motion=False, force_mode=None):
        """Arm scrolling or a deliberate shake-grab; never move a window during the shake."""
        if self.fist_hold_armed.get(side):
            return
        origin = self.fist_origin_points.get(side) or QPointF(cursor)
        origin = QPointF(origin)
        self.fist_hold_armed[side] = True
        self.fist_arm_points[side] = QPointF(cursor)
        self.fist_last_points[side] = QPointF(cursor)

        if self.fist_start_controls.get(side):
            self.fist_action[side] = "slider"
            self.status = "Dragging settings slider"
            return
        virtual_modifier_drag = force_mode == "drag_window" and self.desktop_mode != "os"
        if self._canvas_menu_panel_contains(origin):
            if force_mode == "drag_window":
                # The existing right-fist modifier drag moves the glass panel itself
                # when the gesture starts on the panel, rather than the desktop below it.
                self.fist_action[side] = "menu_drag"
                self.fist_last_points[side] = QPointF(cursor)
                self.status = "Glass menu drag active — move the right hand; open to release"
                return
            if not virtual_modifier_drag:
                # Quick grasp/pinch remains reserved for menu controls.
                self.fist_action[side] = None
                self.status = "Use a quick grasp to activate glass menu controls"
                return
        if self.desktop_mode == "os" and (
                self.fist_started_on_taskbar.get(side, False) or self._is_native_taskbar_point(origin)):
            # Taskbar icons are click targets, not application windows to move.
            self.fist_action[side] = "taskbar_hold_ignored"
            self.status = "Taskbar: use a quick grasp or pinch to activate icons"
            return
        if (force_mode == "drag_window" and self.desktop_mode == "os" and
                not should_arm_shake_drag(
                    hit_test=self.fist_start_hit_test.get(side),
                    has_target_window=bool(self.fist_target_window.get(side)))):
            # Minimise, maximise and close buttons should remain precise quick-click targets;
            # never turn a fist shake over those controls into a whole-window drag.
            self.fist_action[side] = "window_control_hold_ignored"
            self.status = "Use a quick grasp or pinch on minimise, maximise, or close controls"
            return
        if self.desktop_mode != "os" and force_mode != "scroll_window":
            self.fist_action[side] = "canvas_drag"
            self.status = "Moving virtual desktop canvas"
            return

        hwnd = self.fist_target_window.get(side)
        if hwnd is None:
            hwnd = self._find_native_window_at(origin)
        hit_test = self.fist_start_hit_test.get(side)
        # The mode must be explicit: vertical movement scrolls, while the two-fist
        # clutch starts a window drag. Never infer drag from an arbitrary threshold.
        mode = force_mode or ("scroll_window" if self.scroll_mode_enabled else "drag_window")
        if mode == "scroll_window" and self.mouse_detached_by_side.get(side, False):
            # Drag scrolling is disabled for a detached hand: it would scroll a window
            # the user isn't pointing the OS cursor at, and at a different speed.
            self.fist_action[side] = "detached_scroll_ignored"
            self.status = f"{side.title()} hand is detached: drag scrolling is off until you reattach it"
            return

        active = {
            "side": side,
            "origin": QPointF(origin),
            # For drags, movement accumulated during confirmation can catch up naturally.
            # Scroll starts at the current cursor so the confirmation-phase travel cannot
            # become one large, jerky first scroll event.
            "last_cursor": QPointF(cursor if (mode == "scroll_window" or not automatic_motion) else origin),
            "hwnd": hwnd,
            "open_since": None,
            "ambiguous_since": None,
            "scroll_remainder": 0.0,
            "scroll_total_travel": 0.0,
            "scroll_smoothed_dy": 0.0,
            "scroll_smoothed_dx": 0.0,
            "scroll_axis": "vertical",
            "horizontal_scroll_supported": None,
            "scroll_nav_fired": False,
            "scroll_stage": 0,
        }
        if mode == "scroll_window":
            active["mode"] = "scroll_window"
            self.fist_scroll_markers[side] = {"point": QPointF(origin), "until": 0.0}
            self.active_action[side] = active
            self.fist_action[side] = "native_action"
            self.status = "Scroll action active — keep fist closed; open hand to stop"
            return

        if mode == "drag_window" and sys.platform.startswith("linux"):
            self._prepare_native_window_drag(hwnd)
        rect = self._native_window_rect(hwnd) if hwnd else None
        if rect is None:
            self.fist_action[side] = None
            self.status = "No movable window under the starting TLI"
            return
        if sys.platform.startswith("win"):
            try:
                user32 = ctypes.windll.user32
                if user32.IsZoomed(wintypes.HWND(int(hwnd))):
                    user32.ShowWindow(wintypes.HWND(int(hwnd)), 9)  # SW_RESTORE
                    rect = self._native_window_rect(hwnd) or rect
            except Exception:
                pass
        active["mode"] = "drag_window"
        active["rect"] = rect
        self.active_action[side] = active
        self.fist_action[side] = "native_action"
        self.status = "Window drag active — keep fist closed; open hand to release"

    def _dismiss_pinch_menu(self, side, now):
        """Dismiss the Pinch Menu and consume the outside gesture."""
        self._set_pinch_menu_open(side, False, now)
        self._reset_fist_gesture(side)
        self.menu_pinch_started[side] = None
        self.menu_pinch_accumulated_seconds[side] = 0.0
        self.menu_pinch_pulse_started_at[side] = None
        self.menu_pinch_origin_points[side] = None
        self.menu_pinch_release_candidate_since[side] = None
        self.menu_pinch_click_fired[side] = False
        self.menu_pinch_latched.discard(side)
        self.pin_menu_ready_for_selection[side] = False
        self.pin_menu_selection_started_at[side] = None
        self.pin_menu_selection_kind[side] = None
        self.pin_menu_selection_option[side] = None
        self.action_stop_wait_open[side] = True
        self.action_stop_open_since[side] = None
        if self.gesture_owner == side:
            self.gesture_owner = None
        self.status = "Pinch Menu closed — outside grab/pinch consumed"
        self.update()

    def _window_has_horizontal_scrollbar(self, active, cursor=None):
        """Best-effort check for a native horizontal scroll range under the TLI."""
        if not sys.platform.startswith("win"):
            return False
        point = cursor or active.get("last_cursor") or active.get("origin")
        if point is None:
            return False
        try:
            screen_point = self._virtual_point_to_global(point)
            user32 = ctypes.windll.user32
            native_point = wintypes.POINT(int(screen_point.x()), int(screen_point.y()))
            window_from_point = user32.WindowFromPoint
            window_from_point.argtypes = [wintypes.POINT]
            window_from_point.restype = wintypes.HWND
            get_parent = user32.GetParent
            get_parent.argtypes = [wintypes.HWND]
            get_parent.restype = wintypes.HWND
            hwnd = window_from_point(native_point)
            if not hwnd and active.get("hwnd"):
                hwnd = wintypes.HWND(int(active["hwnd"]))
            class SCROLLINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("fMask", wintypes.UINT),
                            ("nMin", ctypes.c_int), ("nMax", ctypes.c_int),
                            ("nPage", wintypes.UINT), ("nPos", ctypes.c_int),
                            ("nTrackPos", ctypes.c_int)]
            get_scroll_info = user32.GetScrollInfo
            get_scroll_info.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.POINTER(SCROLLINFO)]
            get_scroll_info.restype = wintypes.BOOL
            # Check the control under the point and a few parents (many apps put the
            # scroll range on a parent/client window rather than the leaf control).
            for _ in range(5):
                if not hwnd:
                    break
                info = SCROLLINFO(ctypes.sizeof(SCROLLINFO), 0x17, 0, 0, 0, 0, 0)
                if get_scroll_info(hwnd, 0, ctypes.byref(info)):
                    effective_max = int(info.nMax) - max(0, int(info.nPage) - 1)
                    if effective_max > int(info.nMin):
                        return True
                hwnd = get_parent(hwnd)
            # Also inspect the original top-level target captured when the fist began.
            target_hwnd = active.get("hwnd")
            if target_hwnd:
                info = SCROLLINFO(ctypes.sizeof(SCROLLINFO), 0x17, 0, 0, 0, 0, 0)
                if get_scroll_info(wintypes.HWND(int(target_hwnd)), 0, ctypes.byref(info)):
                    effective_max = int(info.nMax) - max(0, int(info.nPage) - 1)
                    if effective_max > int(info.nMin):
                        return True
        except Exception:
            pass
        return False

    def _send_browser_navigation(self, direction):
        """Send the mouse side-button Back (-1) or Forward (+1) action."""
        try:
            if pyautogui is not None:
                pyautogui.click(button="x1" if int(direction) < 0 else "x2", _pause=False)
                return True
        except Exception:
            pass
        if sys.platform.startswith("win"):
            try:
                class _MOUSEINPUT(ctypes.Structure):
                    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
                class _INPUTUNION(ctypes.Union):
                    _fields_ = [("mi", _MOUSEINPUT)]
                class _INPUT(ctypes.Structure):
                    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]
                button_data = 1 if int(direction) < 0 else 2
                inputs = (_INPUT * 2)(
                    _INPUT(0, _INPUTUNION(_MOUSEINPUT(0, 0, button_data, 0x0080, 0, 0))),
                    _INPUT(0, _INPUTUNION(_MOUSEINPUT(0, 0, button_data, 0x0100, 0, 0))),
                )
                return bool(ctypes.windll.user32.SendInput(2, inputs, ctypes.sizeof(_INPUT)))
            except Exception:
                pass
        return False

    def _scroll_target_window(self, active, steps, cursor=None, horizontal=False):
        """Scroll at the tracked point without moving a detached hand's OS cursor.

        `steps` is in raw wheel units: 120 = one wheel notch (Windows WHEEL_DELTA).
        Previously Windows treated it as raw units via pyautogui but the detached path
        multiplied by 120, so normal scrolling was 1/120th the intended speed.

        Attached scrolling follows physical-mouse behaviour: move the pointer to the
        tracked TLI and inject a wheel event. Detached hands keep their native pointer
        position; on Windows, route WM_MOUSEWHEEL to the window under the tracked point
        instead. Platforms without targeted wheel-message support still scroll at the
        current OS pointer, but never reposition it from this detached gesture.
        """
        if not steps:
            return
        point = cursor or active.get("last_cursor") or active.get("origin")
        screen_point = None
        if point is not None:
            screen_point = self._virtual_point_to_global(point)
        wheel_steps = int(steps) if self.invert_scroll else -int(steps)
        side = active.get("side")
        message_id = 0x020E if horizontal else 0x020A  # WM_MOUSEHWHEEL / WM_MOUSEWHEEL
        detached = bool(side in ("left", "right") and
                        self.mouse_detached_by_side.get(side, False))

        # Windows supports targeting the control under the tracked point without moving
        # the actual cursor. This prevents detached drag-scroll from covertly moving it.
        if detached and screen_point is not None and sys.platform.startswith("win"):
            try:
                user32 = ctypes.windll.user32
                native_point = wintypes.POINT(int(screen_point.x()), int(screen_point.y()))
                window_from_point = user32.WindowFromPoint
                window_from_point.argtypes = [wintypes.POINT]
                window_from_point.restype = wintypes.HWND
                target = window_from_point(native_point)
                own_window = int(self.winId())
                if (not target or int(target) == own_window) and active.get("hwnd"):
                    target = wintypes.HWND(int(active["hwnd"]))
                if target:
                    delta = int(max(-1200, min(1200, wheel_steps)))
                    wparam = (delta & 0xFFFF) << 16
                    lparam = ((int(screen_point.y()) & 0xFFFF) << 16) | (int(screen_point.x()) & 0xFFFF)
                    result = ctypes.c_size_t(0)
                    send_timeout = user32.SendMessageTimeoutW
                    send_timeout.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                             ctypes.c_ssize_t, wintypes.UINT, wintypes.UINT,
                                             ctypes.POINTER(ctypes.c_size_t)]
                    send_timeout.restype = ctypes.c_size_t
                    ok = send_timeout(wintypes.HWND(int(target)), message_id, wparam, lparam,
                                      0x0002, 30, ctypes.byref(result))  # SMTO_ABORTIFHUNG
                    if ok:
                        return
            except Exception:
                # Fall back to a wheel event at the current pointer without moving it.
                pass

        try:
            if pyautogui is not None:
                if screen_point is not None and not detached:
                    pyautogui.moveTo(screen_point.x(), screen_point.y(), duration=0, _pause=False)
                clicks = int(round(wheel_steps/120.0))
                if horizontal:
                    if clicks and hasattr(pyautogui, "hscroll"):
                        pyautogui.hscroll(clicks, _pause=False)
                elif sys.platform.startswith("win"):
                    # pyautogui on Windows passes vertical wheel units directly.
                    pyautogui.scroll(wheel_steps, _pause=False)
                elif clicks:
                    pyautogui.scroll(clicks, _pause=False)
                return
            if sys.platform.startswith("win"):
                # Minimal SendInput fallback. Detached mode deliberately does not move
                # the pointer, even if the direct target-window message was unavailable.
                class _MOUSEINPUT(ctypes.Structure):
                    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
                class _INPUTUNION(ctypes.Union):
                    _fields_ = [("mi", _MOUSEINPUT)]
                class _INPUT(ctypes.Structure):
                    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]
                wheel_delta = int(max(-1200, min(1200, wheel_steps)))
                mouse = _MOUSEINPUT(0, 0, wheel_delta & 0xFFFFFFFF, 0x01000 if horizontal else 0x0800, 0, 0)
                payload = _INPUT(0, _INPUTUNION(mi=mouse))
                ctypes.windll.user32.SendInput(1, ctypes.byref(payload), ctypes.sizeof(_INPUT))
                return
            if screen_point is not None and sys.platform.startswith("linux"):
                # xdotool can issue wheel clicks at the current pointer. Do not use its
                # mousemove command for a detached hand.
                command = ["xdotool"]
                if not detached:
                    command += ["mousemove", str(screen_point.x()), str(screen_point.y())]
                command += ["click", "--repeat", str(max(1, min(20, round(abs(wheel_steps)/120.0)))),
                            "--delay", "0", "4" if wheel_steps > 0 else "5"]
                try:
                    subprocess.run(command, check=True, timeout=0.35,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception as exc:
            self.status = f"Native scroll input failed: {exc}"

    def _begin_action_from_popup(self, side, action, cursor, now):
        """Activate a selected input mode; a timed pinch returns to normal mode."""
        self._set_pinch_menu_open(side, False, now)
        if action == "right_click":
            # The menu selection gesture can overlap the native context-menu opening.
            # Ignore outside-dismiss gestures briefly so the OS menu stays open.
            self.pinch_menu_dismiss_block_until = now + 2.0
            self.right_click_mode_enabled = False
            self.scroll_mode_enabled = False
            # Right-click must target the original TLI captured when the Pinch Menu
            # opened, not the secondary TLI used to hover/select its buttons.
            original_tli = self.fist_origin_points.get(side)
            if original_tli is not None:
                target = QPointF(original_tli)
                self._activate_canvas_right_click(target, side, now)
            else:
                self._activate_canvas_right_click(QPointF(cursor), side, now)
            self.status = "Right-click sent at the Pinch Menu selection point (original TLI)"
        elif action == "scroll_window":
            self.right_click_mode_enabled = False
            self.scroll_mode_enabled = True
            self.status = "Scroll Window mode active — hold a timed pinch to return to normal"
        elif action == "glass_menu":
            self.right_click_mode_enabled = False
            self.scroll_mode_enabled = False
            self._set_menu_open(not (self.menu_open and self.hub_visible), now, reset_settings=True)
            self.status = "Glass menu " + ("shown" if self.menu_open and self.hub_visible else "hidden")
        elif action == "dictation_toggle":
            self._toggle_focused_dictation_from_menu()
        elif action == "stt_toggle":
            self._toggle_stt_from_menu()
        else:
            self.status = "Unknown Pinch Menu option"

        # Selection only changes mode/visibility; it never clicks through to the desktop.
        self._reset_fist_gesture(side)
        self.active_action[side] = None
        # Consume the grab/pinch used to select the menu item. The same held pose must
        # be released before it can begin the selected mode's normal interaction.
        self.action_stop_wait_open[side] = True
        self.action_stop_open_since[side] = None
        self.menu_pinch_started[side] = None
        self.menu_pinch_accumulated_seconds[side] = 0.0
        self.menu_pinch_pulse_started_at[side] = None
        self.menu_pinch_origin_points[side] = None
        self.menu_pinch_release_candidate_since[side] = None
        self.menu_pinch_click_fired[side] = False
        self.menu_pinch_latched.discard(side)
        self.pin_menu_ready_for_selection[side] = False
        self.pin_menu_selection_started_at[side] = None
        self.pin_menu_selection_kind[side] = None
        self.pin_menu_selection_option[side] = None
        if self.gesture_owner == side:
            self.gesture_owner = None
        self.update()

    def _update_canvas_slider(self, name, point):
        definition = next((item for item in self._settings_slider_specs() if item[0] == name), None)
        geometry = self._settings_slider_layouts().get(name)
        if definition is None or geometry is None:
            return QPointF(point)
        left, right, track_y, _label_rect = geometry
        ratio = min(1.0, max(0.0, (point.x()-left)/max(1.0, right-left)))
        _name, _offset, _old_value, minimum, maximum, _label = definition
        value = minimum+ratio*(maximum-minimum)
        if name == "ui_font_scale":
            value /= 100.0
        attributes = {
            "hand_scale": ("hand_visual_scale", 0.005),
            "hand_recognition_threshold": ("hand_recognition_threshold", 0.01),
            "sensitivity": ("hand_tracking_sensitivity", 0.01),
            "scroll_acceleration": ("scroll_acceleration", 0.01),
            "scroll_deadzone": ("scroll_deadzone", 0.5),
            "scroll_backdrop_opacity": ("scroll_backdrop_opacity", 0.5),
            "scroll_backdrop_hue": ("scroll_backdrop_hue", 1.0),
            "scroll_notch_hue": ("scroll_notch_hue", 1.0),
            "detached_notice_opacity": ("detached_notice_opacity", 0.5),
            "ui_font_scale": ("ui_font_scale", 0.005),
            "custom_gesture_hold_seconds": ("custom_gesture_hold_seconds", 0.01),
            "custom_gesture_tolerance": ("custom_gesture_tolerance", 0.002),
            "smoothing": ("cursor_smoothing", 0.005),
            "pinch_timer": ("menu_pinch_hold_seconds", 0.01),
            "default_gesture_hold_seconds": ("default_gesture_hold_seconds", 0.01),
            "pinch_buffer_seconds": ("pinch_buffer_seconds", 0.01),
            "hand_transparency": ("hand_transparency", 1.0),
            "skin_tone_darkness": ("skin_tone_darkness", -50.5),
            "finger_thickness": ("finger_thickness", 0.01),
            "palm_width_scale": ("palm_width_scale", 0.01),
            "tli_glow_intensity": ("tli_glow_intensity", 0.01),
            "tli_colour_shift_degrees": ("tli_colour_shift_degrees", 0.5),
            "tli_border_thickness": ("tli_border_thickness", 0.05),
            "tli_diameter": ("tli_diameter", 0.5),
            "skeleton_line_thickness": ("skeleton_line_thickness", 0.02),
            "hand_contact_band_ratio": ("hand_contact_band_ratio", 0.005),
            "palm_attract_acceleration": ("palm_attract_acceleration", 0.02),
        }
        if name == "tts_speed":
            if abs(value-float(self.tts.settings["speed"])) > 0.015:
                self.tts.update_settings(speed=value, persist=False)
                self.tts_settings_save_timer.start(450)
        else:
            attr, threshold = attributes[name]
            old_value = float(getattr(self, attr))
            if abs(value-old_value) >= threshold:
                setattr(self, attr, value)
                if name == "default_gesture_hold_seconds":
                    self.menu_pinch_hold_seconds = value
                    self.peace_toggle_hold_seconds = value
                    self.gesture_tracking_toggle_hold_seconds = value
                if name == "ui_font_scale":
                    self.ui_preferences["ui_font_scale"] = value
                    save_ui_preferences(self.ui_preferences)
                    self.update()
                else:
                    self.schedule_gesture_settings_save()
        return QPointF(left+ratio*(right-left), track_y)

    def _cycle_hand_colour(self, side):
        palette = ["#5091ff", "#ff5555", "#ffb347", "#65dd8a", "#d78bff", "#ffffff"]
        attribute = "left_click_color" if side == "left" else "right_click_color"
        current = getattr(self, attribute).name().lower()
        next_value = palette[(palette.index(current)+1) % len(palette)] if current in palette else palette[0]
        setattr(self, attribute, QColor(next_value))
        self.schedule_gesture_settings_save()

    def _gesture_ui_modal_open(self):
        """True while glass UI owns hand gestures; never click/drag OS content behind it."""
        return bool((getattr(self, "menu_open", False) and getattr(self, "hub_visible", False))
                    or getattr(self, "settings_open", False))

    def _activate_canvas_click(self, point, side, now):
        """Route a discrete gesture click to glass controls or the native OS pointer."""
        if point is None:
            return
        # An open glass menu is modal for gesture clicks: a pinch outside its panel
        # must not click an underlying application or the desktop.
        if self.menu_open and self.hub_visible and not self._canvas_menu_panel_contains(point):
            return
        self.click_side = side
        self.click_flash_until = now+0.24
        self.last_active = now
        cx, cy = self.hub_pos.x()*self.width(), self.hub_pos.y()*self.height()
        scale = self.menu_ui_scale()

        if self._canvas_menu_panel_contains(point):
            self._glass_pressed_point = QPointF(point)
            self._glass_pressed_at = now
            self.update()
            if now < self.menu_input_block_until:
                return
            if not self.settings_open:
                if self._main_menu_close_rect(cx, cy, scale).contains(point):
                    self._set_menu_open(False, now, reset_settings=True)
                    self.status = "Glass menu hidden"
                    return
                tiles = [
                    (QRectF(cx-166*scale, cy-145*scale, 100*scale, 88*scale), "desktop"),
                    (QRectF(cx-50*scale, cy-145*scale, 100*scale, 88*scale), "gesture_map"),
                    (QRectF(cx+66*scale, cy-145*scale, 100*scale, 88*scale), "speech_commands"),
                    (QRectF(cx-166*scale, cy+35*scale, 100*scale, 88*scale), "voice"),
                    (QRectF(cx-50*scale, cy+35*scale, 100*scale, 88*scale), "settings"),
                    (QRectF(cx+66*scale, cy+35*scale, 100*scale, 88*scale), "search"),
                ]
                action = next((name for rect, name in tiles if rect.contains(point)), None)
                if action == "desktop":
                    self.desktop_mode = "os" if self.desktop_mode == "virtual" else "virtual"
                    self.apply_desktop_mode()
                    self.status = "OS desktop input" if self.desktop_mode == "os" else "Virtual desktop"
                elif action == "voice":
                    if self.stt_enabled:
                        self.speech.stop()
                        self.stt_enabled = False
                        self.status = "Voice input off"
                    else:
                        self.stt_enabled = bool(self.speech.start())
                        if self.stt_enabled:
                            self.status = "Voice input on — listening locally with Vosk"
                        else:
                            self.status = "Voice input could not start"
                    self._announce_voice_input_state(self.stt_enabled)
                elif action == "settings":
                    self.settings_open = True
                    self.settings_page = "interaction"
                    self._begin_menu_transition(now)
                elif action == "search":
                    self.open_search_bar(from_voice=False)
                elif action == "gesture_map":
                    self.settings_open = True
                    self.settings_page = "gesture_map"
                    self._begin_menu_transition(now)
                elif action == "speech_commands":
                    self.settings_open = True
                    self.settings_page = "speech_commands"
                    self._begin_menu_transition(now)
                return  # every press in the glass panel is consumed, even blank space

            # An open custom-gesture action list takes every press until it closes.
            if self.settings_page == "gestures" and self.custom_gesture_dropdown is not None:
                self._click_custom_gesture_dropdown(point, cx, cy, scale)
                return

            # Back and the three Settings sub-pages share a stable navigation row.
            if self._settings_back_rect(cx, cy, scale).contains(point):
                self.settings_open = False
                self.settings_page = "interaction"
                self._begin_menu_transition(now)
                return
            if self.settings_page not in ("gesture_map", "speech_commands"):
                for page_name, tab_rect in self._settings_tab_rects(cx, cy, scale).items():
                    if tab_rect.contains(point):
                        # Keep the settings panel continuously visible while navigating.
                        # Its dimensions are recalculated from the selected page, so it can
                        # grow or shrink to fit that page without replaying the menu fade or
                        # temporarily blocking input.
                        # Keep the old page visible for the first half of the
                        # transition. The target page is swapped in at zero opacity
                        # halfway through, then fades up while the panel flexes.
                        previous_size = self._menu_panel_dimensions()
                        self.settings_page_pending = page_name
                        self._begin_settings_page_transition(previous_size, now)
                        self.tli_overrides.pop(side, None)
                        self.update()
                        return

            slider_name = self._canvas_slider_name_at(point)
            if slider_name:
                reticle = self._update_canvas_slider(slider_name, point)
                self.tli_overrides[side] = ("slider", reticle)
                return

            controls = self._settings_control_rects(cx, cy, scale)

            def pressed(name):
                rect = controls.get(name)
                return rect is not None and rect.contains(point)

            if self.settings_page == "interaction":
                if pressed("scroll_toggle"):
                    self.invert_scroll = not self.invert_scroll
                    self.schedule_gesture_settings_save()
                    self.status = "Scroll direction inverted" if self.invert_scroll else "Scroll direction normal"
                elif pressed("swap_drag_scroll_toggle"):
                    self.swap_drag_scroll_notches = not self.swap_drag_scroll_notches
                    self.schedule_gesture_settings_save()
                    self.status = ("Drag-scroll Back/Forward notches swapped" if self.swap_drag_scroll_notches
                                   else "Drag-scroll Back/Forward notches normal")
                elif pressed("priority_toggle"):
                    self.cursor_lock_priority = "left" if self.cursor_lock_priority == "right" else "right"
                    self.schedule_gesture_settings_save()
                    self.status = f"Cursor lock priority: {self.cursor_lock_priority.title()} hand"
                return

            if self.settings_page == "tli":
                # Per-hand TLI colour swatches live on this page, away from the
                # crowded Hands-page recognition threshold and calibration controls.
                if pressed("swatch_left"):
                    self._cycle_hand_colour("left")
                    self.status = "Left TLI colour changed"
                elif pressed("swatch_right"):
                    self._cycle_hand_colour("right")
                    self.status = "Right TLI colour changed"
                elif pressed("glow_toggle"):
                    self.tli_glow_enabled = not self.tli_glow_enabled
                    self.schedule_gesture_settings_save()
                    self.status = "TLI glow enabled" if self.tli_glow_enabled else "TLI glow disabled"
                return

            if self.settings_page == "gestures":
                self._click_custom_gestures_page(point, cx, cy, scale, now)
                return

            if self.settings_page == "hands":
                if pressed("render_style"):
                    # Three-way switch: the third of the control that was pressed.
                    style = next((name for name, seg in self._render_style_segments(controls["render_style"], scale)
                                  if seg.contains(point)), None)
                    if style is None:  # the "RENDER" caption: step to the next style
                        index = HAND_RENDER_STYLES.index(self.hand_render_style)
                        style = HAND_RENDER_STYLES[(index+1) % len(HAND_RENDER_STYLES)]
                    if style != self.hand_render_style:
                        self.hand_render_style = style
                        self.hand_skin_effect_enabled = style != "skeleton"
                        self._hand_mesh_notice_shown = False
                        if self._hand_mesh_renderer is not None and self._hand_mesh_renderer.failed:
                            self._hand_mesh_renderer = None  # retry 3D after a previous failure
                        self.schedule_gesture_settings_save()
                    self.status = "Hand rendering: " + HAND_RENDER_STYLE_LABELS[style].lower()
                    return
                if pressed("skin_tint"):
                    self.skin_use_skeleton_color = not self.skin_use_skeleton_color
                    self.schedule_gesture_settings_save()
                    self.status = "Skin uses skeleton colour" if self.skin_use_skeleton_color else "Skin uses natural colour"
                    return
                for calibration_side, calibration_rect in self._tli_calibration_button_rects(cx, cy, scale).items():
                    if calibration_rect.contains(point):
                        self._arm_tli_calibration(calibration_side)
                        return
                if pressed("hand_setup"):
                    self._start_hand_onboarding(first_run=False)
                    return
                return

            if self.settings_page == "notifications":
                for key, _label, _default in NOTIFICATION_SETTING_ROWS:
                    if pressed("notify:" + key):
                        self._cycle_notification_position(key)
                        break
                return

            if self.settings_page == "voice":
                if pressed("tts_toggle"):
                    self.tts.set_enabled(not self.tts.enabled)
                    self.status = "Local TTS " + ("enabled" if self.tts.enabled else "disabled")
                elif pressed("voice_cycle"):
                    current = self.tts.settings.get("voice", "bf_emma")
                    try:
                        next_index = (TTS_VOICE_OPTIONS.index(current)+1) % len(TTS_VOICE_OPTIONS)
                    except ValueError:
                        next_index = 0
                    self.tts.set_voice(TTS_VOICE_OPTIONS[next_index])
                    self.status = f"TTS voice set to {TTS_VOICE_OPTIONS[next_index]}"
                elif pressed("test_voice"):
                    if not self.tts.speak("Hello. This is a test of your selected local voice."):
                        self.status = f"TTS is not ready: {self.tts.status}"
                elif pressed("face_welcome"):
                    self.manage_face_welcome()
                elif pressed("streamtts_config"):
                    config_gui = Path(__file__).resolve().parent / "streamtts_config.py"
                    if config_gui.is_file():
                        try:
                            subprocess.Popen([sys.executable, str(config_gui)], cwd=str(config_gui.parent))
                            self.status = "Opened StreamTTS configuration."
                        except Exception as exc:
                            self.status = f"Could not open StreamTTS configuration: {exc}"
                    else:
                        self.status = "StreamTTS configuration tool is missing."
                return

            if self.settings_page == "update":
                if pressed("update_action") and not self.update_action_running and not self.update_check_running:
                    if self.update_installed:
                        self._restart_spatial_desktop()
                    elif self.update_download_path:
                        self._start_update_install()
                    else:
                        self._start_update_check()
                return

            if self.settings_page == "d1":
                if pressed("d1_model"):
                    # Keep the supported default pinned to d1-omni-600M Q8.
                    # Clicking this row ensures the default model is selected/downloaded;
                    # the larger Q4 option is intentionally unavailable for now.
                    ready = select_d1_model("600m")
                    spec = D1_MODELS["600m"]
                    self.status = (f"d1 model selected: {spec['label']}" if ready else
                                   f"Downloading {spec['estimated']} d1 model in background.")
                    self._log_d1_ui("Model ensured: " + ("available" if ready else "download requested"))
                elif pressed("d1_start"):
                    start_d1_server()
                    self.status = "Starting local d1 runtime."
                    self._log_d1_ui("Start / ensure running requested")
                elif pressed("d1_stop"):
                    stop_d1_server()
                    self.status = "Local d1 runtime stopped."
                    self._log_d1_ui("Stop local server requested")
                elif pressed("d1_retry"):
                    start_d1_loading()
                    self.status = "Checking d1 model and runtime; download/start requested."
                    self._log_d1_ui("Download / retry / start requested")
                return
            return

        # Detached hands can still operate Spatial Desktop's own glass controls, but
        # an outside-menu pinch must never move/click the native OS pointer.
        if self.mouse_detached_by_side.get(side, False):
            return
        if self.desktop_mode == "os" and pyautogui is not None:
            try:
                global_pos = self._virtual_point_to_global(point)
                QCursor.setPos(global_pos)
                self.last_os_cursor_pos = global_pos
                pyautogui.click(button="left", _pause=False)
                self.right_click_context_active[side] = False
            except Exception as exc:
                self.status = f"OS click failed: {exc}"

    def _activate_quick_pinch_click(self, point, side, now):
        """Emit one click for a quick pinch and align successive clicks for double-click.

        The first click is sent immediately; a second pinch at nearly the same point is
        snapped back to the first point, allowing the operating system to recognise its
        ordinary native double-click sequence. This state is deliberately independent
        of the Pinch Menu timer and the quick-grasp tracker.
        """
        if point is None:
            return
        click_point = QPointF(point)
        native_target = (self.desktop_mode == "os" and
                         not self.mouse_detached_by_side.get(side, False) and
                         not self.menu_open and
                         not self._canvas_menu_panel_contains(click_point))
        previous = self.last_quick_pinch_click.get(side)
        if native_target and previous is not None:
            previous_time, previous_point = previous
            aligned, is_second_click = align_double_click_point(
                previous_time,
                (previous_point.x(), previous_point.y()),
                now,
                (click_point.x(), click_point.y()),
                self.quick_pinch_double_click_window,
                90.0,
            )
            click_point = QPointF(aligned[0], aligned[1])
            if is_second_click:
                self.last_quick_pinch_click[side] = None
            else:
                self.last_quick_pinch_click[side] = (float(now), QPointF(click_point))
        elif native_target:
            self.last_quick_pinch_click[side] = (float(now), QPointF(click_point))
        else:
            self.last_quick_pinch_click[side] = None
        self._activate_canvas_click(click_point, side, now)

    def _activate_canvas_right_click(self, point, side, now):
        """Send a right-click at the captured TLI point (used by Right-click Mode)."""
        if point is None or self._canvas_menu_panel_contains(point):
            return
        if self.menu_open and self.hub_visible:
            return
        self.click_side = side
        self.click_flash_until = now + 0.24
        self.last_active = now
        # A detached hand may still operate glass UI, but must never reposition
        # or click the native OS pointer.
        if (self.desktop_mode == "os" and pyautogui is not None
                and not self.mouse_detached_by_side.get(side, False)):
            try:
                global_pos = self._virtual_point_to_global(point)
                QCursor.setPos(global_pos)
                self.last_os_cursor_pos = global_pos
                pyautogui.click(button="right", _pause=False)
                self.right_click_context_active[side] = True
            except Exception as exc:
                self.status = f"OS right-click failed: {exc}"

    def _virtual_point_to_global(self, point):
        """Convert a virtual-hand point to physical desktop coordinates.

        interactive_visual_point() maps landmarks in primary-screen-local pixels.
        Qtile can position a Qt.Tool fullscreen overlay at the work-area origin (below
        its top bar), so mapToGlobal() would incorrectly add that bar offset and make
        the cursor unable to reach the top bar. Convert from the actual QScreen origin
        on Linux; use the native Qt mapping on other platforms.
        """
        if point is None:
            return None
        pixel = QPoint(round(point.x()), round(point.y()))
        if sys.platform.startswith("linux"):
            screen = QApplication.primaryScreen()
            geometry = screen.geometry() if screen is not None else None
            if geometry is not None:
                return geometry.topLeft() + pixel
        return self.mapToGlobal(pixel)

    def _set_os_pointer(self, point):
        if point is None or self._gesture_ui_modal_open():
            return
        global_pos = self._virtual_point_to_global(point)
        if global_pos is None:
            return
        QCursor.setPos(global_pos)
        self.last_os_cursor_pos = global_pos

    def _release_fist_mouse(self, side):
        if self.fist_mouse_down.get(side) and pyautogui is not None:
            try:
                pyautogui.mouseUp(button="left", _pause=False)
            except Exception:
                pass
        self.fist_mouse_down[side] = False

    def _begin_pinch_drag(self, side, cursor, now):
        """Begin a native drag, with a small forgiving capture zone for resize borders."""
        if self._gesture_ui_modal_open():
            return False
        if self.mouse_detached_by_side.get(side, False):
            # Preserve pinch/glass interactions for detached hands without allowing
            # the native cursor or mouse button to follow that hand.
            return False
        if self.desktop_mode != "os":
            self.status = "Switch to OS Desktop mode to drag desktop objects with a pinch"
            return False
        if pyautogui is None:
            self.status = "Drag unavailable: install the mouse automation dependency"
            return False
        try:
            drag_cursor = QPointF(cursor)
            # Native resize borders are often only a few physical pixels wide. The TLI
            # is smoothed and may land just inside the client area, where a held click
            # starts an ordinary content drag instead of resizing. For sizeable windows,
            # gently snap a pinch-drag start to the nearest border only when already
            # within this small capture zone; normal interior drags remain untouched.
            if sys.platform.startswith("win"):
                hwnd = self._find_native_window_at(drag_cursor)
                rect = self._native_window_rect(hwnd) if hwnd else None
                if hwnd and rect:
                    try:
                        user32 = ctypes.windll.user32
                        # Do not steal scrollbar interactions: fist scrolling relies on
                        # the native scrollbar/client hit target remaining untouched.
                        class_buf = ctypes.create_unicode_buffer(128)
                        user32.GetClassNameW(wintypes.HWND(int(hwnd)), class_buf, len(class_buf))
                        if "scrollbar" in class_buf.value.lower():
                            hwnd = None
                        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
                        get_style.argtypes = [wintypes.HWND, ctypes.c_int]
                        get_style.restype = ctypes.c_ssize_t
                        style = int(get_style(wintypes.HWND(int(hwnd)), -16)) if hwnd else 0
                        if hwnd and style & 0x00040000:  # WS_THICKFRAME / WS_SIZEBOX
                            global_pos = self.mapToGlobal(QPoint(round(drag_cursor.x()), round(drag_cursor.y())))
                            gx, gy = float(global_pos.x()), float(global_pos.y())
                            left, top, right, bottom = map(float, rect)
                            # Match the forgiving edge zone to the configured TLI diameter.
                            capture = max(12.0, float(getattr(self, "tli_diameter", 30.0)))
                            candidates = []
                            if top <= gy < bottom and abs(gx-left) <= capture:
                                candidates.append((abs(gx-left), left+2.0, gy))
                            if top <= gy < bottom and abs(gx-right) <= capture:
                                candidates.append((abs(gx-right), right-3.0, gy))
                            if left <= gx < right and abs(gy-top) <= capture:
                                candidates.append((abs(gy-top), gx, top+2.0))
                            if left <= gx < right and abs(gy-bottom) <= capture:
                                candidates.append((abs(gy-bottom), gx, bottom-3.0))
                            if candidates:
                                _distance, snap_x, snap_y = min(candidates, key=lambda item: item[0])
                                local = self.mapFromGlobal(QPoint(round(snap_x), round(snap_y)))
                                drag_cursor = QPointF(local.x(), local.y())
                    except Exception:
                        pass
            self._set_os_pointer(drag_cursor)
            pyautogui.mouseDown(button="left", _pause=False)
            self.fist_mouse_down[side] = True
            self.pinch_drag_active[side] = True
            self.pinch_drag_last_points[side] = QPointF(drag_cursor)
            self.pinch_drag_release_candidate_since[side] = None
            self._clear_menu_pinch_sequence(side)
            self.gesture_owner = side
            self.canvas_active_side = side
            self.active_push_sides.add(side)
            self.last_active = float(now)
            self.status = "Click-and-hold drag active — move the pinched hand; release to drop"
            return True
        except Exception as exc:
            try:
                pyautogui.mouseUp(button="left", _pause=False)
            except Exception:
                pass
            self.fist_mouse_down[side] = False
            self.pinch_drag_active[side] = False
            self.pinch_drag_origin_points[side] = None
            self.pinch_drag_last_points[side] = None
            self.pinch_drag_release_candidate_since[side] = None
            self.status = f"Could not start pinch drag: {exc}"
            return False

    def _end_pinch_drag(self, side, status=True):
        """Release a pinch-held mouse button safely and clear drag state."""
        was_active = bool(self.pinch_drag_active.get(side, False))
        self.pinch_drag_active[side] = False
        self.pinch_drag_origin_points[side] = None
        self.pinch_drag_last_points[side] = None
        self.pinch_drag_release_candidate_since[side] = None
        self._release_fist_mouse(side)
        self.active_push_sides.discard(side)
        self._clear_menu_pinch_sequence(side)
        if self.gesture_owner == side:
            self.gesture_owner = None
        if was_active and status:
            self.status = "Drag released — object dropped"
        return was_active

    def _needs_tracking_loss_grace(self, side):
        """Keep active manipulation, Pinch Menu selection, and context menus through brief camera loss."""
        active = self.active_action.get(side)
        if active is not None and active.get("mode") in ("right_click", "drag_window", "scroll_window"):
            return True
        if self.right_click_context_active.get(side, False):
            return True
        if self.fist_action_menu_open.get(side, False):
            return True
        return bool(self.fist_hold_armed.get(side, False) and
                    self.fist_action.get(side) in ("slider", "menu_drag", "canvas_drag"))

    def _cancel_side_interactions_for_tracking_pause(self, side):
        """Stop every interaction for a disabled hand but keep its last TLI position frozen."""
        self._end_pinch_drag(side, status=False)
        self._release_fist_mouse(side)
        self._reset_fist_gesture(side, release_mouse=False)
        self.active_action[side] = None
        self.action_stop_wait_open[side] = False
        self.action_stop_open_since[side] = None
        self._clear_menu_pinch_sequence(side)
        self.menu_pinch_started[side] = None
        self.menu_pinch_accumulated_seconds[side] = 0.0
        self.menu_pinch_pulse_started_at[side] = None
        self.menu_pinch_origin_points[side] = None
        self.menu_pinch_release_candidate_since[side] = None
        self.menu_pinch_click_fired[side] = False
        self.menu_pinch_latched.discard(side)
        self._clear_modifier_click(side)
        self.right_click_context_active[side] = False
        self.active_push_sides.discard(side)
        self.cursor_updated_at[side] = 0.0
        self.hand_missing_since[side] = None
        if self.gesture_owner == side:
            self.gesture_owner = None
        if self.canvas_active_side == side:
            self.canvas_active_side = None
        if side == "left":
            self.left_click_modifier_active = False
            self.left_modifier_fist_frames = 0
            self.left_modifier_open_frames = 0
            self.left_modifier_last_fist_at = None
            self.left_modifier_release_candidate_since = None

    def _reset_fist_gesture(self, side, release_mouse=True):
        if release_mouse:
            self._release_fist_mouse(side)
        marker = getattr(self, "fist_scroll_markers", {}).get(side)
        if marker is not None:
            marker["until"] = time.monotonic() + 1.25
        self.fist_started_at[side] = None
        self.fist_release_candidate_since[side] = None
        self.fist_open_candidate_since[side] = None
        self.fist_origin_points[side] = None
        self.fist_arm_points[side] = None
        self.fist_hold_armed[side] = False
        self.fist_action[side] = None
        self.fist_last_points[side] = None
        self.fist_start_controls[side] = None
        self.fist_slider_offset_x[side] = 0.0
        self.slider_last_good_cursor[side] = None
        self.slider_cursor_jump_candidate[side] = None
        self.fist_scroll_remainder[side] = 0.0
        self.fist_action_menu_open[side] = False
        self.fist_hovered_action[side] = None
        self.pin_menu_ready_for_selection[side] = False
        self.pin_menu_selection_started_at[side] = None
        self.pin_menu_selection_kind[side] = None
        self.pin_menu_selection_option[side] = None
        self.fist_target_window[side] = None
        self.fist_target_window_resolved[side] = False
        self.fist_started_on_taskbar[side] = False
        self.fist_start_hit_test[side] = None
        self.fist_shake_samples[side] = []
        self.fist_hold_ready_notified[side] = False

    def on_face_status(self, text: str):
        self.face_enrollment_status = str(text)[:160]
        self.status = self.face_enrollment_status
        self.update()

    def manage_face_welcome(self):
        """Explicitly enroll, replace, or remove the local face welcome profile."""
        # Native OS dialogs can otherwise be covered by the always-on-top 3D overlay.
        if self.desktop_mode != "os":
            self.desktop_mode = "os"
            self.apply_desktop_mode()
        # Hide the glass controls before native dialogs appear, then give the OS dialog focus.
        self._set_menu_open(False, reset_settings=True)
        existing = self.face_welcome.profile_name()
        if existing:
            box = QMessageBox(self)
            box.setWindowTitle("Face Welcome")
            box.setWindowFlags(box.windowFlags() | Qt.WindowStaysOnTopHint)
            box.setIcon(QMessageBox.Question)
            box.setText(f"Local face welcome is enrolled for {existing}.")
            box.setInformativeText("Choose Re-enrol to replace the local face template, Remove to delete it, or Cancel.")
            reenroll = box.addButton("Re-enrol", QMessageBox.AcceptRole)
            remove = box.addButton("Remove profile", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.raise_()
            box.activateWindow()
            box.exec()
            clicked = box.clickedButton()
            if clicked is remove:
                confirmation_box = QMessageBox(self)
                confirmation_box.setWindowTitle("Remove Face Welcome")
                confirmation_box.setWindowFlags(confirmation_box.windowFlags() | Qt.WindowStaysOnTopHint)
                confirmation_box.setIcon(QMessageBox.Warning)
                confirmation_box.setText("Delete the saved face embedding and disable welcome-back recognition? No photos are stored.")
                confirmation_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                confirmation_box.setDefaultButton(QMessageBox.No)
                confirmation_box.raise_()
                confirmation_box.activateWindow()
                confirmation = confirmation_box.exec()
                if confirmation == QMessageBox.Yes:
                    self.face_welcome.remove_profile()
                    self.face_enrollment_status = "Face welcome profile removed"
                    self.update()
                return
            if clicked is not reenroll:
                return

        consent_box = QMessageBox(self)
        consent_box.setWindowTitle("Enable Local Face Welcome")
        consent_box.setWindowFlags(consent_box.windowFlags() | Qt.WindowStaysOnTopHint)
        consent_box.setIcon(QMessageBox.Question)
        consent_box.setText(
            "Spatial Desktop will process camera frames locally to recognise the enrolled user after a long absence. "
            "It stores a numeric face embedding and display name in your local .spatial_desktop folder; it does not save face photos or send images to a server. "
            "This is a convenience feature, not secure authentication. Continue?"
        )
        consent_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        consent_box.setDefaultButton(QMessageBox.No)
        consent_box.raise_()
        consent_box.activateWindow()
        consent = consent_box.exec()
        # Previous QMessageBox.question call replaced by explicit frontmost dialog.
        if consent != QMessageBox.Yes:
            return
        name_dialog = QInputDialog(self)
        name_dialog.setWindowTitle("Enroll Face")
        name_dialog.setLabelText("Name to speak when you return:")
        name_dialog.setWindowFlags(name_dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        name_dialog.raise_()
        name_dialog.activateWindow()
        accepted = name_dialog.exec() == QInputDialog.Accepted
        username = name_dialog.textValue()
        if not accepted or not username.strip():
            return
        if self.face_welcome.begin_enrollment(username.strip()):
            # No modal follow-up dialog: the face worker downloads/prepares models
            # in the background and reports progress in Settings without blocking
            # the camera or closing the application.
            self.face_enrollment_status = "Preparing face recognition automatically…"
            self.status = self.face_enrollment_status
            self.update()

    def speak_startup_greeting(self):
        """Speak a local time-of-day greeting, using the enrolled name when available."""
        if getattr(self, "hand_calibration_active", False):
            return
        if not self.tts.ready:
            return
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 22:
            greeting = "Good evening"
        else:
            greeting = "Good night"
        name = self.face_welcome.profile_name()
        if not name:
            name = " ".join(getpass.getuser().replace("_", " ").replace(".", " ").split()).title() or "there"
        self.tts.speak(f"{greeting}, {name}.")

    def _remember_external_foreground_window(self, force=False):
        """Remember the last focused non-Spatial-Desktop window on the current OS."""
        if sys.platform.startswith("win"):
            try:
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                if not hwnd:
                    return
                pid = wintypes.DWORD(0)
                user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
                if pid.value == os.getpid() or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
                    return
                self.last_external_foreground_hwnd = int(hwnd)
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            # Optional xdotool support lets us close the external active window by ID
            # rather than sending Alt+F4 blindly to the fullscreen overlay.
            # This runs from the scene timer, and each check launches two processes,
            # so poll only once per second rather than on every UI frame.
            now = time.monotonic()
            if not force and now - getattr(self, "_last_foreground_poll", 0.0) < 1.0:
                return
            self._last_foreground_poll = now
            if not hasattr(self, "_xdotool_available"):
                self._xdotool_available = shutil.which("xdotool") is not None
            if not self._xdotool_available:
                return
            try:
                active = subprocess.check_output(["xdotool", "getactivewindow"], text=True,
                                                 stderr=subprocess.DEVNULL, timeout=0.25).strip()
                pid_text = subprocess.check_output(["xdotool", "getwindowpid", active], text=True,
                                                   stderr=subprocess.DEVNULL, timeout=0.25).strip()
                if active and int(pid_text) != os.getpid():
                    self.last_external_window_id = active
            except Exception:
                pass

    def close_focused_window(self):
        """Close the last focused external native window, never Spatial Desktop itself."""
        if sys.platform.startswith("win"):
            try:
                user32 = ctypes.windll.user32
                candidates = []
                if self.search_dialog is not None and self.search_dialog.isVisible() and self.search_return_hwnd:
                    candidates.append(self.search_return_hwnd)
                foreground = user32.GetForegroundWindow()
                if foreground:
                    pid = wintypes.DWORD(0)
                    user32.GetWindowThreadProcessId(wintypes.HWND(foreground), ctypes.byref(pid))
                    if pid.value != os.getpid():
                        candidates.insert(0, int(foreground))
                candidates.append(self.last_external_foreground_hwnd)
                target = None
                for candidate in candidates:
                    if candidate and user32.IsWindow(wintypes.HWND(int(candidate))):
                        pid = wintypes.DWORD(0)
                        user32.GetWindowThreadProcessId(wintypes.HWND(int(candidate)), ctypes.byref(pid))
                        if pid.value != os.getpid():
                            target = int(candidate)
                            break
                if target is None:
                    self.status = "No external focused window could be identified to close."
                    self.tts.speak("I couldn't identify an open window to close.")
                    return False
                user32.PostMessageW(wintypes.HWND(target), 0x0010, 0, 0)  # WM_CLOSE
                self.status = "Close requested for the focused window"
                self.tts.speak("Closing the active window.")
                self.last_external_foreground_hwnd = None
                return True
            except Exception as exc:
                self.status = f"Could not close focused window: {exc}"
                return False

        if sys.platform.startswith("linux"):
            candidates = []
            if self.search_dialog is not None and self.search_dialog.isVisible() and self.search_return_window_id:
                candidates.append(self.search_return_window_id)
            try:
                active = subprocess.check_output(["xdotool", "getactivewindow"], text=True,
                                                 stderr=subprocess.DEVNULL, timeout=0.5).strip()
                pid_text = subprocess.check_output(["xdotool", "getwindowpid", active], text=True,
                                                   stderr=subprocess.DEVNULL, timeout=0.5).strip()
                if active and int(pid_text) != os.getpid():
                    candidates.insert(0, active)
            except Exception:
                pass
            candidates.append(self.last_external_window_id)
            for target in candidates:
                if not target:
                    continue
                try:
                    pid_text = subprocess.check_output(["xdotool", "getwindowpid", str(target)], text=True,
                                                       stderr=subprocess.DEVNULL, timeout=0.5).strip()
                    if int(pid_text) == os.getpid():
                        continue
                    subprocess.run(["xdotool", "windowclose", str(target)], check=True, timeout=2,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.status = "Close requested for the focused window"
                    self.tts.speak("Closing the active window.")
                    self.last_external_window_id = None
                    return True
                except Exception:
                    continue

        # Avoid Alt+F4/Cmd+W fallbacks here: if the native target can't be identified,
        # those shortcuts could close Spatial Desktop or this search dialog instead.
        self.status = "Could not safely identify an external focused window on this platform."
        self.tts.speak("I couldn't safely identify an external window to close.")
        return False

    def open_user_folder(self):
        """Open the current user's home directory in the native file manager."""
        home = str(Path.home())
        try:
            if sys.platform.startswith("win"):
                os.startfile(home)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", home])
            else:
                subprocess.Popen(["xdg-open", home])
            self.status = "Opened your user folder"
            self.tts.speak("Opening your user folder.")
            return True
        except Exception as exc:
            self.status = f"Could not open user folder: {exc}"
            self.tts.speak("I couldn't open your user folder.")
            return False

    def open_search_bar(self, from_voice=False):
        """Open one titleless local file launcher; repeated glass-menu activations reuse it."""
        if self.search_dialog is not None and self.search_dialog.isVisible():
            dialog = self.search_dialog
            dialog.raise_()
            dialog.activateWindow()
            if from_voice:
                dialog.set_dictation_enabled(True)
            return
        if self.search_dialog is None:
            self.search_dialog = SearchLauncherDialog(parent=self)
            self.search_dialog.dictation_toggled.connect(self.on_search_dictation_toggled)
            self.search_dialog.closed.connect(self.on_search_dialog_closed)
            self.search_dialog.speak_requested.connect(self.speak_search_result)
        dialog = self.search_dialog
        if not dialog.isVisible():
            self._remember_external_foreground_window(force=True)
            self.search_return_hwnd = self.last_external_foreground_hwnd
            self.search_return_window_id = self.last_external_window_id
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        # The launcher is topmost, but the click-through overlay must remain above it
        # so the user can still see the tracked skeleton, TLI, hold ring and Pinch Menu.
        # Reassert z-order after activation; this does not activate/focus the overlay.
        if getattr(self, "search_hand_overlay", None) is not None:
            self.search_hand_overlay.sync_to_launcher()
            QTimer.singleShot(60, self.search_hand_overlay.sync_to_launcher)
        if from_voice:
            # The phrase that opened search proves the Vosk worker is active; keep
            # streaming partials into the query field for the next utterance.
            dialog.set_dictation_enabled(True)
            if not self.speech.running:
                self.on_search_dictation_toggled(True)
            self.status = "Search open — live Vosk dictation"
        else:
            dialog.set_dictation_enabled(False)
            self.status = "Local search open — Ctrl+Space to toggle"

    def on_search_dictation_toggled(self, enabled):
        if enabled:
            if not self.speech.running:
                ok = self.speech.start()
                self.stt_enabled = bool(ok)
                if not ok:
                    if self.search_dialog is not None:
                        self.search_dialog.set_dictation_enabled(False)
                    return
            else:
                self.stt_enabled = True
            self.status = "Search dictation on — words stream into the search box"
        else:
            self.status = "Search dictation off"

    def on_search_dialog_closed(self):
        # Global voice commands may remain enabled after closing the launcher.
        self.status = "Search closed"

    def speak_search_result(self, text):
        if not self.tts.speak(text):
            self.status = "Local TTS is not ready — run install_voice_models.py after installing requirements-voice.txt"

    @staticmethod
    def _normalised_dictation_words(text):
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    @staticmethod
    def _format_dictation_numbers(text):
        """Convert spoken standalone number words one through nine to digits."""
        number_words = {
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        }
        return re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine)\b",
                      lambda match: number_words[match.group(0).lower()],
                      str(text or ""), flags=re.IGNORECASE)

    @staticmethod
    def _strip_end_dictation_phrase(text):
        """Keep content before a terminal 'end dictation' voice-control phrase."""
        value = (text or "").strip()
        match = re.search(r"(?i)(?:^|\s)(?:end|and)\s+dictation\s*[.!?]*$", value)
        if match:
            return value[:match.start()].strip(), True
        return value, False

    def _partial_may_be_voice_command(self, text):
        words = " ".join(self._normalised_dictation_words(text))
        if not words:
            return True
        phrases = [alias for aliases in aliases_for_help().values() for alias in aliases]
        phrases.extend(("begin dictation", "end dictation", "and dictation", "stop dictation", "pause dictation", "resume dictation", "pause tracking", "resume tracking",
                        "show menu", "show context", "search bar", "close window", "open user folder",
                        "stop listening", "stop dictation", "open top result"))
        normalized_phrases = [" ".join(self._normalised_dictation_words(phrase)) for phrase in phrases]
        # Hold exact commands and partial command prefixes until Vosk has enough context
        # to decide whether the user is speaking a command or dictating those words.
        return any(phrase.startswith(words) for phrase in normalized_phrases if phrase)

    def _type_focus_text(self, text):
        if not text or pyautogui is None:
            return False
        try:
            pyautogui.write(self._format_dictation_numbers(str(text)), interval=0, _pause=False)
            return True
        except Exception as exc:
            self.status = f"Dictation typing failed: {exc}"
            return False

    def begin_focused_dictation(self):
        """Dictate into the current focused application field, or the launcher when open."""
        dialog = self.search_dialog
        if dialog is not None and dialog.isVisible():
            dialog.set_dictation_enabled(True)
            if not self.speech.running:
                self.on_search_dictation_toggled(True)
            else:
                self.stt_enabled = True
            self.focus_dictation_enabled = False
            self.status = "Dictation started in the search field"
            return
        if not self.speech.running:
            if not self.speech.start():
                self.stt_enabled = False
                self.status = "Cannot begin dictation because local Vosk input is unavailable"
                return
        self.stt_enabled = True
        self.focus_dictation_enabled = True
        self.focus_dictation_paused = False
        self.focus_dictation_committed_words = []
        self.focus_dictation_last_partial = ""
        self.status = "Dictation active in the focused text field — say 'end dictation' to finish"

    def end_focused_dictation(self, final_text=""):
        """Commit the final stable words and stop focused-field dictation."""
        content, _ending = self._strip_end_dictation_phrase(final_text)
        words = self._normalised_dictation_words(content)
        committed = list(self.focus_dictation_committed_words)
        if words[:len(committed)] == committed:
            remainder = words[len(committed):]
            if remainder:
                self._type_focus_text(" ".join(remainder) + " ")
        elif words and not committed:
            self._type_focus_text(" ".join(words) + " ")
        self.focus_dictation_enabled = False
        self.focus_dictation_paused = False
        self.focus_dictation_committed_words = []
        self.focus_dictation_last_partial = ""
        self.status = "Dictation ended"

    def _stream_focused_dictation_partial(self, text):
        if not self.focus_dictation_enabled or self.focus_dictation_paused or not text:
            return
        content, ending = self._strip_end_dictation_phrase(text)
        words = self._normalised_dictation_words(content)
        # Keep the trailing word mutable; Vosk commonly revises it as more audio arrives.
        stable = words if ending else words[:-1]
        if self._partial_may_be_voice_command(text):
            return
        committed = self.focus_dictation_committed_words
        if stable[:len(committed)] != committed:
            # A recognizer correction to already typed stable words is rare; do not risk
            # backspacing through user text or a changed focus target to repair it.
            return
        new_words = stable[len(committed):]
        if new_words and self._type_focus_text(" ".join(new_words) + " "):
            committed.extend(new_words)
        self.focus_dictation_last_partial = text

    def on_speech_partial(self, text):
        self.speech_partial = text
        # A new phrase should immediately replace any previous unrecognised-command
        # feedback, even before the next final recognition arrives.
        if text and text.strip() and self.heard_popup_mode == "unrecognised":
            self._clear_heard_popup()
        dialog = self.search_dialog
        if text and dialog is not None and dialog.isVisible() and dialog.dictation_enabled:
            # Don't write the stop phrase into the text field when it is spoken.
            content, ending = self._strip_end_dictation_phrase(text)
            if content.lower() == "end" or content.lower().endswith(" end"):
                content = content[:-4].strip() if content.lower().endswith(" end") else ""
            # A terminal stop phrase must not erase a previously entered search query.
            if content or not (ending or (text or "").strip().lower() in ("end", "end dictation")):
                dialog.set_dictation_text(self._format_dictation_numbers(content))
        elif self.focus_dictation_enabled:
            self._stream_focused_dictation_partial(text)

    def on_voice_status(self, text):
        # Status callbacks from STT/TTS are routed through Qt signals.
        if text:
            if text.startswith(("Voice input error:", "Microphone unavailable:", "Vosk model not found")):
                self.stt_enabled = False
            self.status = text

    def set_hand_tracking_paused(self, paused: bool, announce: bool = True):
        """Pause hand inference and cancel active hand actions without stopping Vosk."""
        paused = bool(paused)
        if paused == self.hand_tracking_paused:
            self.status = "Hand tracking is already paused" if paused else "Hand tracking is already resumed"
            self.update()
            return

        self.hand_tracking_paused = paused
        self.tracker.set_hand_tracking_enabled(not paused)
        if paused:
            if self.voice_selection_mode is not None:
                self._clear_voice_selection(release_mouse=True)
            # End any active click/drag/scroll and release native mouse buttons so pausing
            # in the middle of a gesture cannot leave an action or held button stuck.
            for side in ("left", "right"):
                self._end_pinch_drag(side, status=False)
                self._release_fist_mouse(side)
                self._reset_fist_gesture(side, release_mouse=False)
                self.active_action[side] = None
                self.action_stop_wait_open[side] = False
                self.action_stop_open_since[side] = None
                self.right_click_context_active[side] = False
                self._clear_menu_pinch_sequence(side)
                self.fist_release_required_for_pinch[side] = False
                self.fist_pose_confirmation_frames[side] = 0
                self.side_cursor_positions[side] = None
                self.smoothed_side_cursor_positions[side] = None
                self.hand_positions[side] = []
            self.gesture_owner = None
            self.pointer_side = None
            self.canvas_active_side = None
            self.active_slider = None
            self.active_slider_side = None
            self.tli_overrides = {}
            self.active_push_sides = set()
            self.status = "Hand tracking paused — say 'resume tracking' to continue"
            if announce:
                self.tts.speak("Hand tracking paused. Say resume tracking when you are ready.")
        else:
            # Drop stale cursor smoothing so resuming starts at the current hand position.
            self.cursor_updated_at = {"left": 0.0, "right": 0.0}
            self.side_cursor_positions = {"left": None, "right": None}
            self.smoothed_side_cursor_positions = {"left": None, "right": None}
            self.hand_positions = {"left": [], "right": []}
            self.status = "Hand tracking resumed"
            if announce:
                self.tts.speak("Hand tracking resumed.")
        self.update()

    def _current_tli_global_position(self):
        """Return the active TLI's global screen position, falling back to the OS cursor."""
        side = self.pointer_side
        if side not in ("left", "right") or self.side_cursor_positions.get(side) is None:
            side = "right" if self.side_cursor_positions.get("right") is not None else "left"
        point = self.side_cursor_positions.get(side)
        if point is not None:
            return self._virtual_point_to_global(point)
        if pyautogui is not None:
            try:
                x, y = pyautogui.position()
                return QPoint(int(x), int(y))
            except Exception:
                pass
        return QCursor.pos()

    def _focus_external_target(self, close_search=True):
        """Restore the last external app (not Spatial Desktop) for OS-level commands."""
        dialog = self.search_dialog
        dialog_was_visible = bool(dialog is not None and dialog.isVisible())
        target = None
        if sys.platform.startswith("win"):
            try:
                user32 = ctypes.windll.user32
                foreground = int(user32.GetForegroundWindow() or 0)
                candidates = []
                if dialog_was_visible and self.search_return_hwnd:
                    candidates.append(int(self.search_return_hwnd))
                if foreground:
                    pid = wintypes.DWORD(0)
                    user32.GetWindowThreadProcessId(wintypes.HWND(foreground), ctypes.byref(pid))
                    if pid.value != os.getpid():
                        candidates.insert(0, foreground)
                if self.last_external_foreground_hwnd:
                    candidates.append(int(self.last_external_foreground_hwnd))
                for hwnd in candidates:
                    if not hwnd or not user32.IsWindow(wintypes.HWND(hwnd)):
                        continue
                    pid = wintypes.DWORD(0)
                    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
                    if pid.value != os.getpid():
                        target = hwnd
                        break
            except Exception:
                target = None
        elif sys.platform.startswith("linux"):
            candidates = []
            if dialog_was_visible and self.search_return_window_id:
                candidates.append(str(self.search_return_window_id))
            if self.last_external_window_id:
                candidates.append(str(self.last_external_window_id))
            for candidate in candidates:
                try:
                    pid_text = subprocess.check_output(["xdotool", "getwindowpid", candidate], text=True,
                                                       stderr=subprocess.DEVNULL, timeout=0.35).strip()
                    if int(pid_text) != os.getpid():
                        target = candidate
                        break
                except Exception:
                    continue

        if dialog_was_visible and close_search:
            dialog.close()
        if sys.platform.startswith("win") and target:
            try:
                user32 = ctypes.windll.user32
                user32.ShowWindow(wintypes.HWND(target), 9)  # SW_RESTORE
                user32.SetForegroundWindow(wintypes.HWND(target))
                self.last_external_foreground_hwnd = int(target)
                time.sleep(0.045)
                return True
            except Exception:
                return False
        if sys.platform.startswith("linux") and target:
            try:
                subprocess.run(["xdotool", "windowactivate", "--sync", str(target)], check=True,
                               timeout=0.75, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.last_external_window_id = str(target)
                return True
            except Exception:
                return False
        # Never send keyboard shortcuts to our own overlay when an external target
        # could not be identified. This is especially important for close/minimise/delete.
        if sys.platform == "darwin":
            return not dialog_was_visible or close_search
        return False

    def _voice_hotkey(self, *keys):
        if pyautogui is None:
            self.status = "Keyboard automation is unavailable; install pyautogui."
            return False
        if not self._focus_external_target(close_search=True):
            self.status = "No external application could be focused safely. Switch to OS Desktop mode and try again."
            return False
        try:
            pyautogui.hotkey(*keys, interval=0.02, _pause=False)
            return True
        except Exception as exc:
            self.status = f"Voice keyboard command failed: {exc}"
            return False

    def _voice_mouse_shortcut(self, key, *mods):
        if pyautogui is None:
            self.status = "Mouse automation is unavailable; install pyautogui."
            return False
        if not self._focus_external_target(close_search=True):
            self.status = "No external application could be focused safely."
            return False
        try:
            if mods:
                pyautogui.hotkey(*mods, key, interval=0.02, _pause=False)
            else:
                pyautogui.press(key, _pause=False)
            return True
        except Exception as exc:
            self.status = f"Voice keyboard command failed: {exc}"
            return False

    def _begin_voice_text_selection(self):
        if self.hand_tracking_paused:
            self.status = "Resume hand tracking before starting a TLI text selection."
            self.tts.speak(self.status)
            return False
        if self.desktop_mode != "os":
            self.status = "Switch to OS Desktop mode before starting a voice text selection."
            self.tts.speak(self.status)
            return False
        point = self._current_tli_global_position()
        if not self._focus_external_target(close_search=True):
            self.status = "I couldn't focus the target window for text selection."
            self.tts.speak(self.status)
            return False
        if pyautogui is None:
            return False
        try:
            pyautogui.moveTo(point.x(), point.y(), duration=0, _pause=False)
            pyautogui.mouseDown(button="left", _pause=False)
            self.voice_selection_mode = "text"
            self.voice_lasso_start = QPointF(point)
            self.gesture_owner = None
            for side in ("left", "right"):
                self._reset_fist_gesture(side)
                self._clear_menu_pinch_sequence(side)
            self.status = "Text selection active — move your hand, then say 'copy this' or 'finish selection'."
            self.tts.speak("Text selection started. Move your hand to select, then say copy this.")
            return True
        except Exception as exc:
            try:
                pyautogui.mouseUp(button="left", _pause=False)
            except Exception:
                pass
            self.voice_selection_mode = None
            self.voice_lasso_start = None
            self.status = f"Could not start text selection: {exc}"
            return False

    def _begin_voice_image_lasso(self):
        if self.hand_tracking_paused:
            self.status = "Resume hand tracking before starting an image lasso."
            self.tts.speak(self.status)
            return False
        if self.desktop_mode != "os":
            self.status = "Switch to OS Desktop mode before starting an image lasso."
            self.tts.speak(self.status)
            return False
        start = self._current_tli_global_position()
        self._focus_external_target(close_search=True)
        self.voice_selection_mode = "image"
        self.voice_lasso_start = QPointF(start)
        self.status = "Image lasso active — move to the opposite corner and say 'copy this'."
        self.tts.speak("Image selection started. Move to the opposite corner and say copy this.")
        self.update()
        return True

    def _clear_voice_selection(self, release_mouse=True):
        if release_mouse and self.voice_selection_mode == "text" and pyautogui is not None:
            try:
                pyautogui.mouseUp(button="left", _pause=False)
            except Exception:
                pass
        self.voice_selection_mode = None
        self.voice_lasso_start = None
        self.voice_selection_target = None
        self.update()

    def _copy_image_lasso_to_clipboard(self):
        if self.voice_selection_mode != "image" or self.voice_lasso_start is None:
            return False
        end = self._current_tli_global_position()
        x1, y1 = int(self.voice_lasso_start.x()), int(self.voice_lasso_start.y())
        x2, y2 = int(end.x()), int(end.y())
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2-x1), abs(y2-y1)
        if width < 12 or height < 12:
            self.status = "Image lasso is too small. Move farther to the opposite corner and try again."
            return False
        if pyautogui is None:
            self.status = "Screen capture is unavailable; install pyautogui/Pillow."
            return False
        original_opacity = self.windowOpacity()
        try:
            # Hide the full-screen TLI/skeleton briefly so it is not copied into the image.
            self.setWindowOpacity(0.0)
            QApplication.processEvents()
            time.sleep(0.06)
            captured = pyautogui.screenshot(region=(left, top, width, height)).convert("RGBA")
            raw = captured.tobytes("raw", "RGBA")
            image = QImage(raw, captured.width, captured.height, captured.width*4,
                           QImage.Format_RGBA8888).copy()
            QApplication.clipboard().setImage(image)
        except Exception as exc:
            self.status = f"Could not capture image region: {exc}"
            return False
        finally:
            self.setWindowOpacity(original_opacity)
            QApplication.processEvents()
        self._clear_voice_selection(release_mouse=False)
        self.status = f"Copied image region ({width} × {height}) to the clipboard. Say 'paste here' to insert it."
        self.tts.speak("Image region copied. Say paste here to insert it.")
        return True

    def _copy_current_selection(self):
        if self.voice_selection_mode == "image":
            return self._copy_image_lasso_to_clipboard()
        if self.voice_selection_mode == "text":
            self._clear_voice_selection(release_mouse=True)
        if self._voice_hotkey("ctrl", "c"):
            self.status = "Copy sent to the selected text or item. Say 'paste here' at the destination."
            self.tts.speak("Copied. Move to the destination and say paste here.")
            return True
        return False

    def _paste_at_tli(self):
        point = self._current_tli_global_position()
        if self.desktop_mode != "os":
            self.status = "Switch to OS Desktop mode to paste at the TLI position."
            self.tts.speak(self.status)
            return False
        if self.voice_selection_mode is not None:
            self._clear_voice_selection(release_mouse=True)
        if not self._focus_external_target(close_search=True):
            self.status = "I couldn't focus the destination window. Click or hover its text field and try again."
            self.tts.speak(self.status)
            return False
        if pyautogui is None:
            return False
        try:
            pyautogui.click(point.x(), point.y(), clicks=1, interval=0, _pause=False)
            time.sleep(0.045)
            pyautogui.hotkey("ctrl", "v", interval=0.02, _pause=False)
            self.status = "Pasted clipboard contents at the TLI position."
            self.tts.speak("Pasted here.")
            return True
        except Exception as exc:
            self.status = f"Paste failed: {exc}"
            return False

    def _open_known_folder(self, folder_name=None):
        base = Path.home()
        target = base if folder_name is None else base / folder_name
        try:
            if sys.platform.startswith("win"):
                # Explorer's known-folder aliases work with redirected OneDrive and
                # localized Windows profile folders, where Path.home()/Documents
                # may not exist even though the Documents library does.
                windows_known_folders = {
                    "Documents": "shell:Personal", "Downloads": "shell:Downloads",
                    "Desktop": "shell:Desktop", "Pictures": "shell:My Pictures",
                    "Videos": "shell:My Video", "Music": "shell:My Music",
                }
                shell_target = windows_known_folders.get(folder_name, str(target)) if folder_name else str(target)
                subprocess.Popen(["explorer.exe", shell_target])
            else:
                if not target.exists():
                    self.status = f"Folder not found: {target}"
                    self.tts.speak(f"I couldn't find the {folder_name or 'user'} folder.")
                    return False
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
            self.status = f"Opened {folder_name or 'user'} folder"
            return True
        except Exception as exc:
            self.status = f"Could not open folder: {exc}"
            return False

    def _media_key(self, action):
        """Send common media/volume keys without requiring a resident media service."""
        if sys.platform.startswith("win"):
            vk = {
                "media_play_pause": 0xB3, "media_next": 0xB0, "media_previous": 0xB1,
                "volume_mute": 0xAD, "volume_up": 0xAF, "volume_down": 0xAE,
                "volume_unmute": 0xAF,
            }.get(action)
            if vk is None:
                return False
            try:
                user32 = ctypes.windll.user32
                user32.keybd_event(vk, 0, 0, 0)
                user32.keybd_event(vk, 0, 2, 0)
                return True
            except Exception:
                return False
        if sys.platform.startswith("linux"):
            if action.startswith("media_") and shutil.which("playerctl"):
                command = {"media_play_pause": "play-pause", "media_next": "next", "media_previous": "previous"}[action]
                return subprocess.run(["playerctl", command], timeout=1.0,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if action.startswith("volume_") and shutil.which("pactl"):
                args = {"volume_mute": ["set-sink-mute", "@DEFAULT_SINK@", "1"],
                        "volume_unmute": ["set-sink-mute", "@DEFAULT_SINK@", "0"],
                        "volume_up": ["set-sink-volume", "@DEFAULT_SINK@", "+5%"],
                        "volume_down": ["set-sink-volume", "@DEFAULT_SINK@", "-5%"]}.get(action)
                if args:
                    return subprocess.run(["pactl", *args], timeout=1.0,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        # Fallback for macOS or systems exposing the media keys through pyautogui.
        key = {"media_play_pause": "playpause", "media_next": "nexttrack", "media_previous": "prevtrack",
               "volume_mute": "volumemute", "volume_unmute": "volumeup", "volume_up": "volumeup",
               "volume_down": "volumedown"}.get(action)
        try:
            if pyautogui is not None and key:
                pyautogui.press(key, _pause=False)
                return True
        except Exception:
            pass
        return False

    def execute_pc_voice_command(self, action):
        """Run a matched deterministic desktop command; no shell text is interpreted."""
        self._clear_heard_popup()
        if action == "copy":
            self._copy_current_selection(); return True
        if action == "cut":
            if self.voice_selection_mode is not None:
                self._clear_voice_selection(release_mouse=True)
            if self._voice_hotkey("ctrl", "x"):
                self.status = "Cut requested for the current selection."
                self.tts.speak("Cut selection.")
            return True
        if action == "paste_here":
            self._paste_at_tli(); return True
        if action == "paste":
            if self.voice_selection_mode is not None:
                self._clear_voice_selection(release_mouse=True)
            if self._voice_hotkey("ctrl", "v"):
                self.status = "Clipboard pasted into the focused application."
            return True
        if action == "copy_image_lasso":
            if self.voice_selection_mode == "image":
                self._copy_image_lasso_to_clipboard()
            else:
                self._begin_voice_image_lasso()
            return True
        if action == "start_text_selection":
            self._begin_voice_text_selection(); return True
        if action == "finish_selection":
            if self.voice_selection_mode == "text":
                self._clear_voice_selection(release_mouse=True)
                self.status = "Text selection finished. Say 'copy this' to copy it."
            elif self.voice_selection_mode == "image":
                self._copy_image_lasso_to_clipboard()
            else:
                self.status = "There is no active lasso selection."
            return True
        if action == "cancel_selection":
            self._clear_voice_selection(release_mouse=True)
            self.status = "Voice selection cancelled."
            return True
        if action in ("select_all", "undo", "redo"):
            if self.voice_selection_mode is not None:
                self._clear_voice_selection(release_mouse=True)
            key = {"select_all": "a", "undo": "z", "redo": "y"}[action]
            if self._voice_hotkey("ctrl", key):
                self.status = {"select_all": "Selected all.", "undo": "Undo.", "redo": "Redo."}[action]
            return True
        if action.startswith("media_") or action.startswith("volume_"):
            if self._media_key(action):
                labels = {"media_play_pause": "Playback toggled.", "media_next": "Next track.",
                          "media_previous": "Previous track.", "volume_up": "Volume up.",
                          "volume_down": "Volume down.", "volume_mute": "Audio muted.", "volume_unmute": "Volume unmuted."}
                self.status = labels.get(action, "Media command sent.")
            else:
                self.status = "Media control is unavailable on this system."
            return True
        folders = {"open_downloads": "Downloads", "open_documents": "Documents", "open_desktop_folder": "Desktop",
                   "open_pictures": "Pictures", "open_videos": "Videos", "open_music_folder": "Music"}
        if action in folders:
            self._open_known_folder(folders[action]); return True
        if action == "open_file_explorer":
            self.open_user_folder(); return True
        if action in ("open_notepad", "open_calculator"):
            try:
                if sys.platform.startswith("win"):
                    subprocess.Popen(["notepad.exe"] if action == "open_notepad" else ["calc.exe"])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", "TextEdit" if action == "open_notepad" else "Calculator"])
                else:
                    app = "mousepad" if action == "open_notepad" else "gnome-calculator"
                    subprocess.Popen([app])
                self.status = "Application launched."
            except Exception as exc:
                self.status = f"Could not launch application: {exc}"
            return True
        if action == "fullscreen_toggle":
            # Route to the external application, never the Spatial Desktop overlay.
            if self._focus_external_target(close_search=True):
                if self._voice_hotkey("f11"):
                    self.status = "Fullscreen toggled for the active application."
                else:
                    self.status = "Could not send the fullscreen shortcut to the active application."
            else:
                self.status = "No external application is available for fullscreen."
            return True
        if action in ("minimize_window", "maximize_window", "restore_window"):
            if self._focus_external_target(close_search=True) and sys.platform.startswith("win"):
                try:
                    user32 = ctypes.windll.user32
                    hwnd = int(user32.GetForegroundWindow() or 0)
                    cmd = {"minimize_window": 6, "maximize_window": 3, "restore_window": 9}[action]
                    user32.ShowWindow(wintypes.HWND(hwnd), cmd)
                    self.status = action.replace("_", " ").title() + "."
                except Exception as exc:
                    self.status = f"Window command failed: {exc}"
            elif action == "minimize_window":
                self._voice_hotkey("win", "down")
            elif action == "maximize_window":
                self._voice_hotkey("win", "up")
            else:
                self.status = "Restore is supported for Windows windows; focus the target and try again."
            return True
        shortcuts = {
            "switch_window": ("alt", "tab"), "show_desktop": ("win", "d"),
            "snap_left": ("win", "left"), "snap_right": ("win", "right"),
            "new_folder": ("ctrl", "shift", "n"), "new_tab": ("ctrl", "t"), "close_tab": ("ctrl", "w"),
            "go_back": ("alt", "left"), "go_forward": ("alt", "right"), "go_up": ("alt", "up"),
            "refresh_window": ("ctrl", "r"), "rename_selected": ("f2",), "delete_selected": ("delete",),
        }
        if action == "open_start_menu":
            if sys.platform.startswith("win"):
                success = self._voice_hotkey("win")
            elif sys.platform == "darwin":
                success = self._voice_hotkey("ctrl", "f2")
            else:
                success = self._voice_hotkey("win")
            self.status = "Opening the system application menu." if success else "Could not open the system application menu."
            return True
        if action == "show_running_apps":
            # Use the operating system's native task/window overview, which works
            # consistently with the active desktop environment and exposes real apps.
            if sys.platform.startswith("win"):
                success = self._voice_hotkey("win", "tab")
            elif sys.platform == "darwin":
                success = self._voice_hotkey("ctrl", "up")
            else:
                # GNOME/KDE commonly bind Super+Tab to application/window switching.
                success = self._voice_hotkey("win", "tab")
                if not success:
                    success = self._voice_hotkey("alt", "tab")
            self.status = ("Showing running applications." if success else
                           "Could not open the system app switcher. Try 'switch window'.")
            return True
        if action in shortcuts:
            keys = shortcuts[action]
            success = self._voice_hotkey(*keys)
            if success:
                if action == "delete_selected":
                    self.status = "Delete requested. In Explorer this moves the selected item to the Recycle Bin; review the selection first."
                else:
                    self.status = action.replace("_", " ").title() + "."
            return True
        return False

    def _clear_heard_popup(self):
        """Reset transient feedback while keeping the live STT pill visible."""
        self.heard_popup_generation += 1
        self.heard_popup_mode = "transcript" if self.stt_enabled else "hidden"
        self.heard_popup_message = ""
        self.heard_popup_shake_until = 0.0
        self.update()

    def _show_unrecognised_command(self, phrase):
        """Show red/shaking feedback, label it after 2s, then clear after 5s."""
        self.heard_popup_generation += 1
        generation = self.heard_popup_generation
        self.heard_popup_mode = "unrecognised"
        self.heard_popup_message = f"Heard: {phrase}"
        self.heard_popup_shake_until = time.monotonic() + 0.7
        self.update()

        def show_label():
            if generation != self.heard_popup_generation or self.heard_popup_mode != "unrecognised":
                return
            self.heard_popup_message = "Unrecognised Command"
            self.heard_popup_shake_until = 0.0
            self.update()
            # Keep the feedback brief, then return to a blank live-listening bar.
            # A generation check prevents this timer clearing a newer phrase.
            QTimer.singleShot(2000, clear_label)

        def clear_label():
            if generation != self.heard_popup_generation or self.heard_popup_mode != "unrecognised":
                return
            self._clear_heard_popup()

        QTimer.singleShot(2000, show_label)

    def _on_general_voice_answer(self, question, answer):
        if not answer:
            answer = "I couldn't get an answer from the local D1 model. Check that the model is loaded and try again."
        self.status = answer
        self.tts.speak(answer)

    def _ask_d1_general_question(self, question):
        # Keep model inference off the Qt UI thread; the signal returns the result safely.
        def worker():
            try:
                from d1_decision import answer_general_question
                answer = answer_general_question(question)
            except Exception:
                answer = None
            self.voice_reply_signals.finished.emit(question, answer or "")
        threading.Thread(target=worker, name="D1-general-answer", daemon=True).start()

    def on_speech(self, text):
        # A completed phrase has been consumed; don't keep it as the idle pill's
        # fallback transcript. The next phrase is shown from speech_partial.
        self.last_speech = ""
        self.speech_partial = ""
        self.heard_popup_generation += 1
        self.heard_popup_mode = "transcript"
        self.heard_popup_message = ""
        self.heard_popup_shake_until = 0.0
        low = text.lower().strip()
        low = re.sub(r"\s+", " ", low)
        dialog = self.search_dialog

        # Handle global stop-listening commands before any dictation path can type
        # the phrase into a field. Keep confirmation identical to the Glass Menu.
        stop_listening_phrases = {
            "stop listening", "voice off", "stop voice control", "disable voice input",
            "turn off voice input", "stop voice input", "turn voice input off",
            "turn the voice input off", "disable listening", "stop listening now",
            "turn off listening", "stop voice recognition", "disable voice recognition",
        }
        if low in stop_listening_phrases:
            if dialog is not None and dialog.isVisible() and dialog.dictation_enabled:
                dialog.set_dictation_enabled(False, notify=True)
            if self.focus_dictation_enabled:
                self.end_focused_dictation("")
            self.focus_dictation_paused = False
            self.focus_dictation_committed_words = []
            self.focus_dictation_last_partial = ""
            self.speech.stop()
            self.stt_enabled = False
            self.status = "Voice input off"
            self._announce_voice_input_state(False)
            return

        # Live local facts must be answered from the OS, never guessed by a model.
        if not self.focus_dictation_enabled and not (dialog is not None and dialog.isVisible()):
            live = live_answer(text)
            if live:
                kind, answer = live
                self.status = answer
                self.tts.speak(answer)
                return

            # Spoken file queries reuse the existing indexed local-search UI.
            query = file_query(text)
            if query:
                self.open_search_bar(from_voice=False)
                if self.search_dialog is not None:
                    self.search_dialog.set_dictation_enabled(False)
                    self.search_dialog.query.setText(query)
                    self.search_dialog.query.setFocus()
                self.status = "Searching local files for: " + query
                self.tts.speak("Searching your local files for " + query + ".")
                return
        if low in ("fullscreen", "full screen", "toggle fullscreen", "toggle full screen", "make fullscreen", "make full screen"):
            if sys.platform.startswith("win"):
                try:
                    hwnd = int(ctypes.windll.user32.GetForegroundWindow() or 0)
                    overlay_hwnd = int(self.winId())
                    if hwnd and hwnd != overlay_hwnd:
                        # Windows' standard maximize/fullscreen toggle is F11 for common apps.
                        self._voice_hotkey("f11")
                        self.status = "Fullscreen toggle sent to active application"
                    else:
                        self.status = "Focus the application you want to toggle fullscreen"
                except Exception as exc:
                    self.status = f"Fullscreen toggle failed: {exc}"
            else:
                self._voice_hotkey("f11")
                self.status = "Fullscreen toggle sent to active application"
            return
        if low in ("begin dictation", "start dictation", "dictation on", "start typing"):
            self.begin_focused_dictation()
            return
        if low in ("pause dictation", "dictation pause", "hold dictation"):
            self.focus_dictation_paused = True
            self.status = "Dictation paused — say 'resume dictation' to continue"
            return
        if low in ("resume dictation", "continue dictation", "dictation resume"):
            self.focus_dictation_paused = False
            self.focus_dictation_enabled = True
            self.status = "Dictation resumed"
            return
        if low in ("stop dictation", "dictation off", "stop typing"):
            if dialog is not None and dialog.isVisible():
                dialog.set_dictation_enabled(False, notify=True)
            self.end_focused_dictation("")
            self.status = "Dictation stopped"
            return
        dictation_content, ending_dictation = self._strip_end_dictation_phrase(text)
        if ending_dictation:
            if dialog is not None and dialog.isVisible() and dialog.dictation_enabled:
                if dictation_content:
                    dialog.set_dictation_text(dictation_content, final=True)
                dialog.set_dictation_enabled(False, notify=True)
                self.status = "Search dictation ended"
            elif self.focus_dictation_enabled:
                self.end_focused_dictation(text)
            else:
                self.status = "Dictation is not active"
            return
        # Tracking commands run before search dictation so they always control the
        # hand-input layer, even when the launcher is open and dictation is enabled.
        if low in ("pause tracking", "pause hand tracking", "disable hand tracking", "stop hand tracking"):
            self._clear_heard_popup()
            self.set_hand_tracking_paused(True)
            return
        if low in ("resume tracking", "resume hand tracking", "enable hand tracking", "start hand tracking"):
            self._clear_heard_popup()
            self.set_hand_tracking_paused(False)
            return
        if low in ("show menu", "toggle menu", "toggle glass menu", "open glass menu", "hide menu", "close glass menu"):
            self._clear_heard_popup()
            self._toggle_glass_menu_from_voice()
            return
        if low in ("show context", "toggle context", "show context menu", "toggle pinch menu", "open context menu", "hide context", "close context menu"):
            self._clear_heard_popup()
            self._toggle_context_menu_from_voice()
            return
        # These global commands must run before dictation can place the phrase in
        # the search field as ordinary text.
        if low in ("close window", "close active window", "close focused window", "close this window"):
            self._clear_heard_popup()
            self.close_focused_window()
            return
        if low in ("open user folder", "open my user folder", "open home folder", "open home directory", "open my home folder"):
            self._clear_heard_popup()
            self.open_user_folder()
            return
        # These broad natural-language requests deliberately go through d1 first,
        # so its semantic routing is exercised rather than bypassed by exact matching.
        d1_first = low in ("open start menu", "show apps", "show the apps", "show me apps",
                           "show running apps", "open apps", "show all apps",
                           "open documents", "open documents folder", "open downloads",
                           "open downloads folder", "open desktop folder", "open pictures",
                           "open pictures folder", "open videos", "open music folder",
                           "open file explorer", "open explorer", "open files", "open file browser")
        pc_action = None
        if d1_first:
            try:
                pc_action = classify_voice_command(text, aliases_for_help())
            except Exception:
                pc_action = None
        if pc_action:
            self.execute_pc_voice_command(pc_action)
            return
        pc_action = match_pc_command(low)
        if pc_action:
            self.execute_pc_voice_command(pc_action)
            return
        # d1 interprets less exact natural phrasing; its output is restricted to
        # supported action IDs and falls through on uncertainty.
        try:
            pc_action = classify_voice_command(text, aliases_for_help())
        except Exception:
            pc_action = None
        if pc_action:
            self.execute_pc_voice_command(pc_action)
            return
        if self.focus_dictation_enabled:
            if self.focus_dictation_paused:
                return
            # Final Vosk result commits any words not already streamed from partials.
            words = self._normalised_dictation_words(text)
            committed = list(self.focus_dictation_committed_words)
            if words[:len(committed)] == committed:
                remainder = words[len(committed):]
                if remainder:
                    self._type_focus_text(" ".join(remainder) + " ")
            elif words and not committed:
                self._type_focus_text(" ".join(words) + " ")
            self.focus_dictation_committed_words = []
            self.focus_dictation_last_partial = ""
            return
        if dialog is not None and dialog.isVisible():
            if low in ("close search", "close search bar", "cancel search", "exit search"):
                dialog.close()
                self.status = "Search closed"
                return
            if low in ("stop dictation", "dictation off", "stop typing"):
                dialog.set_dictation_enabled(False, notify=True)
                return
            if low in ("stop listening", "voice off", "stop voice control"):
                dialog.set_dictation_enabled(False, notify=True)
                self.speech.stop()
                self.stt_enabled = False
                self.status = "Voice input off"
                return
            if low in ("open top result", "open first result", "open result"):
                dialog.open_selected()
                return
            if dialog.dictation_enabled:
                dialog.set_dictation_text(text, final=True)
                self.status = f"Searching for: {text}"
                return
        if low in ("search bar", "open search", "open search bar", "search files", "find files", "file search"):
            self.open_search_bar(from_voice=True)
            return
        if "next desktop" in low or "next workspace" in low:
            self.target_depth = min(2, self.target_depth+1)
            self.status = "Voice command: next desktop"
            self.tts.speak("Moved to the next desktop.")
        elif "previous desktop" in low or "previous workspace" in low:
            self.target_depth = max(0, self.target_depth-1)
            self.status = "Voice command: previous desktop"
            self.tts.speak("Moved to the previous desktop.")
        elif low in ("home", "go home", "desktop home"):
            self.target_depth = 0
            self.status = "Voice command: home desktop"
            self.tts.speak("Returned to the home desktop.")
        elif low in ("stop listening", "voice off", "stop voice control"):
            self.speech.stop()
            self.stt_enabled = False
            self.status = "Voice input off"
            self.tts.speak("Voice input is off.")
        elif low in ("what can you do", "help", "voice help"):
            self.status = "Voice commands: pause/resume tracking, search bar, copy this/copy text, paste here/paste this, show running apps, start text selection, copy image, media playback/volume, window management, file navigation, close window, open folders, stop listening"
            self.tts.speak("You can say pause tracking, resume tracking, begin dictation, end dictation, show menu, show context, search bar, start text selection, copy this, copy text, copy image, paste here, paste this, show running apps, switch window, play pause, next track, previous track, volume up, volume down, mute, unmute, minimize window, maximize window, show desktop, open downloads, new folder, select all, undo, redo, close window, or open user folder.")
        else:
            # General conversational questions are answered by the local D1 model.
            # Do not steal ordinary dictation, search terms, or unrecognised commands.
            question_start = re.match(r"^(what|why|how|who|when|where|which|can you|could you|do you|is it|are you|tell me|explain|define)\b", low)
            if question_start and not self.focus_dictation_enabled and not (dialog is not None and dialog.isVisible()):
                self.status = "Thinking…"
                self._ask_d1_general_question(text)
            else:
                self.status = f"Heard: {text}"
                self._show_unrecognised_command(text)

    def apply_desktop_mode(self):
        """Configure a fullscreen overlay; OS mode passes physical mouse input through."""
        self._applied_desktop_mode = self.desktop_mode
        self._applied_menu_state = bool(self.menu_open and self.hub_visible)
        self.setCursor(Qt.ArrowCursor if self.desktop_mode == "os" else Qt.BlankCursor)
        if sys.platform.startswith("win"):
            try:
                hwnd = int(self.winId())
                get_long = ctypes.windll.user32.GetWindowLongW
                set_long = ctypes.windll.user32.SetWindowLongW
                style = get_long(hwnd, -20)
                transparent, toolwindow, layered = 0x00000020, 0x00000080, 0x00080000
                style |= toolwindow
                if self.desktop_mode == "os":
                    style |= transparent | layered
                else:
                    style &= ~transparent
                set_long(hwnd, -20, style)
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0,
                    0x0001 | 0x0002 | 0x0004 | 0x0010)
            except Exception as exc:
                print("Overlay window configuration:", exc)
        elif sys.platform.startswith("linux"):
            # The fullscreen overlay is deliberately topmost. In OS Desktop mode it
            # must also be transparent to native input, otherwise clicks generated by
            # PyAutoGUI hit this overlay instead of Qtile/the application underneath.
            # Our glass UI is activated from hand-landmark hit-testing, not QWidget
            # mouse events, so the hand-operated menu and Settings continue to work.
            transparent_flag = Qt.WindowTransparentForInput
            want_transparent = self.desktop_mode == "os"
            has_transparent = bool(self.windowFlags() & transparent_flag)
            if has_transparent != want_transparent:
                was_visible = self.isVisible()
                was_fullscreen = self.isFullScreen()
                flags = self.windowFlags()
                if want_transparent:
                    flags |= transparent_flag
                else:
                    flags &= ~transparent_flag
                self.setWindowFlags(flags)
                # Changing top-level window flags hides the widget in Qt; restore its
                # prior visible/fullscreen state without changing the selected mode.
                if was_visible:
                    if was_fullscreen:
                        self.showFullScreen()
                    else:
                        self.show()

    def _arm_tli_calibration(self, side):
        """Open a dedicated target and isolate all hand input until calibration ends."""
        if side not in ("left", "right"):
            return
        if self.tli_calibration_active_side is not None:
            self._cancel_tli_calibration(announce=False)

        # Pressing this hand's calibration button also reattaches its native mouse control
        # if a peace-sign gesture detached it. Tracking and TLI remain live either way.
        if self.mouse_detached_by_side.get(side, False):
            self.mouse_detached_by_side[side] = False
            self.peace_hold_started[side] = None
            self.peace_toggle_latched[side] = False
            self.smoothed_side_cursor_positions[side] = None
            self.cursor_updated_at[side] = 0.0
            self._refresh_hand_tracking_notice(time.monotonic(), show_enabled=True)

        # End any existing drag/scroll/click state before showing the calibration target.
        for tracked_side in ("left", "right"):
            self._cancel_side_interactions_for_tracking_pause(tracked_side)
            self.action_stop_wait_open[tracked_side] = True
            self.action_stop_open_since[tracked_side] = None
            self.tli_calibration_pending[tracked_side] = False
            self.tli_calibration_wait_release[tracked_side] = False
            self.tli_calibration_samples[tracked_side] = []
        self.tli_calibration_active_side = side
        self.tli_calibration_pending[side] = True
        self.tli_calibration_wait_release[side] = True
        self.tli_calibration_samples[side] = []
        self.status = (f"{side.title()} TLI calibration — release the selecting pinch, then pinch "
                       "inside the calibration window")

        window = TliCalibrationWindow(side)
        window.cancel_requested.connect(self._cancel_tli_calibration)
        self.tli_calibration_window = window
        window.set_phase("Release the pinch used to open this window")
        window.show()
        window.raise_()
        window.activateWindow()
        self.update()

    def _maybe_start_hand_onboarding(self):
        """Show hand setup once unless it was completed or explicitly skipped before."""
        if not getattr(self, "hand_calibration", {}).get("completed", False):
            self._start_hand_onboarding(first_run=True)

    def _start_hand_onboarding(self, first_run=False):
        """Open the isolated hand scale/proximity wizard from first run or Settings."""
        if getattr(self, "hand_calibration_active", False):
            self._cancel_hand_onboarding(announce=False)
        if self.tli_calibration_active_side is not None:
            self._cancel_tli_calibration(announce=False)
        if self.hand_tracking_paused:
            self.set_hand_tracking_paused(False, announce=False)
        for side in ("left", "right"):
            if self.mouse_detached_by_side.get(side, False):
                self.mouse_detached_by_side[side] = False
                self.peace_hold_started[side] = None
                self.peace_toggle_latched[side] = False
                self.smoothed_side_cursor_positions[side] = None
                self.cursor_updated_at[side] = 0.0
            self._cancel_side_interactions_for_tracking_pause(side)
            self.action_stop_wait_open[side] = True
            self.action_stop_open_since[side] = None
        self._refresh_hand_tracking_notice(time.monotonic(), show_enabled=True)
        self.settings_open = False
        self.hub_visible = False
        self.menu_open = False
        self.hand_calibration_active = True
        self.hand_calibration_stage = 0
        self.hand_calibration_capture_requested = False
        self.hand_calibration_capture_samples = []
        self.hand_calibration_draft = default_calibration()
        self.hand_calibration_draft["origin"] = "first_run" if first_run else "settings"
        window = HandCalibrationWizard()
        window.action_requested.connect(self._on_hand_calibration_action)
        self.hand_calibration_window = window
        window.show()
        window.raise_()
        window.activateWindow()
        self.status = "Hand setup: screen touch, palm-size, and two-hand proximity calibration"
        self.update()

    def _cancel_hand_onboarding(self, announce=True):
        """Close onboarding without replacing a previous valid calibration."""
        self.hand_calibration_active = False
        self.hand_calibration_capture_requested = False
        self.hand_calibration_capture_samples = []
        window = self.hand_calibration_window
        self.hand_calibration_window = None
        if window is not None:
            window.hide()
            window.close()
            window.deleteLater()
        for side in ("left", "right"):
            self.action_stop_wait_open[side] = True
            self.action_stop_open_since[side] = None
            self.smoothed_side_cursor_positions[side] = None
            self.cursor_updated_at[side] = 0.0
        self.pointer_side = None
        if announce:
            self.status = "Hand calibration cancelled; previous calibration retained"
        self.update()

    def _on_hand_calibration_action(self, action):
        if not self.hand_calibration_active:
            return
        window = self.hand_calibration_window
        if action == "start":
            self.hand_calibration_stage = 1
            if window is not None:
                window.set_stage(1)
            self.status = "Calibration step 1: extend your arm and touch below the camera"
            return
        if action == "capture":
            if self.hand_calibration_stage not in (1, 2, 3):
                return
            self.hand_calibration_capture_requested = True
            self.hand_calibration_capture_samples = []
            if window is not None:
                window.set_capture_progress(0, True)
            self.status = f"Sampling hand calibration pose {self.hand_calibration_stage}"
            return
        if action == "skip":
            # Skip only affects first-run onboarding; a previous valid calibration is kept.
            current = self.hand_calibration if self.hand_calibration.get("calibrated", False) else default_calibration()
            current = dict(current)
            current["completed"] = True
            current["skipped"] = not bool(current.get("calibrated", False))
            if save_hand_calibration(HAND_CALIBRATION_FILE, current):
                self.hand_calibration = current
            self._cancel_hand_onboarding(announce=False)
            self.status = "Hand setup skipped — rerun it any time in Settings → Hands"
            return
        if action == "cancel":
            self._cancel_hand_onboarding()
            return
        if action == "finish":
            self._cancel_hand_onboarding(announce=False)
            self.status = "Hand setup complete — calibrated screen anchor and hand proximity saved"
            self.update()

    def _finalize_hand_calibration_capture(self):
        """Aggregate one stable capture and advance the wizard to the next pose."""
        stage = self.hand_calibration_stage
        samples = self.hand_calibration_capture_samples
        draft = self.hand_calibration_draft
        if stage == 1:
            ax = mean_finite([row.get("anchor_x") for row in samples])
            ay = mean_finite([row.get("anchor_y") for row in samples])
            if ax is None or ay is None:
                self.hand_calibration_capture_requested = False
                if self.hand_calibration_window is not None:
                    self.hand_calibration_window.set_capture_progress(0, False)
                self.status = "Screen touch capture was unstable — try again"
                return
            draft["screen_anchor_camera"] = {"x": ax, "y": ay}
            draft["screen_offset"] = screen_touch_offset(ax, ay, target_x=0.5, target_y=0.08)
            for side in ("left", "right"):
                values = [row.get("palm_sizes", {}).get(side) for row in samples
                          if row.get("palm_sizes", {}).get(side) is not None]
                draft["extended_palm_size"][side] = mean_finite(values)
            next_stage = 2
        elif stage == 2:
            for side in ("left", "right"):
                values = [row.get("palm_sizes", {}).get(side) for row in samples
                          if row.get("palm_sizes", {}).get(side) is not None]
                measured = mean_finite(values)
                if measured is not None:
                    draft["reference_palm_size"][side] = measured
                elif draft["extended_palm_size"].get(side) is not None:
                    draft["reference_palm_size"][side] = draft["extended_palm_size"][side]
            next_stage = 3
        elif stage == 3:
            measured_ratio = median_finite([row.get("contact_ratio") for row in samples])
            if measured_ratio is not None:
                # A little slack compensates for jitter in fingertip landmarks even when
                # the user physically touches tips together during calibration.
                draft["proximity_touch_ratio"] = min(0.75, max(0.12, measured_ratio+0.12))
            draft["proximity_interaction_ratio"] = min(5.0, max(2.2, draft["proximity_touch_ratio"]+1.8))
            draft["completed"] = True
            draft["calibrated"] = bool(
                draft.get("screen_anchor_camera") and
                any(draft.get("reference_palm_size", {}).get(side) for side in ("left", "right"))
            )
            draft["skipped"] = False
            if not draft["calibrated"]:
                self.hand_calibration_capture_requested = False
                if self.hand_calibration_window is not None:
                    self.hand_calibration_window.set_capture_progress(0, False)
                self.status = "Calibration was incomplete — repeat the palm-size step"
                return
            if not save_hand_calibration(HAND_CALIBRATION_FILE, draft):
                self.status = "Calibration measured, but couldn't be saved; check folder permissions"
                draft["completed"] = False
                draft["calibrated"] = False
                self.hand_calibration_capture_requested = False
                if self.hand_calibration_window is not None:
                    self.hand_calibration_window.set_capture_progress(0, False)
                return
            self.hand_calibration = dict(draft)
            next_stage = 4
            self.status = "Hand calibration saved — screen anchor, palm size, and proximity registered"
        else:
            return
        self.hand_calibration_capture_requested = False
        self.hand_calibration_capture_samples = []
        self.hand_calibration_stage = next_stage
        if self.hand_calibration_window is not None:
            self.hand_calibration_window.set_stage(next_stage)
        self.update()

    def _collect_hand_calibration_frame(self, hands_by_side):
        """Take one valid frame for the currently requested calibration pose."""
        if not self.hand_calibration_active:
            return
        window = self.hand_calibration_window
        overlays, tli_overlays = [], []
        gesture_tracking_just_toggled = False
        for side, (landmarks, _score) in hands_by_side.items():
            if len(landmarks) < 21:
                continue
            base_points = []
            for landmark in landmarks:
                mapped = self.interactive_visual_point(float(landmark.x), float(landmark.y))
                global_point = self._virtual_point_to_global(mapped)
                base_points.append(window.mapFromGlobal(global_point) if window is not None else QPoint())
            color = QColor(self.left_click_color if side == "left" else self.right_click_color)
            overlays.append((base_points, color))
            tx = (float(landmarks[4].x)+float(landmarks[8].x))*0.5
            ty = (float(landmarks[4].y)+float(landmarks[8].y))*0.5
            mapped_tli = self.interactive_visual_point(tx, ty)
            global_tli = self._virtual_point_to_global(mapped_tli)
            local_tli = window.mapFromGlobal(global_tli) if window is not None else QPoint()
            tli_overlays.append((QPointF(local_tli), color))
        if window is not None:
            window.set_live_hand_overlay(overlays, tli_overlays)

        if not self.hand_calibration_capture_requested:
            return
        stage = self.hand_calibration_stage
        sample = None
        if stage == 1 and hands_by_side:
            # Use the highest-confidence detected hand's index fingertip as the screen anchor.
            side, (landmarks, _score) = max(hands_by_side.items(), key=lambda entry: entry[1][1])
            if len(landmarks) >= 21:
                x, y = float(landmarks[8].x), float(landmarks[8].y)
                palm_sizes = {}
                for tracked_side, (tracked_points, _tracked_score) in hands_by_side.items():
                    metrics = palm_measurements(tracked_points)
                    if metrics is not None:
                        palm_sizes[tracked_side] = metrics["size"]
                sample = {"anchor_x": x, "anchor_y": y, "palm_sizes": palm_sizes}
        elif stage == 2 and hands_by_side:
            palm_sizes = {}
            for tracked_side, (tracked_points, _tracked_score) in hands_by_side.items():
                metrics = palm_measurements(tracked_points)
                if metrics is not None:
                    palm_sizes[tracked_side] = metrics["size"]
            if palm_sizes:
                sample = {"palm_sizes": palm_sizes}
        elif stage == 3 and "left" in hands_by_side and "right" in hands_by_side:
            ratio = fingertip_contact_ratio(hands_by_side["left"][0], hands_by_side["right"][0])
            left_metrics = palm_measurements(hands_by_side["left"][0])
            right_metrics = palm_measurements(hands_by_side["right"][0])
            if ratio is not None and left_metrics is not None and right_metrics is not None:
                sample = {"contact_ratio": ratio,
                          "palm_sizes": {"left": left_metrics["size"], "right": right_metrics["size"]}}
        if sample is not None:
            self.hand_calibration_capture_samples.append(sample)
            count = len(self.hand_calibration_capture_samples)
            if window is not None:
                window.set_capture_progress(count, count < self.hand_calibration_capture_count)
            if count >= self.hand_calibration_capture_count:
                self._finalize_hand_calibration_capture()
        elif window is not None:
            # Keep the progress indication honest: only valid tracked frames advance it.
            window.set_capture_progress(len(self.hand_calibration_capture_samples), True)

    def _cancel_tli_calibration(self, announce=True):
        """Close the temporary calibration target without retaining partial samples."""
        side = self.tli_calibration_active_side
        if side in ("left", "right"):
            self.tli_calibration_pending[side] = False
            self.tli_calibration_wait_release[side] = False
            self.tli_calibration_samples[side] = []
        self.tli_calibration_active_side = None
        window = self.tli_calibration_window
        self.tli_calibration_window = None
        if window is not None:
            window.hide()
            window.close()
            window.deleteLater()
        # Require a neutral/open pose before gestures can resume after closing the target.
        for tracked_side in ("left", "right"):
            self.action_stop_wait_open[tracked_side] = True
            self.action_stop_open_since[tracked_side] = None
        if announce:
            self.status = "TLI calibration cancelled"
        self.update()

    def _calibration_pinch_inside_window(self, landmarks):
        """Test the exact scaled/calibrated TLI point against the outlined target area."""
        window = self.tli_calibration_window
        side = self.tli_calibration_active_side
        if window is None or not window.isVisible() or landmarks is None or len(landmarks) < 21:
            return False
        points = [(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in landmarks]
        raw_anchor = self.interactive_hand_anchor(landmarks, side)
        mapped = self._scale_hand_tli_point(raw_anchor, points, side)
        global_point = self._virtual_point_to_global(mapped)
        return window.pinch_area_global_rect().contains(QPointF(float(global_point.x()), float(global_point.y())))

    def _update_tli_calibration_live_overlay(self, side, landmarks, cursor):
        """Draw the currently transformed hand and TLI above the target window itself."""
        window = self.tli_calibration_window
        if window is None or not window.isVisible():
            return
        points = [(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in landmarks]
        origin, user_scale = self._hand_visual_transform(points, side)
        mapped_points = []
        for x, y, _z in points:
            q = self.interactive_visual_point(x, y)
            sx, sy = scale_hand_point((q.x(), q.y()), (origin.x(), origin.y()), user_scale)
            global_point = self._virtual_point_to_global(QPointF(sx, sy))
            local = window.mapFromGlobal(global_point)
            mapped_points.append(QPointF(float(local.x()), float(local.y())))
        color = QColor(self.left_click_color if side == "left" else self.right_click_color)
        local_tli = None
        if cursor is not None:
            global_tli = self._virtual_point_to_global(cursor)
            local = window.mapFromGlobal(global_tli)
            local_tli = QPointF(float(local.x()), float(local.y()))
        window.set_hand_overlay([(mapped_points, color)], [(local_tli, color)] if local_tli is not None else [])

    def _process_tli_calibration_frame(self, side, landmarks, is_pinch):
        """Consume calibration gestures and sample only pinches physically inside the target."""
        active_side = self.tli_calibration_active_side
        if active_side is None:
            return False
        # Calibration is modal: even the other hand is prevented from clicking or moving OS input.
        if side != active_side:
            return True
        if not self.tli_calibration_pending.get(side, False):
            return True
        window = self.tli_calibration_window
        if self.tli_calibration_wait_release.get(side, False):
            if not is_pinch:
                self.tli_calibration_wait_release[side] = False
                self.tli_calibration_samples[side] = []
                self.status = f"{side.title()} TLI calibration ready — pinch inside the outlined area"
                if window is not None:
                    window.set_phase("Pinch thumb + index together inside the outlined area")
            elif window is not None:
                window.set_phase("Release the pinch used to open this window")
            return True
        if not is_pinch:
            # Calibration must be one continuous, deliberate pinch, not pooled pulses.
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase("Pinch and hold inside the outlined area", 0)
            return True
        if not self._calibration_pinch_inside_window(landmarks):
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase("Move the pinch point inside the outlined area", 0)
            return True
        sample = encode_relative_tli(landmarks)
        if sample is None:
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase("Landmarks unstable — hold the pinch steady", 0)
            return True
        samples = self.tli_calibration_samples[side]
        samples.append(sample)
        if window is not None:
            window.set_phase(
                f"Hold steady — sampling {len(samples)}/{self.tli_calibration_sample_count}",
                len(samples),
            )
        if len(samples) >= self.tli_calibration_sample_count:
            calibration = mean_relative_tli(samples)
            if calibration is not None:
                self.tli_calibrations[side] = calibration
                persisted = save_tli_calibrations(self.tli_calibrations)
                self.tli_calibration_persisted[side] = persisted
                self.tli_calibration_pending[side] = False
                self.tli_calibration_wait_release[side] = False
                self.tli_calibration_samples[side] = []
                self.tli_calibration_active_side = None
                self.status = (f"{side.title()} TLI calibrated and saved — palm-relative point registered"
                               if persisted else f"{side.title()} TLI calibrated for this session, but saving failed")
                self._cancel_tli_calibration(announce=False)
                # _cancel_tli_calibration only closes the temporary UI; restore success status.
                self.status = (f"{side.title()} TLI calibrated and saved — palm-relative point registered"
                               if persisted else f"{side.title()} TLI calibrated for this session, but saving failed")
            else:
                self.tli_calibration_samples[side] = []
                if window is not None:
                    window.set_phase("Unstable sample — pinch and hold again")
        else:
            self.status = (f"{side.title()} TLI calibration sampling "
                           f"{len(samples)}/{self.tli_calibration_sample_count} — keep pinching inside")
            if window is not None:
                window.set_phase(f"Sampling {len(samples)}/{self.tli_calibration_sample_count} — keep pinching",
                                 len(samples))
        return True

    def raw_hand_anchor(self, landmarks, side=None):
        """Return the calibrated palm-relative TLI or the live fingertip midpoint fallback."""
        calibration = self.tli_calibrations.get(side) if side in ("left", "right") else None
        calibrated = resolve_relative_tli(landmarks, calibration)
        if calibrated is not None:
            return float(calibrated[0]), float(calibrated[1])

        def xy(point):
            if hasattr(point, "x") and hasattr(point, "y"):
                return float(point.x), float(point.y)
            return float(point[0]), float(point[1])

        tx, ty = xy(landmarks[4])
        ix, iy = xy(landmarks[8])
        return (tx+ix)*0.5, (ty+iy)*0.5

    def interactive_point(self, x, y):
        """Map the calibrated camera area to the physical screen and clamp it there."""
        point = self.interactive_visual_point(x, y)
        screen = QApplication.primaryScreen()
        geometry = screen.geometry() if screen is not None else None
        width = geometry.width() if geometry is not None else self.width()
        height = geometry.height() if geometry is not None else self.height()
        return QPointF(min(max(0.0, point.x()), max(1, width-1)),
                       min(max(0.0, point.y()), max(1, height-1)))

    def interactive_visual_point(self, x, y):
        """Map camera landmarks across the full physical screen, including Qtile bars.

        Qtile's usable work area excludes reserved top/bottom bars. Using this widget's
        dimensions can therefore trap the OS pointer inside the work area on X11/XLibre.
        The hand-to-screen transform must use the actual screen geometry instead.
        Coordinates remain screen-local because the main overlay is fullscreen.
        """
        nx = (float(x)-0.20)/(0.80-0.20)
        ny = (float(y)-0.16)/(0.84-0.16)
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry() if screen is not None else None
        screen_width = screen_geometry.width() if screen_geometry is not None else self.width()
        screen_height = screen_geometry.height() if screen_geometry is not None else self.height()
        calibration = getattr(self, "hand_calibration", {})
        offset = (calibration.get("screen_offset", {})
                  if calibration.get("calibrated", False) and not getattr(self, "hand_calibration_active", False)
                  else {})
        try:
            nx += min(0.35, max(-0.35, float(offset.get("x", 0.0))))
            ny += min(0.65, max(-0.65, float(offset.get("y", 0.0))))
        except (TypeError, ValueError, AttributeError):
            pass
        return QPointF(nx*max(1, screen_width-1), ny*max(1, screen_height-1))

    def interactive_hand_anchor(self, landmarks, side=None):
        """Map the calibrated palm-relative TLI before the common hand-scale transform."""
        return self.interactive_visual_point(*self.raw_hand_anchor(landmarks, side))

    # ------------------------------------------------------------------
    # Custom gestures (Settings -> Gestures)
    # ------------------------------------------------------------------
    CUSTOM_GESTURE_COUNTDOWN = 2.0
    CUSTOM_GESTURE_SAMPLES = 15
    CUSTOM_GESTURE_TIMEOUT = 7.0

    def _custom_gesture_layout(self, cx, cy, scale):
        """Single source of truth for Gestures-page geometry (drawing and hit-testing)."""
        # Positioned in the shared left Settings column, starting below the tab strip
        # (the page layout reserves 394 logical units for this block).
        top = self._settings_layout("gestures")["top"]
        L, W = self.SETTINGS_LEFT_X, self.SETTINGS_COLUMN_W

        def r(x, y, w, h):
            return QRectF(cx+x*scale, cy+(top+y)*scale, w*scale, h*scale)
        rows = []
        for index in range(CUSTOM_GESTURE_MAX):
            y = 156 + index*40
            rows.append({
                "row": r(L, y, W, 38),
                "side": r(L+2, y+3, 40, 32),
                "timer": r(L+44, y+3, 42, 32),
                "label": r(L+88, y+3, 56, 32),
                "action": r(L+146, y+3, 128, 32),
                "delete": r(L+277, y+3, 20, 32),
            })
        half = (W-8)*0.5
        return {
            # Keep the instruction text clearly below the tab row, with distinct
            # vertical gaps before recording controls and the gesture list.
            "intro": r(L, 0, W, 34),
            "record_left": r(L, 40, half, 32),
            "record_right": r(L+half+8, 40, half, 32),
            "overlap": r(L, 80, W, 30),
            "message": r(L, 116, W, 34),
            "rows": rows,
        }

    def _save_custom_gestures_now(self):
        if not save_custom_gestures(CUSTOM_GESTURES_FILE, self.custom_gestures):
            self.custom_gesture_message = "Could not save custom_gestures.json"

    def _reset_custom_gesture_triggers(self):
        self.custom_gesture_triggers = {"left": HoldTrigger(), "right": HoldTrigger(), "overlap": HoldTrigger()}
        self.custom_gesture_live = {}
        self.custom_gesture_rings = {}

    def _custom_gesture_hold(self, index):
        """Full hold time with the circle timer on; a brief stability hold with it off."""
        if 0 <= index < len(self.custom_gestures) and not self.custom_gestures[index].get("timer", True):
            return CUSTOM_GESTURE_INSTANT_HOLD
        return float(self.custom_gesture_hold_seconds)

    def _start_custom_gesture_recording(self, side, now):
        if len(self.custom_gestures) >= CUSTOM_GESTURE_MAX:
            self.custom_gesture_message = f"All {CUSTOM_GESTURE_MAX} slots are used. Delete one first."
            return
        self.custom_gesture_recording = {"side": side, "started": now, "samples": [], "pinch": []}
        self._reset_custom_gesture_triggers()
        self.custom_gesture_message = f"Make the pose with your {side.upper()} hand and hold it still…"

    def _update_custom_gesture_recording(self, hands_by_side, now):
        rec = self.custom_gesture_recording
        side = rec["side"]
        elapsed = now - rec["started"]
        if elapsed < self.CUSTOM_GESTURE_COUNTDOWN:
            remaining = math.ceil(self.CUSTOM_GESTURE_COUNTDOWN - elapsed)
            self.custom_gesture_message = f"Make the pose with your {side.upper()} hand… recording in {remaining}"
            return
        entry = hands_by_side.get(side)
        if entry is not None:
            features = hand_features(entry[0])
            if features is not None:
                rec["samples"].append(features)
                ratio = custom_pinch_ratio(entry[0])
                if ratio is not None:
                    rec["pinch"].append(ratio)
        count = len(rec["samples"])
        self.custom_gesture_message = f"Recording {side.upper()} hand… hold still  {count}/{self.CUSTOM_GESTURE_SAMPLES}"
        if count >= self.CUSTOM_GESTURE_SAMPLES:
            self._finish_custom_gesture_recording()
        elif elapsed > self.CUSTOM_GESTURE_TIMEOUT:
            self.custom_gesture_recording = None
            self.custom_gesture_message = f"Recording cancelled: your {side.upper()} hand wasn't visible."

    def _finish_custom_gesture_recording(self):
        rec = self.custom_gesture_recording
        self.custom_gesture_recording = None
        side = rec["side"]
        features = mean_features(rec["samples"])
        if features is None:
            self.custom_gesture_message = "Recording failed. Try again."
            return
        pinch = sorted(rec["pinch"])[len(rec["pinch"])//2] if rec["pinch"] else None
        conflict = builtin_conflict(features, pinch)
        if conflict:
            self.custom_gesture_message = conflict + " Try a different pose."
            return
        similar = best_pose_match(features, self.custom_gestures, side, self.custom_gesture_tolerance)
        if similar is not None:
            self.custom_gesture_message = (f"Too similar to slot {similar+1} "
                                           f"({self.custom_gestures[similar]['label']}). Try a different pose.")
            return
        label = describe_pose(features)
        self.custom_gestures.append({"kind": "pose", "side": side, "features": features,
                                     "label": label, "action": "none", "timer": True})
        self._save_custom_gestures_now()
        self.custom_gesture_message = f"Saved {label} ({side.upper()} hand). Pick its action from the ▾ list."

    def _run_custom_gesture_action(self, index, side):
        if not (0 <= index < len(self.custom_gestures)):
            return
        gesture = self.custom_gestures[index]
        action = gesture.get("action", "none")
        if action == "none":
            self.status = f"Gesture {gesture['label']} recognised (no action set)"
            self.custom_gesture_message = self.status
            return
        if action == "custom_key":
            combo = str(gesture.get("key_combo") or "").strip()
            if not combo:
                self.custom_gesture_key_capture = index
                self.custom_gesture_message = "Press the keyboard shortcut to assign it to this gesture."
                self.setFocus()
                return
            if pyautogui is None:
                self.custom_gesture_message = "Keyboard shortcut unavailable: pyautogui is not installed."
                return
            keys = [part.strip().lower() for part in combo.split("+") if part.strip()]
            try:
                pyautogui.hotkey(*keys, _pause=False)
            except Exception as exc:
                self.custom_gesture_message = f"Could not send {combo}: {exc}"
                return
        elif action.startswith("mouse_"):
            if not self._custom_gesture_mouse_button(action, side):
                return
        elif action == "toggle_glass_menu":
            self._toggle_glass_menu_from_voice()
        elif action == "open_search":
            self.open_search_bar(from_voice=False)
        else:
            self.execute_pc_voice_command(action)
        self.status = f"Gesture {gesture['label']} → {CUSTOM_GESTURE_ACTION_LABELS.get(action, action)}"
        self.custom_gesture_message = self.status

    def _custom_gesture_mouse_button(self, action, side):
        """Press a mouse button at the gesture's TLI (midway between TLIs for overlap).

        Follows the same rule as pinch clicks: a detached hand never moves or clicks the
        native pointer, and presses over the glass panel are ignored.
        """
        sides = ("left", "right") if side == "both" else (side,)
        if any(self.mouse_detached_by_side.get(s, False) for s in sides):
            self.status = "Gesture mouse button skipped: that hand is detached"
            return False
        points = [self.side_cursor_positions.get(s) for s in sides]
        if any(p is None for p in points):
            return False
        point = QPointF(sum(p.x() for p in points)/len(points), sum(p.y() for p in points)/len(points))
        if self.desktop_mode != "os" or self._canvas_menu_panel_contains(point):
            self.status = "Gesture mouse buttons only act on the desktop, not the glass menu"
            return False
        now = time.monotonic()
        if action == "mouse_right_click":
            self._activate_canvas_right_click(point, side if side in ("left", "right") else "right", now)
            return True
        self.click_side = side if side in ("left", "right") else "right"
        self.click_flash_until = now + 0.24
        try:
            self._set_os_pointer(point)
            if action == "mouse_middle_click":
                if pyautogui is None:
                    raise RuntimeError("pyautogui is not available")
                pyautogui.click(button="middle", _pause=False)
                return True
            forward = action == "mouse_forward"
            if sys.platform.startswith("win"):
                # pyautogui has no X buttons; send XBUTTON1 (back) / XBUTTON2 (forward).
                class _MOUSEINPUT(ctypes.Structure):
                    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
                class _INPUTUNION(ctypes.Union):
                    _fields_ = [("mi", _MOUSEINPUT)]
                class _INPUT(ctypes.Structure):
                    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]
                button = 2 if forward else 1
                events = (_INPUT * 2)(
                    _INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, button, 0x0080, 0, 0))),  # XDOWN
                    _INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, button, 0x0100, 0, 0))),  # XUP
                )
                ctypes.windll.user32.SendInput(2, events, ctypes.sizeof(_INPUT))
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdotool", "click", "9" if forward else "8"], check=True, timeout=0.35,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.status = "Mouse back/forward buttons aren't supported on this system"
                return False
            return True
        except Exception as exc:
            self.status = f"Gesture mouse button failed: {exc}"
            return False

    def _process_custom_gestures(self, hands_by_side, cursor_points, poses, now):
        """Recognise recorded poses per hand and the two-TLI overlap; fire after a hold."""
        if self.custom_gesture_recording is not None:
            self._update_custom_gesture_recording(hands_by_side, now)
            self._reset_custom_gesture_triggers()
            return
        live = {}
        rings = {}
        has_pose = any(g.get("kind") == "pose" for g in self.custom_gestures)
        for side in ("left", "right"):
            candidate = None
            entry = hands_by_side.get(side)
            pose = poses.get(side)
            if (has_pose and entry is not None and pose is not None
                    and not pose.get("fist") and not pose.get("pinch")
                    and self.active_action.get(side) is None
                    and not self.pinch_drag_active.get(side, False)):
                candidate = best_pose_match(hand_features(entry[0]), self.custom_gestures,
                                            side, self.custom_gesture_tolerance)
            trigger = self.custom_gesture_triggers[side]
            hold = self._custom_gesture_hold(candidate) if candidate is not None else 0.0
            fired = trigger.update(candidate, now, hold)
            if candidate is not None:
                progress = trigger.progress(now, hold)
                live[candidate] = max(live.get(candidate, 0.0), progress)
                if self.custom_gestures[candidate].get("timer", True) and not trigger.fired:
                    rings[side] = (self.custom_gestures[candidate].get("label", ""), progress,
                                   max(0.0, hold*(1.0-progress)))
            if fired is not None:
                self._run_custom_gesture_action(fired, side)

        # Built-in: overlapping both TLIs (open hands, nothing else in progress) and
        # holding for the gesture hold time toggles the glass menu. A ring shows progress.
        overlapping = False
        left_point, right_point = cursor_points.get("left"), cursor_points.get("right")
        if left_point is not None and right_point is not None and all(
                self.active_action.get(s) is None and not self.pinch_drag_active.get(s, False)
                and not poses.get(s, {}).get("fist") and not poses.get(s, {}).get("pinch")
                for s in ("left", "right")):
            overlapping = tli_overlap((left_point.x(), left_point.y()), (right_point.x(), right_point.y()),
                                      float(self.tli_diameter))
        trigger = self.custom_gesture_triggers["overlap"]
        hold = float(self.custom_gesture_hold_seconds)
        fired = trigger.update("glass_menu" if overlapping else None, now, hold)
        if overlapping and not trigger.fired:
            progress = trigger.progress(now, hold)
            rings["overlap"] = ("GLASS MENU", progress, max(0.0, hold*(1.0-progress)))
        if fired is not None:
            self._toggle_glass_menu_from_voice()
            self.status = "TLI overlap → glass menu " + ("shown" if self.menu_open and self.hub_visible else "hidden")
        self.custom_gesture_live = live
        self.custom_gesture_rings = rings

    def _click_custom_gestures_page(self, point, cx, cy, scale, now):
        layout = self._custom_gesture_layout(cx, cy, scale)
        if self.custom_gesture_recording is not None:
            # Any press during recording cancels it, so a stray pinch can't save a bad pose.
            self.custom_gesture_recording = None
            self.custom_gesture_message = "Recording cancelled."
            return
        if layout["record_left"].contains(point):
            self._start_custom_gesture_recording("left", now); return
        if layout["record_right"].contains(point):
            self._start_custom_gesture_recording("right", now); return
        for index, row in enumerate(layout["rows"][:len(self.custom_gestures)]):
            if not row["row"].contains(point):
                continue
            gesture = self.custom_gestures[index]
            if row["delete"].contains(point):
                self.custom_gestures.pop(index)
                self._reset_custom_gesture_triggers()
                self.custom_gesture_message = f"Deleted {gesture['label']}."
            elif row["timer"].contains(point):
                gesture["timer"] = not gesture.get("timer", True)
                self._reset_custom_gesture_triggers()
                self.custom_gesture_message = (f"{gesture['label']}: circle timer "
                                               + ("on" if gesture["timer"] else "off (fires instantly)"))
            elif row["side"].contains(point) and gesture.get("kind") == "pose":
                order = ("left", "right", "either")
                gesture["side"] = order[(order.index(gesture.get("side", "either"))+1) % 3]
                self._reset_custom_gesture_triggers()
            elif row["action"].contains(point):
                # Once CUSTOM KEY is selected, tapping its action field again is a
                # direct, visible way to assign or replace the shortcut.
                if gesture.get("action") == "custom_key":
                    self.custom_gesture_key_capture = index
                    self.custom_gesture_message = (
                        f"Set shortcut for {gesture.get('label', 'gesture')}: "
                        "press the key combination now (Ctrl/Alt/Shift + key). Esc cancels."
                    )
                    self.setFocus(Qt.OtherFocusReason)
                    self.activateWindow()
                    self.update()
                else:
                    self.custom_gesture_dropdown = index
                return
            else:
                return
            self._save_custom_gestures_now()
            return

    def _custom_gesture_dropdown_layout(self, cx, cy, scale):
        """Geometry of the open action list: grouped columns showing every action at once."""
        line_h, item_h, col_w, gap = 28.0, 25.0, 110.0, 6.0
        columns = len(CUSTOM_GESTURE_ACTION_GROUPS)
        width = columns*col_w + (columns-1)*gap + 16.0
        lines = max(sum(1 + len(ids) for _title, ids in column) for column in CUSTOM_GESTURE_ACTION_GROUPS)
        # Open just below the tab strip so the list never covers the Back button.
        top = self._settings_layout("gestures")["top"]
        height = 34.0 + lines*line_h + 8.0
        panel = QRectF(cx-width*0.5*scale, cy+top*scale, width*scale, height*scale)
        headings, items = [], []
        for col_index, column in enumerate(CUSTOM_GESTURE_ACTION_GROUPS):
            x = -width*0.5 + 8.0 + col_index*(col_w+gap)
            y = top + 34.0
            for title, ids in column:
                headings.append((title, QRectF(cx+x*scale, cy+y*scale, col_w*scale, line_h*scale)))
                y += line_h
                for action_id in ids:
                    items.append((action_id, QRectF(cx+x*scale, cy+(y+1.5)*scale, col_w*scale, item_h*scale)))
                    y += line_h
        title = QRectF(panel.left()+12*scale, panel.top()+6*scale, panel.width()-60*scale, 24*scale)
        close = QRectF(panel.right()-38*scale, panel.top()+6*scale, 28*scale, 24*scale)
        return {"panel": panel, "title": title, "close": close, "headings": headings, "items": items}

    def _click_custom_gesture_dropdown(self, point, cx, cy, scale):
        index = self.custom_gesture_dropdown
        self.custom_gesture_dropdown = None
        if index is None or not (0 <= index < len(self.custom_gestures)):
            return
        layout = self._custom_gesture_dropdown_layout(cx, cy, scale)
        for action_id, rect in layout["items"]:
            if rect.contains(point):
                gesture = self.custom_gestures[index]
                gesture["action"] = action_id
                if action_id == "custom_key":
                    self.custom_gesture_key_capture = index
                    self.custom_gesture_message = ("CUSTOM KEY selected. Press any keyboard shortcut "
                                                   "(Ctrl, Alt, Shift and/or a key). Esc cancels capture. "
                                                   "Tap this action field again to change it later.")
                    self.setFocus(Qt.OtherFocusReason)
                    self.activateWindow()
                else:
                    self.custom_gesture_key_capture = None
                    self.custom_gesture_message = (f"{gesture['label']} → "
                                                   f"{CUSTOM_GESTURE_ACTION_LABELS.get(action_id, action_id)}")
                self._save_custom_gestures_now()
                return
        # Anything else (close button, headings, outside the list) just closes it.

    def _paint_custom_gesture_dropdown(self, painter, cx, cy, scale):
        index = self.custom_gesture_dropdown
        if index is None or not (0 <= index < len(self.custom_gestures)):
            self.custom_gesture_dropdown = None
            return
        gesture = self.custom_gestures[index]
        layout = self._custom_gesture_dropdown_layout(cx, cy, scale)
        font = lambda size, weight=QFont.DemiBold: self._ui_font(max(6, int(size*scale)), weight)
        panel = layout["panel"]
        painter.setPen(QPen(QColor(120, 225, 245, 40), 10*scale))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(panel.adjusted(-4*scale, -4*scale, 4*scale, 4*scale), 16*scale, 16*scale)
        painter.setBrush(QBrush(QColor(6, 24, 38, 248)))
        painter.setPen(QPen(QColor(175, 250, 255, 230), 1.5*scale))
        painter.drawRoundedRect(panel, 14*scale, 14*scale)

        painter.setPen(QColor(235, 255, 255))
        painter.setFont(font(9))
        painter.drawText(layout["title"], Qt.AlignLeft | Qt.AlignVCenter,
                         f"CHOOSE ACTION FOR {gesture.get('label', 'GESTURE')}")
        painter.setBrush(QBrush(QColor(120, 30, 38, 230)))
        painter.setPen(QPen(QColor(255, 150, 150, 200), 1.0*scale))
        painter.drawRoundedRect(layout["close"], 7*scale, 7*scale)
        painter.setPen(QColor(255, 235, 235))
        painter.drawText(layout["close"], Qt.AlignCenter, "×")

        painter.setFont(font(7))
        for title, rect in layout["headings"]:
            painter.setPen(QColor(135, 245, 250, 235))
            painter.drawText(rect.adjusted(4*scale, 0, 0, 0), Qt.AlignLeft | Qt.AlignBottom, title)
            painter.setPen(QPen(QColor(135, 245, 250, 70), 1.0*scale))
            painter.drawLine(QPointF(rect.left()+4*scale, rect.bottom()+1*scale),
                             QPointF(rect.right()-4*scale, rect.bottom()+1*scale))

        hover_points = [p for p in self.side_cursor_positions.values() if p is not None]
        current = gesture.get("action", "none")
        for action_id, rect in layout["items"]:
            hovered = any(rect.contains(p) for p in hover_points)
            selected = action_id == current
            if selected:
                fill, edge = QColor(40, 150, 120, 240), QColor(150, 255, 200, 240)
            elif hovered:
                fill, edge = QColor(50, 165, 185, 235), QColor(205, 252, 255, 245)
            else:
                fill, edge = QColor(14, 50, 70, 225), QColor(125, 235, 245, 90)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(edge, 1.0*scale))
            painter.drawRoundedRect(rect, 6*scale, 6*scale)
            painter.setPen(QColor(245, 255, 255) if (selected or hovered or action_id != "none")
                           else QColor(170, 200, 210))
            label = CUSTOM_GESTURE_ACTION_LABELS.get(action_id, action_id)
            self._draw_fitted_button_label(painter, rect, ("✓ " + label) if selected else label,
                                           base_size=7, min_size=6, inset=4)

    def _paint_custom_gestures_page(self, painter, cx, cy, scale, now):
        layout = self._custom_gesture_layout(cx, cy, scale)
        font = lambda size, weight=QFont.DemiBold: self._ui_font(max(6, int(size*scale)), weight)
        border = QPen(QColor(125, 235, 245, 210), 1.5*scale)

        painter.setPen(QColor(195, 235, 242))
        self._draw_fitted_button_label(painter, layout["intro"],
                                       "Record a hand pose, then pick its action from the ▾ list.\n"
                                       "Hold the pose for the hold time to trigger it.",
                                       base_size=max(6, int(8*scale)), min_size=6,
                                       weight=QFont.Normal, inset=2)

        recording = self.custom_gesture_recording
        for side in ("left", "right"):
            rect = layout["record_" + side]
            active = recording is not None and recording["side"] == side
            painter.setBrush(QBrush(QColor(150, 40, 45, 235) if active else QColor(18, 80, 93, 235)))
            painter.setPen(border)
            painter.drawRoundedRect(rect, 9*scale, 9*scale)
            painter.setPen(QColor(240, 255, 255))
            self._draw_fitted_button_label(painter, rect,
                                           ("● RECORDING " if active else "● RECORD ") + side.upper() + " HAND",
                                           base_size=max(6, int(8*scale)), min_size=6, inset=6*scale)

        # Built-in gestures are listed for reference only; they can't be edited here.
        painter.setBrush(QBrush(QColor(10, 37, 55, 140)))
        painter.setPen(QPen(QColor(125, 235, 245, 90), 1.0*scale, Qt.DashLine))
        painter.drawRoundedRect(layout["overlap"], 9*scale, 9*scale)
        painter.setPen(QColor(195, 235, 242))
        self._draw_fitted_button_label(painter, layout["overlap"],
                                       "BUILT-IN: OVERLAP BOTH TLIs AND HOLD → GLASS MENU",
                                       base_size=max(6, int(7*scale)), min_size=6, inset=8*scale)

        if self.custom_gesture_message:
            painter.setPen(QColor(255, 225, 150) if recording is not None else QColor(170, 245, 220))
            self._draw_wrapped_fitted_text(painter, layout["message"], self.custom_gesture_message,
                                           base_size=max(6, int(8*scale)), min_size=6)

        for index, row in enumerate(layout["rows"]):
            if index >= len(self.custom_gestures):
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(125, 235, 245, 70), 1.0*scale, Qt.DashLine))
                painter.drawRoundedRect(row["row"], 10*scale, 10*scale)
                painter.setPen(QColor(170, 220, 230, 90))
                painter.setFont(font(7))
                painter.drawText(row["row"], Qt.AlignCenter, "EMPTY SLOT")
                continue
            gesture = self.custom_gestures[index]
            progress = self.custom_gesture_live.get(index)
            painter.setBrush(QBrush(QColor(24, 89, 92, 225) if progress is not None else QColor(10, 37, 55, 205)))
            painter.setPen(QPen(QColor(150, 255, 200, 240) if progress is not None else QColor(125, 235, 245, 150),
                                1.5*scale))
            painter.drawRoundedRect(row["row"], 10*scale, 10*scale)
            if progress is not None:
                bar = QRectF(row["row"].left()+8*scale, row["row"].bottom()-4*scale,
                             (row["row"].width()-16*scale)*progress, 2.5*scale)
                painter.fillRect(bar, QColor(110, 255, 170, 230))

            side_text = {"left": "LEFT", "right": "RIGHT", "either": "ANY", "both": "BOTH"}.get(gesture.get("side"), "ANY")
            painter.setBrush(QBrush(QColor(18, 80, 93, 235)))
            painter.setPen(QPen(QColor(125, 235, 245, 160), 1.0*scale))
            painter.drawRoundedRect(row["side"], 7*scale, 7*scale)
            painter.setPen(QColor(235, 255, 255))
            painter.setFont(font(7))
            painter.drawText(row["side"], Qt.AlignCenter, side_text)

            # Per-gesture circle timer toggle: ON = hold with a filling ring at the TLI,
            # OFF = fire after a brief stability check.
            timer_on = gesture.get("timer", True)
            painter.setBrush(QBrush(QColor(55, 145, 82, 235) if timer_on else QColor(75, 80, 89, 205)))
            painter.setPen(QPen(QColor(155, 245, 180, 220) if timer_on else QColor(155, 162, 172, 165), 1.0*scale))
            painter.drawRoundedRect(row["timer"], 7*scale, 7*scale)
            icon = QRectF(row["timer"].left()+6*scale, row["timer"].center().y()-7*scale, 14*scale, 14*scale)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(235, 255, 255, 90 if timer_on else 60), 1.6*scale))
            painter.drawEllipse(icon)
            painter.setPen(QPen(QColor(150, 255, 200) if timer_on else QColor(150, 175, 185), 2.2*scale,
                                Qt.SolidLine, Qt.RoundCap))
            if timer_on:
                painter.drawArc(icon, 90*16, -270*16)
            else:
                painter.drawLine(icon.topLeft()+QPointF(2*scale, 2*scale), icon.bottomRight()-QPointF(2*scale, 2*scale))
            painter.setPen(QColor(235, 255, 255) if timer_on else QColor(170, 200, 210))
            painter.setFont(font(7))
            painter.drawText(row["timer"].adjusted(20*scale, 0, -2*scale, 0), Qt.AlignCenter, "ON" if timer_on else "OFF")

            painter.setPen(QColor(245, 255, 255))
            self._draw_fitted_button_label(painter, row["label"], gesture.get("label", "POSE"),
                                           base_size=9, min_size=6, inset=3)

            action = gesture.get("action", "none")
            action_label = CUSTOM_GESTURE_ACTION_LABELS.get(action, action)
            if action == "custom_key":
                action_label = str(gesture.get("key_combo") or "SET KEY COMBO…")
            is_open = self.custom_gesture_dropdown == index
            painter.setBrush(QBrush(QColor(24, 89, 92, 240) if is_open else QColor(10, 47, 67, 235)))
            painter.setPen(QPen(QColor(175, 250, 255, 230) if is_open else QColor(125, 235, 245, 160), 1.0*scale))
            painter.drawRoundedRect(row["action"], 7*scale, 7*scale)
            painter.setPen(QColor(170, 200, 210) if action == "none" else QColor(235, 255, 255))
            self._draw_fitted_button_label(painter, row["action"].adjusted(0, 0, -22*scale, 0),
                                           action_label, base_size=8, min_size=6, inset=4)
            painter.setPen(QColor(175, 250, 255))
            painter.setFont(font(9))
            painter.drawText(QRectF(row["action"].right()-24*scale, row["action"].top(), 20*scale,
                                    row["action"].height()), Qt.AlignCenter, "▴" if is_open else "▾")

            painter.setBrush(QBrush(QColor(120, 30, 38, 230)))
            painter.setPen(QPen(QColor(255, 150, 150, 200), 1.0*scale))
            painter.drawRoundedRect(row["delete"], 7*scale, 7*scale)
            painter.setPen(QColor(255, 235, 235))
            painter.setFont(font(9))
            painter.drawText(row["delete"], Qt.AlignCenter, "×")

    def _update_hand_scale_origins(self, hands_by_side):
        """Per-hand wrist pivots, blended to one shared pivot when the hands touch.

        Apart, each virtual hand scales about its own wrist so moving one hand never
        moves the other. When the real hands are really close (fingertips, fingertip to
        palm, or palm to palm), both hands blend to scaling about the midpoint of the
        wrists. A common pivot preserves relative geometry exactly, so touching hands
        render touching at any hand scale. The blend is eased over a few frames so it
        reads as the hands meeting rather than snapping. Points are screen-space.
        """
        wrists = {}
        screen_points = {}
        palm_sizes_screen = {}
        for side, (landmarks, _score) in hands_by_side.items():
            if len(landmarks) < 21:
                continue
            pts = []
            for lm in landmarks[:21]:
                q = self.interactive_visual_point(float(lm.x), float(lm.y))
                pts.append((q.x(), q.y()))
            screen_points[side] = pts
            wrists[side] = pts[0]
            palm_length = math.hypot(pts[9][0]-pts[0][0], pts[9][1]-pts[0][1])
            palm_width = math.hypot(pts[17][0]-pts[5][0], pts[17][1]-pts[5][1])
            if palm_length > 1.0 and palm_width > 1.0:
                palm_sizes_screen[side] = math.sqrt(palm_length*palm_width)

        # Thresholds are palm-relative, so behaviour is the same near or far from camera.
        calibration = getattr(self, "hand_calibration", {})
        touch_ratio = float(calibration.get("proximity_touch_ratio", 0.28))
        interaction_ratio = float(calibration.get("proximity_interaction_ratio", 2.5))
        touch_distance, _interaction_distance = proximity_thresholds(
            palm_sizes_screen, touch_ratio, interaction_ratio
        )
        palm_unit = (sum(palm_sizes_screen.values())/len(palm_sizes_screen)
                     if palm_sizes_screen else 40.0)
        band = max(0.0, palm_unit*max(0.0, float(getattr(self, "hand_contact_band_ratio", 0.02))))

        target = 0.0
        if band <= 1e-6:
            # Hard bypass at zero: no contact detection, shared pivot, or residual easing.
            # Each virtual hand remains anchored to its own current wrist independently.
            self.hand_contact_last_update = None
            self.hand_contact_weight = 0.0
            self.hand_scale_origins = {side: QPointF(x, y) for side, (x, y) in wrists.items()}
            base_scale = min(1.0, max(0.1, float(getattr(self, "hand_visual_scale", 0.3))))
            self.hand_contact_scales = {side: base_scale for side in wrists}
            return
        elif "left" in screen_points and "right" in screen_points:
            gap = hand_contact_gap(screen_points["left"], screen_points["right"])
            target = shared_contact_weight(gap, touch_distance, touch_distance + band)
            now = time.monotonic()
            last = getattr(self, "hand_contact_last_update", None)
            dt = (now - last) if last is not None else 1.0
            self.hand_contact_last_update = now
            weight = ease_toward(
                getattr(self, "hand_contact_weight", 0.0), target, dt,
                engage_seconds=0.05 / max(0.2, float(getattr(self, "palm_attract_acceleration", 0.33))),
                release_seconds=0.09 / max(0.2, float(getattr(self, "palm_attract_acceleration", 0.33))),
            )
        else:
            # One hand only: there is no partner to share a pivot with.
            self.hand_contact_last_update = None
            weight = 0.0
        self.hand_contact_weight = weight

        origins = shared_contact_origins(wrists, weight)
        self.hand_scale_origins = {side: QPointF(x, y) for side, (x, y) in origins.items()}
        base_scale = min(1.0, max(0.1, float(getattr(self, "hand_visual_scale", 0.3))))
        self.hand_contact_scales = {side: base_scale for side in wrists}

    def _draw_fitted_button_label(self, painter, rect, text, base_size=10, min_size=7,
                                  weight=QFont.DemiBold, inset=4):
        """Draw a legible button label without letting text escape its rounded button."""
        text = str(text)
        lines = text.split("\n")
        available_width = max(8.0, rect.width() - 2.0*inset)
        available_height = max(8.0, rect.height() - 4.0)
        size = int(base_size)
        while size > min_size:
            font = self._ui_font(size, weight)
            metrics = QFontMetricsF(font)
            if (max((metrics.horizontalAdvance(line) for line in lines), default=0.0) <= available_width
                    and metrics.height()*len(lines) <= available_height):
                break
            size -= 1
        font = self._ui_font(max(min_size, size), weight)
        painter.setFont(font)
        # Last resort at the minimum size: elide rather than spill past the button edge.
        metrics = QFontMetricsF(font)
        if max((metrics.horizontalAdvance(line) for line in lines), default=0.0) > available_width:
            text = "\n".join(metrics.elidedText(line, Qt.ElideRight, available_width) for line in lines)
        painter.drawText(rect.adjusted(inset, 1, -inset, -1), Qt.AlignCenter, text)

    def _draw_wrapped_fitted_text(self, painter, rect, text, base_size=8, min_size=6,
                                  weight=QFont.Normal, align=Qt.AlignCenter):
        """Word-wrap text inside rect, shrinking the font until the wrapped block fits."""
        text = str(text)
        flags = int(align) | int(Qt.TextWordWrap)
        size = int(base_size)
        while size > min_size:
            metrics = QFontMetricsF(self._ui_font(size, weight))
            bounds = metrics.boundingRect(QRectF(0, 0, rect.width(), 10000.0), flags, text)
            if bounds.height() <= rect.height() and bounds.width() <= rect.width()+0.5:
                break
            size -= 1
        painter.setFont(self._ui_font(max(min_size, size), weight))
        painter.save()
        painter.setClipRect(rect.adjusted(-2, -2, 2, 2))
        painter.drawText(rect, flags, text)
        painter.restore()

    @staticmethod
    def _settings_slider_value_text(name, value):
        if name == "smoothing":
            return f"{value:.2f}"
        if name == "pinch_timer":
            return f"{value:.1f}s"
        if name in ("custom_gesture_hold_seconds", "default_gesture_hold_seconds", "pinch_buffer_seconds"):
            return f"{value:.2f}s"
        if name == "custom_gesture_tolerance":
            return f"{value:.2f}"
        if name == "hand_contact_band_ratio":
            return f"{value:.2f} PALM"
        if name == "palm_attract_acceleration":
            return f"{value:.2f}x"
        if name in ("skin_tone_darkness", "hand_transparency", "scroll_backdrop_opacity",
                    "detached_notice_opacity", "ui_font_scale"):
            return f"{value:.0f}%"
        if name in ("finger_thickness", "hand_scale", "palm_width_scale", "tli_glow_intensity", "skeleton_line_thickness"):
            return f"{value:.2f}x"
        if name == "tli_colour_shift_degrees":
            return f"{value:.0f}°"
        if name == "scroll_backdrop_hue":
            return f"{value:.0f}/255"
        if name in ("tli_border_thickness", "tli_diameter", "scroll_deadzone"):
            return f"{value:.1f}px"
        if name == "scroll_notch_hue":
            notch = QColor.fromHsv(int(value) % 360, 190, 255)
            return f"RGB {notch.red()}, {notch.green()}, {notch.blue()}"
        return f"{value:.2f}x"

    def _paint_settings_sliders(self, painter, cx, cy, scale):
        """Draw every slider on the current page from the shared layout."""
        geometry = self._settings_slider_layouts(cx, cy, scale)
        for name, _offset, value, minimum, maximum, label in self._settings_slider_specs():
            if name not in geometry:
                continue
            left, right, yy, label_rect = geometry[name]
            painter.setPen(QPen(QColor(100, 190, 205, 180), 12*scale, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(left, yy), QPointF(right, yy))
            painter.setPen(QPen(QColor(125, 245, 250, 245), 4*scale, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(left, yy), QPointF(right, yy))
            ratio = min(1.0, max(0.0, (value-minimum)/max(1e-6, maximum-minimum)))
            painter.setBrush(QBrush(QColor(80, 255, 150, 230)))
            painter.setPen(QPen(QColor(220, 255, 245), 2*scale))
            painter.drawEllipse(QPointF(left+ratio*(right-left), yy), 8*scale, 8*scale)
            if name == "scroll_notch_hue":
                # Colour swatch at the end of its (shortened) track, centred on it.
                painter.setPen(QPen(QColor(225, 250, 255, 235), 1.5*scale))
                painter.setBrush(QBrush(QColor.fromHsv(int(value) % 360, 190, 255)))
                painter.drawEllipse(QPointF(right+20*scale, yy), 10*scale, 10*scale)
            painter.setPen(QColor(210, 245, 240))
            self._draw_fitted_button_label(painter, label_rect,
                                           f"{label}  ·  {self._settings_slider_value_text(name, value)}",
                                           base_size=max(7, int(8*scale)), min_size=6, inset=2)

    def _paint_reference_columns(self, painter, columns, cx, cy, scale):
        """Two-column Gesture Map / Speech Commands reference with one font size per role."""
        layout = self._settings_layout()
        col_w = self.SETTINGS_COLUMN_W*scale
        x_starts = (cx+self.SETTINGS_LEFT_X*scale, cx+self.SETTINGS_RIGHT_X*scale)
        top_y = cy+layout["top"]*scale
        line_h = 17.0*scale
        usable = col_w-8*scale

        def fitted(base, weight, texts):
            size = max(7, int(base*scale))
            while size > 6:
                metrics = QFontMetricsF(self._ui_font(size, weight))
                if all(metrics.horizontalAdvance(t) <= usable for t in texts):
                    break
                size -= 1
            return self._ui_font(size, weight)

        all_entries = [entry for column in columns for entry in column if entry[0]]
        heading_font = fitted(8.8, QFont.DemiBold, [t for t, heading in all_entries if heading])
        body_font = fitted(8.3, QFont.Normal, [t for t, heading in all_entries if not heading])
        for col_index, entries in enumerate(columns):
            yy = top_y
            for text, is_heading in entries:
                if not text:
                    yy += 7*scale
                    continue
                painter.setFont(heading_font if is_heading else body_font)
                painter.setPen(QColor(135, 245, 250, 245) if is_heading else QColor(232, 248, 252, 238))
                text_rect = QRectF(x_starts[col_index], yy, col_w, line_h+2*scale)
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)
                yy += line_h

    def _update_hand_contact_display(self):
        """Let nearly touching fingertips of the two drawn hands touch (see hand_contact).

        Works in the drawn hands' screen space, with the same mapping as
        _paint_hand_skeleton, and stores the result as display landmarks only.
        """
        style = getattr(self, "hand_render_style", "flat")
        if style not in ("3d", "flat"):
            return
        displays = getattr(self, "hand_display_positions", {})
        screen, palms, scales = {}, {}, {}
        for side in ("left", "right"):
            points = self.hand_positions.get(side)
            display = displays.get(side)
            if not points or len(points) < 21 or display is None or display[0] is not points:
                return
            shown = display[1]
            origin, user_scale = self._hand_visual_transform(points, side)
            if self.side_cursor_positions.get(side) is not None:
                off_x, off_y = self.hand_draw_offset.get(side, (0.0, 0.0))
            else:
                off_x = off_y = 0.0
            pts = []
            for p in shown[:21]:
                q = self.interactive_visual_point(p[0], p[1])
                x, y = scale_hand_point((q.x(), q.y()), (origin.x(), origin.y()), user_scale)
                pts.append((x+off_x, y+off_y))
            palm = math.hypot(pts[9][0]-pts[0][0], pts[9][1]-pts[0][1])
            screen[side] = pts
            palms[side] = palm*self._landmark_foreshortening(shown, 0, 9)
            scales[side] = user_scale
        thickness = min(10.0, max(0.8, float(getattr(self, "finger_thickness", 4.24))))/4.24
        offsets = hand_contact_offsets(screen["left"], screen["right"], palms["left"], palms["right"],
                                       style, thickness)
        px_x = max(1.0, self.width()-1)/0.60
        px_y = max(1.0, self.height()-1)/0.68
        for side in ("left", "right"):
            moves = offsets.get(side)
            if not moves:
                continue
            points, shown = displays[side]
            adjusted = list(shown)
            for index, (dx, dy) in moves.items():
                p = adjusted[index]
                nx = p[0] + dx/(px_x*scales[side])
                ny = p[1] + dy/(px_y*scales[side])
                adjusted[index] = (nx, ny) + tuple(p[2:])
            displays[side] = (points, adjusted)

    def _hand_visual_transform(self, points, side=None):
        """Return one consistent fixed-scale transform for landmarks and the TLI.

        The user-selected scale never changes with hand proximity. When the two hands
        are touching, both share one pivot (see _update_hand_scale_origins).
        """
        visual_scale = min(1.0, max(0.1, float(
            getattr(self, "hand_contact_scales", {}).get(
                side, getattr(self, "hand_visual_scale", 0.3)))))
        if not points or len(points) < 21:
            return QPointF(0.0, 0.0), visual_scale
        origin = self.hand_scale_origins.get(side) if side is not None else None
        if origin is None:
            origin = self.interactive_visual_point(points[0][0], points[0][1])
        return QPointF(origin), visual_scale

    def _scale_hand_tli_point(self, cursor, points=None, side=None):
        """Transform the actual interaction point with the same origin/scale as the hand."""
        if cursor is None:
            return None
        if points is None:
            points = []
        if not points or len(points) < 21:
            return QPointF(cursor)
        origin, visual_scale = self._hand_visual_transform(points, side)
        x, y = scale_hand_point((cursor.x(), cursor.y()), (origin.x(), origin.y()), visual_scale)
        # Clamp to the physical screen, not the Qtile work area. On Linux the reserved
        # top/bottom bars are outside availableGeometry(), but are still valid targets.
        screen = QApplication.primaryScreen()
        geometry = screen.geometry() if screen is not None else None
        width = geometry.width() if geometry is not None else self.width()
        height = geometry.height() if geometry is not None else self.height()
        return QPointF(min(max(0.0, x), max(1.0, width-1.0)),
                       min(max(0.0, y), max(1.0, height-1.0)))

    def _visual_tli_position(self, side, cursor, points=None):
        """Return the already-transformed TLI; only slider reticles override its position."""
        if cursor is None:
            return None
        override = self.tli_overrides.get(side)
        if override and override[0] == "slider":
            return QPointF(override[1])
        # side_cursor_positions now stores the transformed interaction point, so applying
        # the hand transform again here would double-scale the displayed TLI.
        return QPointF(cursor)

    def _hand_projection_scale(self, points, side=None):
        """Estimate apparent hand size so mesh thickness follows camera distance.

        MediaPipe landmark positions already scale the hand geometry with perspective,
        but fixed-width painter strokes do not. A geometric mean of palm length and
        width gives a stable apparent-size estimate that is less sensitive to rotation
        than either dimension alone. It is measured before user hand scaling and then
        applied uniformly to all mesh widths/radii.
        """
        if not points or len(points) < 21:
            return 1.0
        wrist = self.interactive_visual_point(points[0][0], points[0][1])
        middle_mcp = self.interactive_visual_point(points[9][0], points[9][1])
        index_mcp = self.interactive_visual_point(points[5][0], points[5][1])
        pinky_mcp = self.interactive_visual_point(points[17][0], points[17][1])
        palm_length = math.hypot(middle_mcp.x()-wrist.x(), middle_mcp.y()-wrist.y())
        palm_width = math.hypot(pinky_mcp.x()-index_mcp.x(), pinky_mcp.y()-index_mcp.y())
        reference_size = max(1.0, min(float(self.width()), float(self.height()))*0.18)
        calibration = getattr(self, "hand_calibration", {})
        per_hand = calibration.get("reference_palm_size", {}) if calibration.get("calibrated", False) else {}
        try:
            calibrated_reference = float(per_hand.get(side)) if side in ("left", "right") else 0.0
            if math.isfinite(calibrated_reference) and calibrated_reference > 1e-5:
                # The measurement is in normalized camera coordinates; reproduce the
                # current crop's average x/y projection in physical screen pixels.
                reference_size = calibrated_reference*math.sqrt(
                    max(1.0, self.width()-1)/0.60 * max(1.0, self.height()-1)/0.68
                )
        except (TypeError, ValueError, AttributeError):
            pass
        # Turning the hand foreshortens the on-screen palm, which made every finger
        # thinner when the hand was side-on or tilted. MediaPipe's z (same scale as x)
        # lets each palm dimension be corrected back to its unrotated length.
        palm_length *= self._landmark_foreshortening(points, 0, 9)
        palm_width *= self._landmark_foreshortening(points, 5, 17)
        # Direct proportional mapping: larger apparent palm (closer camera) means thicker
        # strokes; smaller palm (farther camera) means thinner strokes, for every segment.
        return relative_palm_projection_scale(math.sqrt(max(1.0, palm_length*palm_width)), reference_size)

    # Camera frames are 16:9; landmark y is normalised by height and z roughly by width.
    LANDMARK_Y_TO_X = 9.0/16.0

    def _landmark_xyz(self, points, index):
        point = points[index]
        z = float(point[2]) if len(point) > 2 else 0.0
        return float(point[0]), float(point[1])*self.LANDMARK_Y_TO_X, z

    def _landmark_foreshortening(self, points, a, b, limit=3.0):
        """True 3D length / on-screen length for one landmark pair (1.0 when unknown)."""
        ax, ay, az = self._landmark_xyz(points, a)
        bx, by, bz = self._landmark_xyz(points, b)
        flat = math.hypot(bx-ax, by-ay)
        full = math.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)
        if not math.isfinite(full) or full <= 1e-6:
            return 1.0
        return min(limit, max(1.0, full/max(flat, full/limit)))

    def _paint_hand_skeleton(self, painter, points, side=None):
        """Render the virtual hand (shaded 3D skin, or the alternate skeleton style).

        Geometry follows tracked landmarks; all widths use the same user-size and
        projected-size factors (corrected for hand rotation) so apparent finger
        thickness stays consistent as the hand moves and turns. The skin style is
        drawn by _paint_hand_skin_3d.
        """
        if not points or len(points) < 21:
            return
        scale_origin, user_scale = self._hand_visual_transform(points, side)
        projection_scale = self._hand_projection_scale(points, side)
        width_scale = user_scale*projection_scale

        # Follow the smoothed TLI (see update_scene) so the TLI never drifts off the
        # thumb/index gap while the hand moves; no offset when the hand has no TLI.
        if side in ("left", "right") and self.side_cursor_positions.get(side) is not None:
            off_x, off_y = getattr(self, "hand_draw_offset", {}).get(side, (0.0, 0.0))
        else:
            off_x = off_y = 0.0

        def sp(index):
            q = self.interactive_visual_point(points[index][0], points[index][1])
            x, y = scale_hand_point((q.x(), q.y()), (scale_origin.x(), scale_origin.y()), user_scale)
            return QPointF(x+off_x, y+off_y)

        # The flat and 3D hands show fingertips the camera can't see as tucked in, instead
        # of MediaPipe's guess; the skeleton style keeps showing the raw tracking.
        style = getattr(self, "hand_render_style", "flat")
        display = getattr(self, "hand_display_positions", {}).get(side) if side in ("left", "right") else None
        if style != "skeleton" and display is not None and display[0] is points:
            points = display[1]

        joints = {i: sp(i) for i in range(21)}
        painter.save()
        painter.setOpacity(painter.opacity() * min(1.0, max(0.10, float(getattr(self, "hand_transparency", 50.0)) / 100.0)))
        darkness = min(1.0, max(0.0, float(getattr(self, "skin_tone_darkness", -50.0))/100.0))

        def skin_color(r, g, b, alpha=255):
            if getattr(self, "skin_use_skeleton_color", False):
                base = QColor(self.left_click_color if side == "left" else self.right_click_color)
                return QColor(base.red(), base.green(), base.blue(), alpha)
            strength = darkness*0.82
            target = (56.0, 31.0, 22.0)
            channels = [round(original+(deep-original)*strength)
                        for original, deep in zip((r, g, b), target)]
            return QColor(channels[0], channels[1], channels[2], alpha)

        edges = [
            (0,1,10.0),(1,2,8.3),(2,3,7.0),(3,4,5.8),
            (0,5,11.0),(5,6,7.8),(6,7,6.0),(7,8,4.9),
            (5,9,8.7),(9,10,7.8),(10,11,6.0),(11,12,4.9),
            (9,13,8.0),(13,14,7.4),(14,15,5.8),(15,16,4.8),
            (13,17,7.8),(17,18,7.0),(18,19,5.5),(19,20,4.6),
            (0,17,9.5),(5,9,6.4),(9,13,6.0),(13,17,5.5),
        ]
        finger_edges = {
            (1,2), (2,3), (3,4),
            (5,6), (6,7), (7,8),
            (9,10), (10,11), (11,12),
            (13,14), (14,15), (15,16),
            (17,18), (18,19), (19,20),
        }
        finger_thickness = min(10.0, max(0.8, float(getattr(self, "finger_thickness", 4.24))))

        # Skeleton mode is a genuine alternate rendering style, not a hide/show switch.
        # It reuses the identical transformed landmarks and camera-distance scaling.
        if style == "skeleton":
            skeleton_color = QColor(self.left_click_color if side == "left" else self.right_click_color)
            skeleton_width_scale = min(4.0, max(0.5, float(getattr(self, "skeleton_line_thickness", 1.5))))
            painter.save()
            painter.setBrush(Qt.NoBrush)
            for a, b, base_width in edges:
                finger_factor = 0.45 + finger_thickness*0.15 if (a, b) in finger_edges else 1.0
                width = max(0.6, (1.1 if (a, b) in finger_edges else 1.45)*finger_factor*width_scale*skeleton_width_scale)
                painter.setPen(QPen(QColor(skeleton_color.red(), skeleton_color.green(), skeleton_color.blue(), 220),
                                    width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(joints[a], joints[b])
            for index, point in joints.items():
                radius = max(1.0, (2.0 + (finger_thickness*0.10 if index in (4, 8, 12, 16, 20) else 0.0))*width_scale*skeleton_width_scale)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(skeleton_color.red(), skeleton_color.green(), skeleton_color.blue(), 235)))
                painter.drawEllipse(point, radius, radius)
            painter.restore()
            painter.restore()
            return

        base_skin = skin_color(225, 174, 143, 255)
        if style == "3d" and self._paint_hand_mesh_3d(painter, points, joints, side, user_scale, base_skin):
            painter.restore()
            return
        palm_width_scale = min(1.8, max(0.65, float(getattr(self, "palm_width_scale", 1.0))))
        self._paint_hand_skin_3d(painter, points, joints, width_scale, finger_thickness,
                                 palm_width_scale, base_skin)
        painter.restore()

    def _paint_hand_mesh_3d(self, painter, points, joints, side, user_scale, base):
        """Draw a lower-cost, cached frame of the rigged 3D hand mesh.

        The GL render/readback path is CPU-expensive on software OpenGL (common on
        Linux desktops). Render at 72% resolution and cap expensive pose renders to
        15 FPS, reusing/repositioning the latest image between renders. Large pose
        changes can still trigger an immediate render so fingers don't lag behind a
        quick open/close gesture.
        """
        if self._hand_mesh_renderer is None:
            try:
                from hand_mesh_3d import HandMeshRenderer
                self._hand_mesh_renderer = HandMeshRenderer()
            except Exception as exc:
                self._report_hand_mesh_problem(f"3D hands unavailable: {exc}")
                return False
        renderer = self._hand_mesh_renderer
        if renderer.failed:
            self._report_hand_mesh_problem(renderer.error)
            return False
        px_per_x = max(1.0, self.width()-1)/0.60*user_scale
        z0 = float(points[0][2]) if len(points[0]) > 2 else 0.0
        target = np.array([[joints[i].x(), joints[i].y(),
                            -((float(points[i][2]) if len(points[i]) > 2 else 0.0)-z0)*px_per_x]
                           for i in range(21)], dtype=np.float64)
        model_side = side if side in ("left", "right") else "right"
        now = time.monotonic()
        colour = (base.red(), base.green(), base.blue())
        render_scale = 0.72
        render_interval = 1.0 / 15.0
        cache = self._hand_mesh_frame_cache.get(model_side)
        render_now = cache is None or (now - cache["rendered_at"]) >= render_interval

        if cache is not None and not render_now:
            # Compare pose relative to the wrist so ordinary hand translation can use
            # a cheap image offset. Big articulation/depth changes bypass the throttle.
            translation = target[0, :2] - cache["target"][0, :2]
            shape_delta = target[:, :2] - cache["target"][:, :2] - translation
            max_shape_delta = float(np.max(np.linalg.norm(shape_delta, axis=1)))
            max_depth_delta = float(np.max(np.abs(target[:, 2] - cache["target"][:, 2])))
            if colour != cache["colour"] or max_shape_delta > 18.0 or max_depth_delta > 22.0:
                render_now = True

        if render_now:
            # Lower pixel workload matters much more than multisampling on CPU-backed
            # GL. Qt smooth-scales this cached image to screen size when it is drawn.
            render_target = target * render_scale
            result = renderer.render(model_side, render_target, colour)
            if result is None:
                self._report_hand_mesh_problem(renderer.error or "3D hand rendering failed")
                return False
            data, w, h, left, top = result
            image = QImage(data, w, h, w*4, QImage.Format_RGBA8888_Premultiplied).copy()
            cache = {
                "image": image,
                "rendered_at": now,
                "target": target.copy(),
                "colour": colour,
                "left": float(left) / render_scale,
                "top": float(top) / render_scale,
                "width": float(w) / render_scale,
                "height": float(h) / render_scale,
            }
            self._hand_mesh_frame_cache[model_side] = cache

        # Translate the cached image by the current wrist movement without rebuilding
        # its vertex pose. Relative finger/palm movement is refreshed at least every
        # 1/15 sec, or immediately when it exceeds the limits above.
        translation = target[0, :2] - cache["target"][0, :2]
        dest = QRectF(cache["left"] + float(translation[0]),
                      cache["top"] + float(translation[1]),
                      cache["width"], cache["height"])
        # The mesh shading already supplies depth and volume; drawing a second, fully
        # copied black silhouette costs another image-sized allocation each frame.
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(dest, cache["image"])
        painter.restore()
        return True

    def _report_hand_mesh_problem(self, message):
        if not self._hand_mesh_notice_shown:
            self._hand_mesh_notice_shown = True
            print("Spatial Desktop:", message, "Showing the flat hand style instead.")
            self.status = "3D hands unavailable — using flat style. " + str(message)[:120]

    # Joint radii (before finger-thickness and size scaling) from knuckle to tip.
    # Cartoon proportions: chunky fingers with little taper and rounded, full tips.
    HAND_FINGER_CHAINS = (
        ("thumb", (1, 2, 3, 4), (5.0, 4.6, 4.2, 3.95)),
        ("index", (5, 6, 7, 8), (4.2, 3.9, 3.65, 3.55)),
        ("middle", (9, 10, 11, 12), (4.3, 4.0, 3.75, 3.65)),
        ("ring", (13, 14, 15, 16), (4.1, 3.8, 3.55, 3.45)),
        ("pinky", (17, 18, 19, 20), (3.6, 3.4, 3.2, 3.1)),
    )
    # Key light from the upper left of the screen.
    HAND_LIGHT = (-0.55, -0.835)

    @staticmethod
    def _shade_colour(color, factor, alpha=None):
        """factor > 1 mixes toward white, factor < 1 darkens."""
        r, g, b = color.red(), color.green(), color.blue()
        if factor >= 1.0:
            t = min(1.0, factor-1.0)
            r, g, b = (round(c+(255-c)*t) for c in (r, g, b))
        else:
            r, g, b = (round(c*max(0.0, factor)) for c in (r, g, b))
        return QColor(r, g, b, color.alpha() if alpha is None else alpha)

    @staticmethod
    def _tapered_capsule(p1, r1, p2, r2):
        """A rounded, tapered limb segment from p1 (radius r1) to p2 (radius r2).

        Built as one closed outline (two sides joined by two end arcs), so it never
        has the winding-cancellation holes a polygon-plus-circles path produces.
        """
        path = QPainterPath()
        dx, dy = p2.x()-p1.x(), p2.y()-p1.y()
        length = math.hypot(dx, dy)
        if length <= 1e-3 or abs(r1-r2) >= length:
            r = max(r1, r2)
            centre = p1 if r1 >= r2 else p2
            path.addEllipse(centre, r, r)
            return path
        ux, uy = dx/length, dy/length
        nx, ny = -uy, ux

        def qt_angle(vx, vy):  # Qt arc angles: counter-clockwise on screen from 3 o'clock
            return math.degrees(math.atan2(-vy, vx))

        def sweep(start, through):
            delta = (through-start+180.0) % 360.0 - 180.0
            return 180.0 if delta > 0 else -180.0

        path.moveTo(QPointF(p1.x()+nx*r1, p1.y()+ny*r1))
        path.lineTo(QPointF(p2.x()+nx*r2, p2.y()+ny*r2))
        start = qt_angle(nx, ny)
        path.arcTo(QRectF(p2.x()-r2, p2.y()-r2, 2*r2, 2*r2), start, sweep(start, qt_angle(ux, uy)))
        path.lineTo(QPointF(p1.x()-nx*r1, p1.y()-ny*r1))
        start = qt_angle(-nx, -ny)
        path.arcTo(QRectF(p1.x()-r1, p1.y()-r1, 2*r1, 2*r1), start, sweep(start, qt_angle(-ux, -uy)))
        path.closeSubpath()
        return path

    @staticmethod
    def _convex_hull(points):
        pts = sorted({(round(p.x(), 3), round(p.y(), 3)) for p in points})
        if len(pts) <= 2:
            return [QPointF(x, y) for x, y in pts]

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower, upper = [], []
        for pt in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                lower.pop()
            lower.append(pt)
        for pt in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                upper.pop()
            upper.append(pt)
        return [QPointF(x, y) for x, y in lower[:-1]+upper[:-1]]

    def _paint_hand_skin_3d(self, painter, points, joints, width_scale, finger_thickness,
                            palm_width_scale, base):
        """Depth-ordered, shaded hand: rounded palm body, tapered cylindrical fingers.

        Parts are drawn back-to-front using MediaPipe's per-landmark depth, so curled
        fingers pass in front of (or behind) the palm as they really do. The convex,
        rounded palm can never fold over itself when the hand turns side-on. Everything
        is composed in an offscreen layer, then drawn once with the hand opacity so
        overlapping parts do not double up in transparency.
        """
        # Depth of each landmark relative to the wrist, in palm lengths (+ = farther
        # from the user). First-person view, as in the 3D style: MediaPipe z grows away
        # from the camera, i.e. toward the user, so it is negated. With the palm toward
        # the camera, curled fingers go behind the back of the hand.
        wx, wy, wz = self._landmark_xyz(points, 0)
        mx, my, mz = self._landmark_xyz(points, 9)
        palm_len = max(1e-4, math.sqrt((mx-wx)**2 + (my-wy)**2 + (mz-wz)**2))
        depth = [-(self._landmark_xyz(points, i)[2]-wz)/palm_len for i in range(21)]

        def depth_size(i):  # nearer the camera looks slightly larger
            return min(1.3, max(0.75, 1.0-0.35*depth[i]))

        def depth_tone(i):  # farther parts are a touch darker
            return 1.0-min(0.2, max(-0.1, 0.3*depth[i]))

        # Finger thickness is proportional to this hand's own palm length (corrected
        # for rotation), so fingers stay in cartoon proportion to the palm at any
        # distance or angle. Finger Thickness scales it around its 4.24 default.
        palm_px = math.hypot(joints[9].x()-joints[0].x(), joints[9].y()-joints[0].y())
        palm_px *= self._landmark_foreshortening(points, 0, 9)
        size = (0.125/4.3)*palm_px*(finger_thickness/4.24)
        radii = {}
        for _name, chain, base_radii in self.HAND_FINGER_CHAINS:
            for index, r in zip(chain, base_radii):
                radii[index] = max(0.8, r*size*depth_size(index))

        # Palm: convex hull of the (width-scaled) palm landmarks plus thumb webbing,
        # rounded by roughly a knuckle radius so it keeps a body when seen edge-on.
        centre = QPointF((joints[5].x()+joints[17].x())*0.5, (joints[5].y()+joints[17].y())*0.5)
        ax, ay = joints[5].x()-joints[17].x(), joints[5].y()-joints[17].y()
        al = max(1e-6, math.hypot(ax, ay))
        ax, ay = ax/al, ay/al

        def widened(i):
            pt = joints[i]
            lateral = (pt.x()-centre.x())*ax + (pt.y()-centre.y())*ay
            return QPointF(pt.x()+lateral*ax*(palm_width_scale-1.0), pt.y()+lateral*ay*(palm_width_scale-1.0))

        # The wrist end is as wide as a real wrist rather than a single point.
        knuckle_width = math.hypot(widened(5).x()-widened(17).x(), widened(5).y()-widened(17).y())
        vx, vy = joints[9].x()-joints[0].x(), joints[9].y()-joints[0].y()
        vl = max(1e-6, math.hypot(vx, vy))
        px, py = -vy/vl, vx/vl
        half_wrist = 0.40*knuckle_width
        base_pt = QPointF(joints[0].x()+vx*0.04, joints[0].y()+vy*0.04)
        wrist_pts = [QPointF(base_pt.x()+px*half_wrist, base_pt.y()+py*half_wrist),
                     QPointF(base_pt.x()-px*half_wrist, base_pt.y()-py*half_wrist)]
        palm_pts = wrist_pts + [joints[0], joints[1]] + [widened(i) for i in (5, 9, 13, 17)]
        web = QPointF(joints[2].x()*0.45+joints[5].x()*0.4+joints[1].x()*0.15,
                      joints[2].y()*0.45+joints[5].y()*0.4+joints[1].y()*0.15)
        palm_pts.append(web)
        hull = self._convex_hull(palm_pts)
        palm_round = max(1.0, 1.25*radii[9])
        palm_core = QPainterPath()
        if len(hull) >= 3:
            palm_core.addPolygon(QPolygonF(hull))
            palm_core.closeSubpath()
        else:
            palm_core.addEllipse(joints[0], palm_round, palm_round)
        stroker = QPainterPathStroker()
        stroker.setWidth(2.0*palm_round)
        stroker.setJoinStyle(Qt.RoundJoin)
        stroker.setCapStyle(Qt.RoundCap)
        palm_shape = palm_core.united(stroker.createStroke(palm_core)).simplified()

        # Drawable parts: palm, then each finger split at its middle knuckle so a
        # curling finger can pass in front of / behind the palm part-way along.
        # The palm is biased slightly "back" so fingers level with it (an open hand)
        # always draw on top; landmark depth noise then can't make knuckles flicker.
        # Fingers only go behind the palm when clearly farther away (curled while the
        # back of the hand faces the camera).
        palm_z = sum(depth[i] for i in (0, 5, 9, 13, 17))/5.0 + 0.08
        parts = [("palm", None, palm_z)]
        for name, chain, _r in self.HAND_FINGER_CHAINS:
            proximal = [(chain[0], chain[1])]
            distal = [(chain[1], chain[2]), (chain[2], chain[3])]
            groups = (proximal, distal)
            prox_z = (depth[chain[0]]+depth[chain[1]])*0.5
            dist_z = (depth[chain[1]]+depth[chain[2]]+depth[chain[3]])/3.0
            if abs(dist_z-prox_z) < 0.18:
                # Not strongly curled: draw the finger as one smooth piece (no crease).
                groups = (proximal+distal,)
            for segs in groups:
                idx = sorted({i for seg in segs for i in seg})
                parts.append((name, segs, sum(depth[i] for i in idx)/len(idx)))
        parts.sort(key=lambda part: part[2], reverse=True)  # far first

        # Offscreen layer covering the hand, its outline and its shadow.
        reach = max(radii.values()) + palm_round + 12.0*max(0.5, width_scale)
        xs = [pt.x() for pt in joints.values()]
        ys = [pt.y() for pt in joints.values()]
        left, top = math.floor(min(xs)-reach), math.floor(min(ys)-reach)
        w_px, h_px = math.ceil(max(xs)+reach)-left, math.ceil(max(ys)+reach)-top
        if w_px <= 0 or h_px <= 0 or w_px*h_px > 16_000_000:
            return
        try:
            dpr = float(painter.device().devicePixelRatioF())
        except Exception:
            dpr = 1.0
        layer = QImage(max(1, int(w_px*dpr)), max(1, int(h_px*dpr)), QImage.Format_ARGB32_Premultiplied)
        layer.setDevicePixelRatio(dpr)
        layer.fill(Qt.transparent)
        lp = QPainter(layer)
        try:
            lp.setRenderHint(QPainter.Antialiasing, True)
            lp.translate(-left, -top)
            rim_w = max(0.8, 1.15*max(0.5, width_scale))
            rim = self._shade_colour(base, 0.52, 215)
            light_x, light_y = self.HAND_LIGHT

            # Soft contact shadow under the whole hand.
            silhouette = QPainterPath(palm_shape)
            silhouette.setFillRule(Qt.WindingFill)
            for _name, chain, _r in self.HAND_FINGER_CHAINS:
                for a, b in zip(chain, chain[1:]):
                    silhouette.addPath(self._tapered_capsule(joints[a], radii[a], joints[b], radii[b]))
            lp.setPen(Qt.NoPen)
            o = 4.5*max(0.5, width_scale)
            lp.setBrush(QColor(0, 0, 0, 46))
            lp.drawPath(silhouette.translated(o*0.55, o))

            for name, segs, _z in parts:
                if segs is None:
                    # Palm: rim, then a soft dome-like radial shade lit from the upper left.
                    bounds = palm_shape.boundingRect()
                    extent = max(4.0, 0.5*math.hypot(bounds.width(), bounds.height()))
                    tone = depth_tone(0)*0.5 + depth_tone(9)*0.5
                    lp.setPen(QPen(rim, 2.0*rim_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    lp.setBrush(Qt.NoBrush)
                    lp.drawPath(palm_shape)
                    centre_pt = bounds.center()
                    shade = QRadialGradient(QPointF(centre_pt.x()+light_x*extent*0.35,
                                                    centre_pt.y()+light_y*extent*0.35), extent*1.15)
                    shade.setColorAt(0.0, self._shade_colour(base, 1.11*tone))
                    shade.setColorAt(0.6, self._shade_colour(base, 1.0*tone))
                    shade.setColorAt(1.0, self._shade_colour(base, 0.83*tone))
                    lp.setPen(Qt.NoPen)
                    lp.setBrush(QBrush(shade))
                    lp.drawPath(palm_shape)
                    # Fingers curled away behind the hand (the back of a fist) still show
                    # their knuckles as soft bumps along the top of the hand.
                    for finger_name, chain, _r in self.HAND_FINGER_CHAINS:
                        if finger_name == "thumb" or (depth[chain[0]]+depth[chain[1]])*0.5 <= palm_z:
                            continue
                        knuckle = joints[chain[0]]
                        kr = radii[chain[0]]*0.95
                        tone_k = depth_tone(chain[0])
                        # Soft shadow on the side away from the light, then the lit bump.
                        under = QPointF(knuckle.x()-light_x*kr*0.45, knuckle.y()-light_y*kr*0.45)
                        dark = QRadialGradient(under, kr*1.2)
                        dark.setColorAt(0.0, self._shade_colour(base, 0.70*tone_k, 150))
                        dark.setColorAt(1.0, self._shade_colour(base, 0.80*tone_k, 0))
                        lp.setBrush(QBrush(dark))
                        lp.drawEllipse(under, kr*1.2, kr*1.2)
                        lit = QPointF(knuckle.x()+light_x*kr*0.25, knuckle.y()+light_y*kr*0.25)
                        bump = QRadialGradient(lit, kr*1.05)
                        bump.setColorAt(0.0, self._shade_colour(base, 1.24*tone_k, 255))
                        bump.setColorAt(0.55, self._shade_colour(base, 1.08*tone_k, 170))
                        bump.setColorAt(1.0, self._shade_colour(base, 1.0*tone_k, 0))
                        lp.setBrush(QBrush(bump))
                        lp.drawEllipse(lit, kr*1.05, kr*1.05)
                    continue
                # Finger part: one outline then one fill for the whole part, so joints
                # inside a part stay seamless while overlaps with other parts keep an edge.
                outline = QPainterPath()
                outline.setFillRule(Qt.WindingFill)
                body = QPainterPath()
                body.setFillRule(Qt.WindingFill)
                for a, b in segs:
                    outline.addPath(self._tapered_capsule(joints[a], radii[a]+rim_w, joints[b], radii[b]+rim_w))
                    body.addPath(self._tapered_capsule(joints[a], radii[a], joints[b], radii[b]))
                lp.setPen(Qt.NoPen)
                lp.setBrush(rim)
                lp.drawPath(outline)
                first, last = segs[0][0], segs[-1][1]
                p1, p2 = joints[first], joints[last]
                dx, dy = p2.x()-p1.x(), p2.y()-p1.y()
                length = math.hypot(dx, dy)
                nx, ny = (-dy/length, dx/length) if length > 1e-3 else (light_x, light_y)
                if nx*light_x + ny*light_y < 0:  # point the normal toward the light
                    nx, ny = -nx, -ny
                mid = QPointF((p1.x()+p2.x())*0.5, (p1.y()+p2.y())*0.5)
                joints_in_part = sorted({i for seg in segs for i in seg})
                r = sum(radii[i] for i in joints_in_part)/len(joints_in_part)
                tone = sum(depth_tone(i) for i in joints_in_part)/len(joints_in_part)
                # Cylinder shading across the finger: lit edge, highlight, core, shadow.
                shade = QLinearGradient(QPointF(mid.x()+nx*r, mid.y()+ny*r),
                                        QPointF(mid.x()-nx*r, mid.y()-ny*r))
                shade.setColorAt(0.0, self._shade_colour(base, 1.05*tone))
                shade.setColorAt(0.3, self._shade_colour(base, 1.12*tone))
                shade.setColorAt(0.62, self._shade_colour(base, 0.99*tone))
                shade.setColorAt(1.0, self._shade_colour(base, 0.82*tone))
                lp.setBrush(QBrush(shade))
                lp.drawPath(body)
        finally:
            lp.end()
        painter.drawImage(QPointF(left, top), layer)

    def _paint_tli(self, painter, cursor, side, engaged=False):
        """Paint a bright, large TLI; its centre is exactly the interaction point."""
        if cursor is None:
            return
        color = QColor(self.left_click_color if side == "left" else self.right_click_color)
        hsv = color.toHsv()
        hue = hsv.hue()
        if hue < 0:
            hue = 0
        hue_shift = int(round(float(getattr(self, "tli_colour_shift_degrees", 0.0))))
        hsv.setHsv((hue+hue_shift) % 360, hsv.saturation(), hsv.value(), hsv.alpha())
        color = hsv.toRgb()
        glow = (min(2.5, max(0.0, float(getattr(self, "tli_glow_intensity", 0.0))))
                if getattr(self, "tli_glow_enabled", False) else 0.0)
        border = min(8.0, max(1.0, float(getattr(self, "tli_border_thickness", 1.0))))
        diameter = min(96.0, max(12.0, float(getattr(self, "tli_diameter", 30.0))))
        radius = diameter*0.5
        diameter_scale = diameter/54.0
        painter.save()
        painter.setPen(Qt.NoPen)
        # Strong nested halos keep the activation point vivid on bright and dark scenes.
        for halo_radius, alpha in ((41.0, 16), (36.0, 27), (31.0, 44)):
            halo_radius *= diameter_scale
            halo_alpha = min(255, int(round(alpha*glow)))
            if halo_alpha <= 0:
                continue
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), halo_alpha)))
            painter.drawEllipse(cursor, halo_radius, halo_radius)
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 255), border))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 125 if not engaged else 210)))
        painter.drawEllipse(cursor, radius, radius)
        painter.setPen(QPen(QColor(255, 255, 255, 255), max(1.2, 2.2*diameter_scale)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cursor, max(1.0, radius-4.5*diameter_scale), max(1.0, radius-4.5*diameter_scale))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
        painter.drawEllipse(cursor, max(2.0, 4.2*diameter_scale), max(2.0, 4.2*diameter_scale))
        painter.restore()

    def _pinch_menu_alpha(self, side, now=None):
        now = time.monotonic() if now is None else now
        state = self.pinch_menu_animation.get(side, {})
        duration = max(0.001, float(state.get("duration", 0.20)))
        progress = min(1.0, max(0.0, (now-float(state.get("started_at", 0.0)))/duration))
        # Smoothstep avoids a sharp pop on both entry and exit.
        eased = progress*progress*(3.0-2.0*progress)
        return float(state.get("from_alpha", 1.0)) + (
            float(state.get("to_alpha", 1.0))-float(state.get("from_alpha", 1.0)))*eased

    def _set_pinch_menu_open(self, side, visible, now=None, timer_completed=False):
        """Toggle logical menu state and retain a visual snapshot for a short fade."""
        now = time.monotonic() if now is None else now
        visible = bool(visible)
        was_open = bool(self.fist_action_menu_open.get(side, False))
        if visible == was_open:
            return
        old_alpha = min(1.0, max(0.0, self._pinch_menu_alpha(side, now)))
        snapshot = None
        if not visible:
            panel, items = self._action_menu_geometry(side)
            if panel is not None:
                cursor = self.side_cursor_positions.get(side)
                selected = self._action_menu_option_at(side, cursor) if cursor is not None else self.fist_hovered_action.get(side)
                snapshot = {
                    "panel": QRectF(panel),
                    "items": {name: QRectF(rect) for name, rect in items.items()},
                    "selected": selected,
                    "color": QColor(self.left_click_color if side == "left" else self.right_click_color),
                }
        else:
            # Do not synchronously grab/blur the screen here: this runs exactly when the
            # gesture timer completes, and a full-screen capture made the menu visibly lag.
            # Reuse the cached desktop image if present; otherwise _paint_frosted_backdrop
            # falls back to its pre-blurred virtual grid for this immediate first frame.
            if timer_completed:
                cursor = self.side_cursor_positions.get(side)
                if cursor is not None:
                    self.gesture_timer_fade[side] = {
                        "started_at": now, "point": QPointF(cursor),
                        "color": QColor(self.left_click_color if side == "left" else self.right_click_color),
                    }
            snapshot = None
        self.fist_action_menu_open[side] = visible
        self.pinch_menu_animation[side] = {
            "started_at": now,
            "from_alpha": 0.0 if visible else old_alpha,
            "to_alpha": 1.0 if visible else 0.0,
            "duration": 0.20,
            "snapshot": snapshot,
        }
        self.update()

    def _pinch_menu_visual_state(self, side, now=None):
        """Return current or fading menu geometry, alpha, hover and colour."""
        now = time.monotonic() if now is None else now
        alpha = min(1.0, max(0.0, self._pinch_menu_alpha(side, now)))
        if alpha <= 0.005:
            return None
        if self.fist_action_menu_open.get(side, False):
            panel, items = self._action_menu_geometry(side)
            if panel is None:
                return None
            cursor = self.side_cursor_positions.get(side)
            selected = self._action_menu_option_at(side, cursor) if cursor is not None else self.fist_hovered_action.get(side)
            color = QColor(self.left_click_color if side == "left" else self.right_click_color)
        else:
            snapshot = self.pinch_menu_animation.get(side, {}).get("snapshot")
            if snapshot is None:
                return None
            panel, items = snapshot["panel"], snapshot["items"]
            selected, color = snapshot["selected"], QColor(snapshot["color"])
        return panel, items, selected, color, alpha

    def _paint_pinch_menu(self, painter, side, now=None):
        state = self._pinch_menu_visual_state(side, now)
        if state is None:
            return
        panel, items, selected, color, alpha = state
        painter.save()
        painter.setOpacity(painter.opacity()*alpha)
        self._paint_frosted_backdrop(painter, panel, 15, opacity=0.15)
        # Soft halos remain inside the fading layer too.
        for expansion, a, pen_width in ((8.0, 14, 7.5), (3.5, 21, 4.0)):
            painter.setPen(QPen(QColor(130, 225, 245, a), pen_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(panel.adjusted(-expansion, -expansion, expansion, expansion),
                                    15+expansion*0.5, 15+expansion*0.5)
        glass = QLinearGradient(panel.topLeft(), panel.bottomRight())
        glass.setColorAt(0.0, QColor(70, 185, 210, 217))
        glass.setColorAt(1.0, QColor(5, 17, 35, 217))
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 235), 2))
        painter.setBrush(QBrush(glass))
        painter.drawRoundedRect(panel, 15, 15)
        scale = panel.width()/308.0
        labels = self._action_menu_labels()
        for name, rect in items.items():
            hovered = name == selected
            painter.setPen(QPen(QColor(245,255,255,245) if hovered else QColor(150,220,235,205), 1.5))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 165)
                                    if hovered else QColor(17,55,75,218)))
            painter.drawRoundedRect(rect, 9*scale, 9*scale)
            painter.setPen(QColor(248, 253, 255))
            label_text = labels.get(name, name.replace("_", " ").upper())
            self._draw_fitted_button_label(painter, rect, label_text,
                                           base_size=max(9, int(12*scale)), min_size=7, inset=7*scale)
        painter.restore()

    def _paint_gesture_timer_fade(self, painter, side, now=None):
        now = time.monotonic() if now is None else now
        effect = self.gesture_timer_fade.get(side)
        if not effect:
            return
        elapsed = now-float(effect.get("started_at", 0.0))
        duration = 0.22
        alpha = max(0.0, 1.0-elapsed/duration)
        if alpha <= 0.0:
            self.gesture_timer_fade[side] = None
            return
        point, color = effect["point"], effect["color"]
        ring = QRectF(point.x()-25, point.y()-25, 50, 50)
        painter.save()
        painter.setOpacity(painter.opacity()*alpha)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 90), 3))
        painter.drawEllipse(ring)
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 248), 4))
        painter.drawArc(ring, 90*16, -360*16)
        painter.setPen(QColor(245, 251, 255, 245))
        painter.setFont(self._ui_font(8, QFont.DemiBold))
        painter.drawText(QRectF(point.x()-22, point.y()+17, 44, 16), Qt.AlignCenter, "0.0s")
        painter.restore()

    def _smooth_cursor(self, side, target, now):
        """Smooth the already hand-scaled screen point used by the TLI and OS pointer."""
        previous = self.smoothed_side_cursor_positions.get(side)
        last_time = self.cursor_updated_at.get(side, 0.0)
        h = max(1.0, float(self.height()))
        edge_band = h * 0.055
        target_y = min(h-1.0, max(0.0, target.y()))
        # Near the very top/bottom, snap into a tiny edge band to avoid close/minimise drift.
        if target_y <= h*0.008:
            target_y = min(target_y, 2.0)
        elif target_y >= h*0.992:
            target_y = max(target_y, h-3.0)
        target = QPointF(target.x(), target_y)
        if previous is None or now-last_time > 0.35:
            result = QPointF(target)
        else:
            dx, dy = target.x()-previous.x(), target.y()-previous.y()
            movement = math.hypot(dx, dy)
            near_vertical_edge = (target.y() <= edge_band or target.y() >= h-edge_band or
                                  previous.y() <= edge_band or previous.y() >= h-edge_band)
            if near_vertical_edge:
                # A small deadband rejects noise without creating a sluggish edge band.
                if movement < 1.8:
                    result = QPointF(previous)
                else:
                    alpha = 0.96 if movement >= 28.0 else (0.82 if movement >= 10.0 else 0.58)
                    result = QPointF(previous.x()+dx*alpha, previous.y()+dy*alpha)
            else:
                if movement >= 24:
                    alpha = 0.97
                elif movement >= 8:
                    alpha = 0.90
                else:
                    alpha = max(0.68, min(0.92, 1.0-self.cursor_smoothing))
                result = QPointF(previous.x()+dx*alpha, previous.y()+dy*alpha)
        self.smoothed_side_cursor_positions[side] = QPointF(result)
        self.cursor_updated_at[side] = now
        return result

    def _notification_stack_opacity(self, now):
        """Keep notification bars clear while the glass menu is open, then gently fade them."""
        if self.menu_open and self.hub_visible:
            self._notification_stack_full_since = now
            return 1.0
        if self._notification_stack_full_since <= 0.0:
            self._notification_stack_full_since = now
            return 1.0
        elapsed = now - self._notification_stack_full_since - self.notice_fade_delay_seconds
        if elapsed <= 0.0:
            return 1.0
        progress = min(1.0, elapsed/max(0.01, self.notice_fade_duration_seconds))
        progress = progress*progress*(3.0-2.0*progress)
        # The panel is persistent: fade it to a subdued resting opacity, never to zero.
        configured_target = min(1.0, max(0.0, float(self.detached_notice_opacity)/100.0))
        target = max(0.16, configured_target)
        return 1.0 - progress*(1.0-target)

    def _detached_notice_opacity(self, now):
        """Opacity for the persistent detached notice.

        Fully visible while the glass menu is open, and for notice_fade_delay_seconds
        after it closes or after the notification stack changes (notice text changes or
        another notification appears beside it). Then it fades to the slider value.
        Timed notices (e.g. "Hand tracking enabled") are never faded.
        """
        if self.hand_tracking_notice_until != float("inf"):
            return 1.0
        signature = (self.hand_tracking_notice_text, bool(self.hand_tracking_paused), bool(self.stt_enabled))
        if signature != self._notice_signature:
            self._notice_signature = signature
            self._notice_full_since = now
        if self.menu_open and self.hub_visible:
            self._notice_full_since = now
        elapsed = now - self._notice_full_since - self.notice_fade_delay_seconds
        if elapsed <= 0.0:
            return 1.0
        progress = min(1.0, elapsed/max(0.01, self.notice_fade_duration_seconds))
        progress = progress*progress*(3.0-2.0*progress)
        target = min(1.0, max(0.0, float(self.detached_notice_opacity)/100.0))
        return 1.0 - progress*(1.0-target)

    def _refresh_hand_tracking_notice(self, now=None, show_enabled=False):
        """Keep a disabled-hand notice persistent; show a timed enabled message otherwise."""
        now = time.monotonic() if now is None else now
        disabled = [side.title() for side in ("left", "right")
                    if self.mouse_detached_by_side.get(side, False)]
        if disabled:
            label = " & ".join(disabled)
            self.hand_tracking_notice_text = f"{label} mouse detached"
            self.hand_tracking_notice_until = float("inf")
        elif show_enabled:
            self.hand_tracking_notice_text = "Hand tracking enabled"
            self.hand_tracking_notice_until = now + 5.0
        else:
            self.hand_tracking_notice_text = ""
            self.hand_tracking_notice_until = 0.0

    @staticmethod
    def _is_peace_sign(landmarks):
        """Mouse-detach toggle: index/middle raised, other fingers folded, thumb tucked."""
        try:
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_folded = landmarks[16].y > landmarks[14].y
            pinky_folded = landmarks[20].y > landmarks[18].y
            separation = abs(float(landmarks[8].x)-float(landmarks[12].x))
            # Thumb tip must stay near the palm/index base rather than sticking outward.
            thumb_tucked = math.hypot(float(landmarks[4].x)-float(landmarks[5].x),
                                      float(landmarks[4].y)-float(landmarks[5].y)) < 0.20
            return (index_up and middle_up and ring_folded and pinky_folded and
                    separation > 0.025 and thumb_tucked)
        except (IndexError, AttributeError, TypeError):
            return False

    @staticmethod
    def _is_thumbs_up(landmarks):
        """Recognize a clear thumbs-up: thumb extended upward, four fingers folded."""
        try:
            thumb_up = (float(landmarks[4].y) < float(landmarks[3].y) - 0.035 and
                        float(landmarks[4].y) < float(landmarks[5].y) - 0.025)
            fingers_folded = all(float(landmarks[tip].y) > float(landmarks[pip].y) - 0.005
                                 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
            thumb_extended = math.hypot(float(landmarks[4].x)-float(landmarks[2].x),
                                        float(landmarks[4].y)-float(landmarks[2].y)) > 0.08
            return thumb_up and fingers_folded and thumb_extended
        except (IndexError, AttributeError, TypeError, ValueError):
            return False

    def _update_left_click_modifier(self, poses, now):
        """Debounce the LEFT-fist modifier while allowing ambiguous camera frames briefly."""
        left_pose = poses.get("left")
        if left_pose is None:
            self.left_modifier_fist_frames = 0
            self.left_modifier_open_frames = 0
            if (self.left_modifier_last_fist_at is None or
                    now-self.left_modifier_last_fist_at > 0.30):
                self.left_click_modifier_active = False
            return self.left_click_modifier_active

        if left_pose.get("fist", False):
            self.left_modifier_fist_frames = min(3, self.left_modifier_fist_frames+1)
            self.left_modifier_open_frames = 0
            self.left_modifier_last_fist_at = now
            self.left_modifier_release_candidate_since = None
            if self.left_modifier_fist_frames >= 2:
                self.left_click_modifier_active = True
        elif left_pose.get("pinch", False) and left_pose.get("clear_left_pinch", False):
            # A real left pinch is a normal click gesture, not the fist modifier.
            self.left_click_modifier_active = False
            self.left_modifier_fist_frames = 0
            self.left_modifier_open_frames = 0
            self.left_modifier_last_fist_at = None
            self.left_modifier_release_candidate_since = None
        elif left_pose.get("open", False) and not left_pose.get("pinch", False):
            self.left_modifier_fist_frames = 0
            self.left_modifier_open_frames = min(3, self.left_modifier_open_frames+1)
            if self.left_modifier_release_candidate_since is None:
                self.left_modifier_release_candidate_since = now
            if self.left_modifier_open_frames >= 2 and now-self.left_modifier_release_candidate_since >= 0.04:
                self.left_click_modifier_active = False
                self.left_modifier_last_fist_at = None
                self.left_modifier_release_candidate_since = None
        else:
            self.left_modifier_fist_frames = 0
            self.left_modifier_open_frames = 0
            if (self.left_click_modifier_active and self.left_modifier_last_fist_at is not None and
                    now-self.left_modifier_last_fist_at > 0.24):
                self.left_click_modifier_active = False
                self.left_modifier_release_candidate_since = None
        return self.left_click_modifier_active

    def _clear_modifier_click(self, side):
        self.modifier_click_started_at[side] = None
        self.modifier_click_origin[side] = None
        self.modifier_click_kind[side] = None
        self.modifier_click_release_since[side] = None
        self.modifier_click_last_pose_at[side] = None
        self.modifier_click_armed[side] = True

    def _menu_pinch_elapsed_seconds(self, side, now=None):
        """Count only held pinch time after the continuous pinch confirmation buffer."""
        if now is None:
            now = time.monotonic()
        return pinch_timer_elapsed_seconds(
            self.menu_pinch_started.get(side),
            self.menu_pinch_accumulated_seconds.get(side, 0.0),
            self.menu_pinch_pulse_started_at.get(side),
            self.menu_pinch_release_candidate_since.get(side),
            now,
            self.pinch_buffer_seconds,
        )

    def _start_menu_pinch_sequence(self, side, cursor, now):
        """Start a fresh, unfilled timer sequence at this pinch-down event."""
        self.menu_pinch_started[side] = float(now)
        self.menu_pinch_accumulated_seconds[side] = 0.0
        self.menu_pinch_pulse_started_at[side] = float(now)
        self.menu_pinch_origin_points[side] = QPointF(cursor)
        self.pinch_drag_origin_points[side] = QPointF(cursor)
        self.pinch_drag_last_points[side] = QPointF(cursor)
        self.menu_pinch_release_candidate_since[side] = None
        self.menu_pinch_click_fired[side] = False
        self.menu_pinch_latched.discard(side)

    def _clear_menu_pinch_sequence(self, side):
        self.menu_pinch_started[side] = None
        self.menu_pinch_accumulated_seconds[side] = 0.0
        self.menu_pinch_pulse_started_at[side] = None
        self.menu_pinch_origin_points[side] = None
        self.menu_pinch_release_candidate_since[side] = None
        self.menu_pinch_click_fired[side] = False
        self.menu_pinch_latched.discard(side)

    def _suppress_hand_for_modifier(self, side):
        """Cancel normal hold/menu paths for a hand temporarily dedicated to modifier clicking."""
        self._end_pinch_drag(side, status=False)
        self._reset_fist_gesture(side)
        self.active_action[side] = None
        self.action_stop_wait_open[side] = False
        self.action_stop_open_since[side] = None
        self._clear_menu_pinch_sequence(side)
        self.pin_menu_ready_for_selection[side] = False
        self.pin_menu_selection_started_at[side] = None
        self.pin_menu_selection_kind[side] = None
        self.pin_menu_selection_option[side] = None
        self.active_push_sides.discard(side)
        if self.gesture_owner == side:
            self.gesture_owner = None

    def _activate_modifier_right_click(self, point, now):
        """Right-click natively in OS mode or open the virtual 3D context menu."""
        if point is None:
            return
        point = QPointF(point)
        if self.desktop_mode != "os":
            anchor = QPointF(min(max(18.0, point.x()), self.width()-18.0),
                             min(max(18.0, point.y()), self.height()-18.0))
            if self.fist_action_menu_open.get("right", False):
                current_panel, _items = self._action_menu_geometry("right")
                if current_panel is not None and current_panel.contains(anchor):
                    self._set_pinch_menu_open("right", False, now)
            self.fist_origin_points["right"] = QPointF(anchor)
            self.fist_arm_points["right"] = QPointF(anchor)
            self._set_pinch_menu_open("right", True, now)
            self.fist_hovered_action["right"] = None
            self.pin_menu_ready_for_selection["right"] = True
            self.pin_menu_selection_started_at["right"] = None
            self.pin_menu_selection_kind["right"] = None
            self.pin_menu_selection_option["right"] = None
            self.action_stop_wait_open["right"] = True
            self.action_stop_open_since["right"] = None
            self.gesture_owner = "right"
            self.canvas_active_side = "right"
            self.status = "3D context menu opened at the TLI — hover an option and pinch or grip to select"
            self.update()
            return
        self._activate_canvas_right_click(point, "right", now)
        self.status = "Modifier right-click sent at the starting TLI"

    def _start_modifier_window_drag(self, origin, cursor, now):
        """Start a native window drag in OS mode or a canvas/app drag in 3D mode."""
        if self._gesture_ui_modal_open():
            self.status = "Close the glass menu or settings before dragging OS windows"
            return False
        origin = QPointF(origin if origin is not None else cursor)
        for popup_side in ("right", "left"):
            if self.fist_action_menu_open.get(popup_side, False):
                popup, _items = self._action_menu_geometry(popup_side)
                if popup is not None and popup.contains(origin):
                    self.modifier_click_armed["right"] = False
                    self.status = "Close the Pinch Menu before starting a window drag"
                    return False

        if self.desktop_mode != "os":
            # In virtual/3D mode, use the existing canvas-drag behaviour. It is the
            # movable base for future virtual application surfaces as well as the menu.
            self._reset_fist_gesture("right")
            self.fist_started_at["right"] = now
            self.fist_origin_points["right"] = QPointF(origin)
            self.fist_target_window["right"] = None
            self.fist_target_window_resolved["right"] = False
            self.fist_started_on_taskbar["right"] = False
            self.fist_start_hit_test["right"] = None
            self.fist_start_controls["right"] = None
            self._arm_fist_hold_action("right", cursor, now, automatic_motion=False,
                                       force_mode="drag_window")
            if self.fist_hold_armed.get("right") and self.fist_action.get("right") == "canvas_drag":
                self.fist_last_points["right"] = QPointF(cursor)
                self.gesture_owner = "right"
                self.canvas_active_side = "right"
                self.active_push_sides.add("right")
                self.modifier_click_armed["right"] = False
                self.status = "3D canvas/app drag active — move the right hand; open it to release"
                return True
            self.modifier_click_armed["right"] = False
            return False

        if self._canvas_menu_panel_contains(origin):
            self.modifier_click_armed["right"] = False
            self.status = "Move the TLI onto a desktop window before using grip-drag"
            return False

        hwnd = self._find_native_window_at(origin)
        rect = self._native_window_rect(hwnd) if hwnd else None
        if not hwnd or rect is None:
            self.modifier_click_armed["right"] = False
            self.status = "No movable window under the right-hand TLI"
            return False

        self._reset_fist_gesture("right")
        self.fist_origin_points["right"] = QPointF(origin)
        self.fist_target_window["right"] = hwnd
        self.fist_target_window_resolved["right"] = True
        self.fist_started_on_taskbar["right"] = self._is_native_taskbar_point(origin)
        self.fist_start_hit_test["right"] = self._native_nonclient_hit_test(hwnd, origin)
        self.fist_start_controls["right"] = None
        self._arm_fist_hold_action("right", cursor, now, automatic_motion=False,
                                   force_mode="drag_window")
        active = self.active_action.get("right")
        if active is not None and active.get("mode") == "drag_window":
            active["modifier_drag"] = True
            self.gesture_owner = "right"
            self.canvas_active_side = "right"
            self.active_push_sides.add("right")
            self.modifier_click_armed["right"] = False
            self.status = "Window drag active — move the right hand; open it to release"
            return True
        self.modifier_click_armed["right"] = False
        return False

    def _handle_modifier_click_frame(self, side, is_pinch, is_fist, is_open, cursor, now):
        """Left fist modifies a right pinch into right-click and a right grip into drag."""
        if side == "left":
            self._suppress_hand_for_modifier("left")
            self._clear_modifier_click("left")
            return True
        if side != "right":
            return False

        # Once grip-drag is active, let the ordinary manipulation path move the window
        # and handle the open-hand release. Do not let the modifier cleanup cancel it.
        active = self.active_action.get("right")
        if active is not None and active.get("mode") == "drag_window" and active.get("modifier_drag"):
            return False
        if (self.desktop_mode != "os" and self.left_click_modifier_active and
                self.fist_hold_armed.get("right", False) and
                self.fist_action.get("right") == "canvas_drag"):
            # The regular fist-action loop applies deltas to the virtual canvas/app surface.
            return False

        # A modifier gesture never advances or resumes the regular Pinch Menu timer.
        self._clear_menu_pinch_sequence("right")
        self._set_pinch_menu_open("right", False, now)
        self.pin_menu_ready_for_selection["right"] = False
        self.pin_menu_selection_started_at["right"] = None
        self.pin_menu_selection_kind["right"] = None
        self.pin_menu_selection_option["right"] = None
        self.active_action["right"] = None
        self.action_stop_wait_open["right"] = False
        self.action_stop_open_since["right"] = None
        self._reset_fist_gesture("right")

        if is_pinch:
            if (self.modifier_click_started_at[side] is None or
                    self.modifier_click_kind[side] != "pinch_right_click"):
                self._clear_modifier_click(side)
                self.modifier_click_started_at[side] = now
                self.modifier_click_origin[side] = QPointF(cursor)
                self.modifier_click_kind[side] = "pinch_right_click"
                self.modifier_click_armed[side] = True
            self.modifier_click_release_since[side] = None
            self.modifier_click_last_pose_at[side] = now
            started = self.modifier_click_started_at[side]
            if started is not None and now-started > self.fist_quick_click_max_seconds:
                self.modifier_click_armed[side] = False
            self.gesture_owner = side
            self.canvas_active_side = side
            self.active_push_sides.add(side)
            return True

        if is_fist:
            if (self.modifier_click_started_at[side] is None or
                    self.modifier_click_kind[side] != "grip_drag"):
                self._clear_modifier_click(side)
                self.modifier_click_started_at[side] = now
                self.modifier_click_origin[side] = QPointF(cursor)
                self.modifier_click_kind[side] = "grip_drag"
                self.modifier_click_armed[side] = True
            self.modifier_click_release_since[side] = None
            self.modifier_click_last_pose_at[side] = now
            self.gesture_owner = side
            self.canvas_active_side = side
            self.active_push_sides.add(side)
            started = self.modifier_click_started_at[side]
            # Two consecutive tracked frames are already required by the pose detector;
            # 100ms more filters accidental grips without bringing back a long hold timer.
            if (self.modifier_click_armed[side] and started is not None and
                    now-started >= 0.10):
                self._start_modifier_window_drag(
                    self.modifier_click_origin[side] if self.modifier_click_origin[side] is not None else cursor,
                    cursor, now)
                active = self.active_action.get("right")
                if active is not None and active.get("mode") == "drag_window":
                    return False  # allow native window motion to run this same frame
            return True

        started = self.modifier_click_started_at[side]
        kind = self.modifier_click_kind[side]
        if started is None:
            if self.gesture_owner == side:
                self.gesture_owner = None
            return True

        if is_open:
            if self.modifier_click_release_since[side] is None:
                self.modifier_click_release_since[side] = now
            release_for = now-self.modifier_click_release_since[side]
            if release_for >= 0.07:
                elapsed = self.modifier_click_release_since[side]-started
                origin = self.modifier_click_origin[side]
                if (kind == "pinch_right_click" and self.modifier_click_armed[side] and
                        self.click_debounce_seconds <= elapsed <= self.fist_quick_click_max_seconds):
                    self._activate_modifier_right_click(origin if origin is not None else cursor, now)
                self._clear_modifier_click(side)
                if self.gesture_owner == side:
                    self.gesture_owner = None
            return True

        # Ambiguous frames are tolerated briefly, but cannot turn a long held pinch into a click.
        last_pose = self.modifier_click_last_pose_at[side]
        if last_pose is None:
            self.modifier_click_last_pose_at[side] = now
        elif now-last_pose > 0.16:
            self._clear_modifier_click(side)
            if self.gesture_owner == side:
                self.gesture_owner = None
        return True

    def _maintain_native_topmost(self, now=None):
        """Keep the visual hands above fullscreen launchers without taking focus.

        The AlternixOS osm-launcher is a fullscreen Qt window marked
        WindowStaysOnTopHint. On XLibre/X11 its layer can cover the hand overlay,
        even when the main overlay bypasses the window manager. Reassert the native
        overlay's stacking order periodically, without activating it or stealing
        focus from the launcher or the application under the pointer.
        """
        if getattr(self, "_capturing_glass_backdrop", False):
            return
        now = time.monotonic() if now is None else float(now)
        if sys.platform.startswith("linux") and now-float(
                getattr(self, "_last_linux_geometry_sync_check", 0.0)) >= 0.5:
            self._last_linux_geometry_sync_check = now
            self._sync_linux_fullscreen_geometry()
        if now-self._last_native_topmost_reassert <= 0.06:
            return

        if sys.platform.startswith("linux"):
            # XRaiseWindow must only receive native X11 window IDs. On a Wayland Qt
            # backend, winId() is not an X11 window and should be left to Qt/the WM.
            try:
                if QApplication.platformName().lower() not in ("xcb", "x11"):
                    return
            except Exception:
                return
            # Keep this active even when no window drag is in progress: full-screen
            # launchers such as AlternixOS osm-launcher can cover the hands as soon
            # as they open, and Qtile may not notify us about that stacking change.
            # Preserve the app's normal layer ownership when one of its own dialogs is
            # visible: raise the dedicated hand overlay or calibration surface instead
            # of putting the full-screen owner over its own launcher/dialog.
            search_dialog = getattr(self, "search_dialog", None)
            search_overlay = getattr(self, "search_hand_overlay", None)
            if search_dialog is not None and search_dialog.isVisible():
                if search_overlay is None or not search_overlay.isVisible():
                    return
                target = search_overlay
            elif (getattr(self, "tli_calibration_window", None) is not None and
                  self.tli_calibration_window.isVisible()):
                target = self.tli_calibration_window
            elif (getattr(self, "hand_calibration_window", None) is not None and
                  self.hand_calibration_window.isVisible()):
                target = self.hand_calibration_window
            else:
                target = self
            if not target.isVisible():
                return
            try:
                # XRaiseWindow changes only stacking order; it doesn't activate the
                # target or steal keyboard focus from osm-launcher or the dragged app.
                if getattr(self, "_linux_x11_display", None) is None:
                    x11 = ctypes.CDLL("libX11.so.6")
                    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
                    x11.XOpenDisplay.restype = ctypes.c_void_p
                    x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
                    x11.XRaiseWindow.restype = ctypes.c_int
                    x11.XFlush.argtypes = [ctypes.c_void_p]
                    x11.XFlush.restype = ctypes.c_int
                    display = x11.XOpenDisplay(None)
                    if not display:
                        return
                    self._linux_x11_lib = x11
                    self._linux_x11_display = display
                self._linux_x11_lib.XRaiseWindow(self._linux_x11_display,
                                                  ctypes.c_ulong(int(target.winId())))
                self._linux_x11_lib.XFlush(self._linux_x11_display)
                self._last_native_topmost_reassert = now
            except Exception:
                # Qt's topmost hint remains the safe fallback if libX11 is unavailable.
                try:
                    self.raise_()
                    self._last_native_topmost_reassert = now
                except Exception:
                    pass
            return

        if not sys.platform.startswith("win"):
            return
        search_dialog = getattr(self, "search_dialog", None)
        search_overlay = getattr(self, "search_hand_overlay", None)
        if search_dialog is not None and search_dialog.isVisible():
            # SearchHandOverlay has its own faster z-order timer. When paused it may be
            # hidden, but the main fullscreen surface must still stay behind the launcher.
            if search_overlay is None or not search_overlay.isVisible():
                return
            target = search_overlay
        elif (getattr(self, "tli_calibration_window", None) is not None and
              self.tli_calibration_window.isVisible()):
            target = self.tli_calibration_window
        elif (getattr(self, "hand_calibration_window", None) is not None and
              self.hand_calibration_window.isVisible()):
            target = self.hand_calibration_window
        else:
            target = self

        if not target.isVisible():
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(int(target.winId()))
            # HWND_TOPMOST, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE. Reordering without
            # activation keeps the app under keyboard focus while correcting z-order.
            user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0,
                                0x0001 | 0x0002 | 0x0010)
            self._last_native_topmost_reassert = now
        except Exception:
            pass

    def _sync_linux_fullscreen_geometry(self):
        """Keep the XLibre/X11 overlay aligned to the full physical screen rectangle.

        Some X11 window managers initially report the work area for a fullscreen Qt
        widget rather than the full physical screen. Force the widget's local (0, 0) to correspond
        to screen.geometry().topLeft() and make its drawing area match that screen's
        actual dimensions, so screen-local hand coordinates and mapToGlobal agree.
        """
        if not sys.platform.startswith("linux"):
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        screen_rect = screen.geometry()
        try:
            reported_origin = self.mapToGlobal(QPoint(0, 0))
        except Exception:
            reported_origin = screen_rect.topLeft()
        origin_matches = reported_origin == screen_rect.topLeft()
        size_matches = self.size() == screen_rect.size()
        geometry_matches = self.geometry() == screen_rect
        if not (origin_matches and size_matches and geometry_matches):
            # Qtile/XLibre can reapply the work-area offset after startup. Reassert
            # the physical screen bounds so local drawing and full-screen map coords
            # remain aligned; cursor injection itself uses _virtual_point_to_global().
            self.setGeometry(screen_rect)
            self.update()

    # Repaint interval used while nothing on screen is moving (see _scene_is_idle).
    IDLE_REPAINT_INTERVAL = 0.33

    def _scene_is_idle(self, now):
        """True when no hand is in view and nothing is animating.

        Repainting the full-screen translucent overlay is the most expensive part of
        each frame, so idle frames are repainted at a lower rate. Any visible hand,
        gesture, drag, transition or press feedback restores the full frame rate.
        """
        try:
            if any(now-float(seen or 0.0) < 0.6 for seen in self.hand_last_seen.values()):
                return False
            if any(self.active_action.values()) or any(self.pinch_drag_active.values()):
                return False
            if any(self.fist_action_menu_open.values()):
                return False
            for effect in self.gesture_timer_fade.values():
                if effect and now-float(effect.get("started_at", 0.0)) < 0.3:
                    return False
            if now-self.menu_transition_started < self.menu_transition_duration+0.1:
                return False
            if (self.settings_open and self.settings_page_transition_started and
                    self._settings_page_transition_progress(now) < 1.0):
                return False
            if self.settings_page_pending is not None:
                return False
            if now < float(getattr(self, "heard_popup_shake_until", 0.0)) or now < self.click_flash_until:
                return False
            if now-float(getattr(self, "_glass_pressed_at", 0.0)) < 0.35:
                return False
            for animation in self.pinch_menu_animation.values():
                if now-float(animation.get("started_at", 0.0)) < float(animation.get("duration", 0.2))+0.05:
                    return False
            for marker in self.fist_scroll_markers.values():
                if marker and now <= float(marker.get("until", 0.0)):
                    return False
            if (self.hand_calibration_active or self.tli_calibration_active_side is not None or
                    self.custom_gesture_recording is not None or self.active_slider is not None):
                return False
        except Exception:
            return False
        return True

    def _request_scene_repaint(self, now):
        if (self._scene_is_idle(now) and
                now-getattr(self, "_last_scene_repaint", 0.0) < self.IDLE_REPAINT_INTERVAL):
            return
        self._last_scene_repaint = now
        self.update()

    def _adjust_scene_timer_cadence(self, hand_present, now):
        """Run gesture/UI work at 20 Hz while active and 10 Hz while idle."""
        desired_ms = 50 if hand_present or not self._scene_is_idle(now) else 100
        timer = getattr(self, "timer", None)
        if timer is not None and timer.interval() != desired_ms:
            timer.setInterval(desired_ms)

    def update_scene(self):
        """Mutually-exclusive quick clicks, direct fist manipulation, and Pinch Menu gestures."""
        # Track which hand toggled this frame so only that hand's gesture processing pauses.
        gesture_tracking_just_toggled = set()
        self._remember_external_foreground_window()
        now = time.monotonic()
        previous_scene_time = getattr(self, "_last_update_scene_time", now)
        # Bound the per-frame scroll integration delta so a delayed frame cannot
        # generate a large burst of wheel events.
        scroll_dt = max(0.0, min(0.05, now - previous_scene_time))
        self._last_update_scene_time = now
        self._maintain_native_topmost(now)
        if self.hand_tracking_paused:
            # No hand cursor, skeleton, TLI, or gesture timers while paused. Vosk and camera
            # frames remain active so the user can say "resume tracking".
            self.hand_positions = {"left": [], "right": []}
            self.tli_overrides = {}
            self.active_push_sides = set()
            self.pointer_side = None
            menu_state = bool(self.menu_open and self.hub_visible)
            if self._applied_desktop_mode != self.desktop_mode or getattr(self, "_applied_menu_state", None) != menu_state:
                self.apply_desktop_mode()
            self.depth += (self.target_depth-self.depth)*0.16
            self._adjust_scene_timer_cadence(False, now)
            self._request_scene_repaint(now)
            return
        detected_hands, _frame_id = self.tracker.snapshot()
        hands_by_side = {}
        for label, landmarks, score in detected_hands:
            if len(landmarks) < 21 or score < self.hand_recognition_threshold:
                continue
            side = str(label).lower()
            if side not in ("left", "right"):
                continue
            if side not in hands_by_side or score > hands_by_side[side][1]:
                hands_by_side[side] = (landmarks, score)
        self._adjust_scene_timer_cadence(bool(hands_by_side), now)

        # Full-screen-modal onboarding keeps live inference/rendering available but stops
        # all normal pointer, click, drag, scroll, peace-toggle and menu gesture paths.
        if getattr(self, "hand_calibration_active", False):
            self.hand_positions = {
                side: [(float(lm.x), float(lm.y), float(lm.z)) for lm in hands_by_side[side][0]]
                if side in hands_by_side else []
                for side in ("left", "right")
            }
            self.tli_overrides = {}
            self.active_push_sides = set()
            self.pointer_side = None
            self._collect_hand_calibration_frame(hands_by_side)
            self.depth += (self.target_depth-self.depth)*0.16
            self.update()
            return

        menu_state = bool(self.menu_open and self.hub_visible)
        if self._applied_desktop_mode != self.desktop_mode or getattr(self, "_applied_menu_state", None) != menu_state:
            self.apply_desktop_mode()

        self.hand_positions = {"left": [], "right": []}
        self.tli_overrides = {}
        self.active_push_sides = set()
        self._update_hand_scale_origins(hands_by_side)
        cursor_points = {}
        poses = {}
        calibration_modal_frame = self.tli_calibration_active_side is not None
        active_calibration_side = self.tli_calibration_active_side
        if (active_calibration_side in ("left", "right") and
                active_calibration_side not in hands_by_side and
                not self.tli_calibration_wait_release.get(active_calibration_side, False)):
            self.tli_calibration_samples[active_calibration_side] = []
            window = self.tli_calibration_window
            if window is not None:
                window.set_phase("Hand not detected — show the hand and pinch inside the area", 0)
                window.set_hand_overlay([], [])

        # Pointer and TLI share one transformed anchor: the calibrated palm-relative point
        # where available, otherwise the live thumb/index fingertip midpoint.
        for side, (landmarks, _score) in hands_by_side.items():
            self.hand_positions[side] = [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks]
            was_missing = self.hand_missing_since.get(side)
            self.hand_last_seen[side] = now
            mapped_anchor = self._scale_hand_tli_point(
                self.interactive_hand_anchor(landmarks, side), self.hand_positions[side], side)
            cursor = self._smooth_cursor(side, mapped_anchor, now)
            # The TLI is smoothed; draw the hand with the same offset so it moves with
            # the TLI and the TLI stays on the same spot between thumb and index finger.
            dx, dy = cursor.x()-mapped_anchor.x(), cursor.y()-mapped_anchor.y()
            lag = math.hypot(dx, dy)
            if lag > 90.0:  # a jump (re-acquired hand): don't drag the hand far behind
                dx, dy = dx*90.0/lag, dy*90.0/lag
            self.hand_draw_offset[side] = (dx, dy)
            self.hand_display_positions[side] = (self.hand_positions[side],
                                                 self._hand_tucker.update(side, self.hand_positions[side], now))
            cursor_points[side] = QPointF(cursor)
            # All interaction code and rendering now consume the same scaled TLI.
            self.side_cursor_positions[side] = QPointF(cursor)
            if calibration_modal_frame and side == active_calibration_side:
                self._update_tli_calibration_live_overlay(side, landmarks, cursor)
            if was_missing is not None:
                # Resume from the current point after a dropout; don't jump the window or scroll.
                active = self.active_action.get(side)
                if active is not None:
                    active["last_cursor"] = QPointF(cursor)
                    active["open_since"] = None
                    active["ambiguous_since"] = None
                if self.fist_hold_armed.get(side, False):
                    self.fist_last_points[side] = QPointF(cursor)
                self.hand_missing_since[side] = None

            edge_tolerant = (cursor.x() <= self.width()*0.045 or
                             cursor.x() >= self.width()*0.955 or
                             cursor.y() <= self.height()*0.055 or
                             cursor.y() >= self.height()*0.945)
            # This toggle remains active even while all ordinary gestures are locked.
            thumbs_up = False if calibration_modal_frame else self._is_thumbs_up(landmarks)
            if thumbs_up:
                self.thumbs_up_last_seen[side] = now
                if self.thumbs_up_hold_started[side] is None:
                    self.thumbs_up_hold_started[side] = now
                if (now-self.thumbs_up_hold_started[side] >= self.gesture_tracking_toggle_hold_seconds and
                        not self.thumbs_up_toggle_latched[side]):
                    self.gesture_tracking_enabled[side] = not self.gesture_tracking_enabled.get(side, True)
                    self.thumbs_up_toggle_latched[side] = True
                    gesture_tracking_just_toggled.add(side)
                    state = "enabled" if self.gesture_tracking_enabled[side] else "disabled"
                    self.status = f"{side.title()} hand gesture tracking {state}"
                    self.hand_tracking_notice_text = f"{side.title()} hand gestures {state}"
                    self.hand_tracking_notice_until = now + 5.0
                    # Clear only this hand's pending gestures/actions. Never interrupt the
                    # other hand, which remains fully usable throughout the toggle.
                    self.peace_hold_started[side] = None
                    self.menu_pinch_started[side] = None
                    self.fist_started_at[side] = None
                    self.action_stop_wait_open[side] = True
                    self.action_stop_open_since[side] = None
                    if not self.gesture_tracking_enabled[side]:
                        self._end_pinch_drag(side, status=False)
                        self.active_action[side] = None
                        self.fist_action_menu_open[side] = False
                        self.fist_hovered_action[side] = None
                        if self.gesture_owner == side:
                            self.gesture_owner = None
            else:
                # MediaPipe can briefly lose the exact thumb pose for a frame or two.
                # Keep the hold/latch through a short dropout so re-enabling a disabled
                # hand doesn't require a perfectly uninterrupted classification.
                last_thumb_frame = self.thumbs_up_last_seen.get(side)
                if (last_thumb_frame is None or now-last_thumb_frame > 0.18):
                    self.thumbs_up_hold_started[side] = None
                    self.thumbs_up_last_seen[side] = None
                    self.thumbs_up_toggle_latched[side] = False

            # Gesture tracking and cursor/TLI tracking are one per-hand switch. Keep the
            # camera landmarks in hand_positions so the virtual hand stays visible and the
            # thumbs-up can re-enable the hand, but do not provide a moving cursor while off.
            if not self.gesture_tracking_enabled.get(side, True):
                self.side_cursor_positions[side] = None
                self.smoothed_side_cursor_positions[side] = None
                self.cursor_updated_at[side] = 0.0
                self.tli_overrides.pop(side, None)
                cursor_points.pop(side, None)
                self.active_push_sides.discard(side)
                self._clear_menu_pinch_sequence(side)
                self._reset_fist_gesture(side)
                self.active_action[side] = None
                if self.pinch_drag_active.get(side, False):
                    self._end_pinch_drag(side, status=False)
                continue

            peace = False if (calibration_modal_frame or not self.gesture_tracking_enabled.get(side, True)
                              or side in gesture_tracking_just_toggled) else self._is_peace_sign(landmarks)
            if peace:
                if self.peace_hold_started[side] is None:
                    self.peace_hold_started[side] = now
                if now-self.peace_hold_started[side] >= self.peace_toggle_hold_seconds and not self.peace_toggle_latched[side]:
                    self.mouse_detached_by_side[side] = not self.mouse_detached_by_side.get(side, False)
                    self.peace_toggle_latched[side] = True
                    detached = self.mouse_detached_by_side[side]
                    # Never leave a native mouse button held when this hand is detached.
                    # This prevents click-and-hold dragging from persisting after the
                    # hand has been toggled out of native cursor control.
                    if detached and self.pinch_drag_active.get(side, False):
                        self._end_pinch_drag(side, status=False)
                    # Drag scrolling is disabled while detached, so stop one in progress.
                    active_scroll = self.active_action.get(side)
                    if detached and active_scroll is not None and active_scroll.get("mode") == "scroll_window":
                        self.active_action[side] = None
                        self.action_stop_wait_open[side] = True
                        self.action_stop_open_since[side] = None
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                    state = "detached" if detached else "attached"
                    self.status = f"{side.title()} hand mouse {state}"
                    # Detachment affects only native OS pointer movement. Keep updating
                    # the live TLI and keep this hand fully available for glass/menu gestures.
                    self.cursor_updated_at[side] = now
                    if detached:
                        self._refresh_hand_tracking_notice(now, show_enabled=False)
                    else:
                        self._refresh_hand_tracking_notice(now, show_enabled=True)
            else:
                self.peace_hold_started[side] = None
                self.peace_toggle_latched[side] = False
            raw_fist = is_fist_or_grip_pose(landmarks, edge_tolerant=edge_tolerant)
            raw_pinch = is_thumb_index_menu_pinch(landmarks, edge_tolerant=edge_tolerant)
            raw_open = is_open_hand_pose(landmarks, edge_tolerant=edge_tolerant)
            if not self.gesture_tracking_enabled.get(side, True) or side in gesture_tracking_just_toggled:
                raw_fist = raw_pinch = raw_open = False
            if peace:
                raw_fist = raw_pinch = raw_open = False
            calibration_consumes_gesture = self._process_tli_calibration_frame(
                side, landmarks, raw_pinch)
            if calibration_modal_frame or calibration_consumes_gesture:
                # Calibration pinches must never also click, drag, or start the Pinch Menu timer.
                raw_fist = raw_pinch = raw_open = False
            # Treat the open glass menu/settings as a gesture-modal surface. A hand
            # outside the visible panel may still be tracked/rendered, but its pose
            # cannot start a pinch timer, drag, scroll, or click through to an app.
            search_surface_active = (self.search_dialog is not None and self.search_dialog.isVisible())
            if (self.menu_open and self.hub_visible and not search_surface_active and
                    not self._canvas_menu_panel_contains(cursor)):
                self._clear_menu_pinch_sequence(side)
                self._reset_fist_gesture(side)
                self.active_action[side] = None
                self.pinch_drag_active[side] = False
                self.pinch_drag_origin_points[side] = None
                self.pinch_drag_last_points[side] = None
                self.update()
                continue
            clear_left_pinch = False
            if side == "left" and raw_pinch:
                # A tight thumb/index contact is the left-hand click pose even when the
                # broad fist classifier overlaps it. It also clears a stale fist latch
                # from prior noisy frames; the right-hand fist-scroll priority is untouched.
                palm = max(1e-5, distance3d(landmarks, 0, 9))
                pinch_gap = distance3d(landmarks, 4, 8) / palm
                clear_left_pinch = should_prefer_clear_left_pinch_over_fist(
                    is_pinch=raw_pinch, normalized_gap=pinch_gap)
            if side == "right" and raw_fist:
                # The RIGHT fist/grip always wins if MediaPipe briefly labels the same
                # pose as a pinch. This is intentionally stronger than the left-hand
                # tight-pinch exception below, so fist scrolling/dragging cannot be stolen.
                raw_pinch = False
            elif clear_left_pinch:
                raw_fist = False
                self.fist_release_required_for_pinch[side] = False
                self.fist_pose_confirmation_frames[side] = 0
                self.fist_release_buffer_started_at[side] = None
                if (self.fist_started_at[side] is not None and
                        self.active_action.get(side) is None and
                        not self.fist_action_menu_open.get(side, False)):
                    self._reset_fist_gesture(side)
            elif raw_fist:
                # For other hands/poses, a confirmed fist still suppresses pinch.
                raw_pinch = False
            if raw_fist:
                self.fist_pose_confirmation_frames[side] = min(3, self.fist_pose_confirmation_frames[side] + 1)
                self.pinch_recovery_frames[side] = 0
                self.fist_release_buffer_started_at[side] = None
                if self.fist_pose_confirmation_frames[side] >= 2:
                    self.fist_release_required_for_pinch[side] = True
            else:
                self.fist_pose_confirmation_frames[side] = 0
                if self.fist_release_required_for_pinch[side]:
                    # A confirmed fist owns the gesture until a clearly open, unpinched
                    # hand is seen. Pinch classifications and elapsed time must not steal
                    # an active fist-drag/scroll gesture.
                    if raw_open and not raw_pinch:
                        self.fist_release_required_for_pinch[side] = False
                        self.pinch_recovery_frames[side] = 0
                        self.fist_release_buffer_started_at[side] = None
                    else:
                        self.fist_release_buffer_started_at[side] = None
            effective_fist = (
                raw_fist
                or (self.fist_release_required_for_pinch[side] and not clear_left_pinch)
                or (self.fist_started_at[side] is not None and
                    not (raw_open and not raw_pinch) and not clear_left_pinch)
            )
            effective_pinch = (raw_pinch and not effective_fist and
                               (not self.fist_release_required_for_pinch[side] or clear_left_pinch))
            poses[side] = {
                "pinch": effective_pinch,
                "fist": effective_fist,
                "open": raw_open,
                "clear_left_pinch": clear_left_pinch,
                # Fingers folded with or without the thumb out: a thumbs-up counts.
                "fingers_closed": fingers_closed_pose(landmarks, edge_tolerant),
            }

        self._update_hand_contact_display()

        # A pinch outside the visible Search launcher dismisses it, and is consumed so
        # it cannot activate anything underneath the window.
        search_dialog = getattr(self, "search_dialog", None)
        if search_dialog is not None and search_dialog.isVisible():
            try:
                top_left = self.mapFromGlobal(search_dialog.mapToGlobal(QPoint(0, 0)))
                search_rect = QRectF(top_left.x(), top_left.y(),
                                     search_dialog.width(), search_dialog.height())
                outside_search_pinch = any(
                    pose.get("pinch", False) and cursor_points.get(side) is not None and
                    not search_rect.contains(cursor_points[side])
                    for side, pose in poses.items()
                )
                if outside_search_pinch:
                    search_dialog.close()
                    for side in poses:
                        poses[side]["pinch"] = False
                        self._clear_menu_pinch_sequence(side)
                    self.status = "Search closed — outside pinch"
                    self.update()
            except RuntimeError:
                pass

        if calibration_modal_frame:
            # Do not move the OS pointer or pass poses into any normal gesture handler
            # while the standalone calibration target is open. Sampling continues above.
            self.pointer_side = None
            self.active_push_sides = set()
            self.depth += (self.target_depth-self.depth)*0.16
            self.update()
            return

        self._process_custom_gestures(hands_by_side, cursor_points, poses, now)

        modifier_was_active = self.left_click_modifier_active
        modifier_active = self._update_left_click_modifier(poses, now)
        if modifier_active and not modifier_was_active:
            # Entering modifier mode cancels any half-started menu timer or long action.
            self._suppress_hand_for_modifier("left")
            self._suppress_hand_for_modifier("right")
            self._clear_modifier_click("right")
        elif modifier_was_active and not modifier_active:
            # Do not reinterpret the still-held RIGHT pose as a normal long pinch/hold.
            self._clear_modifier_click("right")
            right_pose = poses.get("right", {})
            if right_pose and not right_pose.get("open", False):
                self.action_stop_wait_open["right"] = True
                self.action_stop_open_since["right"] = None

        # Voice-directed text/image lasso owns the mouse while active. Continue to track
        # and render the TLI, but suppress ordinary gestures until the user says copy/finish/cancel.
        if self.voice_selection_mode is not None:
            if self.cursor_lock_priority in cursor_points:
                self.pointer_side = self.cursor_lock_priority
            elif ("right" if self.cursor_lock_priority == "left" else "left") in cursor_points:
                self.pointer_side = "right" if self.cursor_lock_priority == "left" else "left"
            else:
                self.pointer_side = None
            if self.desktop_mode == "os":
                mouse_side = self.pointer_side
                if mouse_side not in cursor_points or self.mouse_detached_by_side.get(mouse_side, False):
                    alternate = "left" if mouse_side == "right" else "right"
                    mouse_side = alternate if (alternate in cursor_points and
                                               not self.mouse_detached_by_side.get(alternate, False)) else None
                if mouse_side in cursor_points:
                    self._set_os_pointer(cursor_points[mouse_side])
            self.depth += (self.target_depth-self.depth)*0.16
            self._request_scene_repaint(now)
            return

        preferred = self.cursor_lock_priority if self.cursor_lock_priority in ("left", "right") else "right"
        nonpreferred = "left" if preferred == "right" else "right"

        # Brief camera dropouts must not abort an active drag, scroll, Pinch Menu selection,
        # or native right-click context menu. Keep the interaction/cursor lock for 1.5s.
        for side in ("right", "left"):
            # A disabled hand is still detected (for a peace-sign re-enable), so it is
            # not tracking-lost even though it is deliberately absent from cursor_points.
            if side in hands_by_side:
                continue
            if self.hand_missing_since[side] is None:
                self.hand_missing_since[side] = now
            missing_for = now-self.hand_missing_since[side]
            timeout = self.tracking_loss_grace_seconds if self._needs_tracking_loss_grace(side) else 0.16
            if missing_for <= timeout:
                continue
            was_active = (self.fist_started_at[side] is not None or
                          self.menu_pinch_started[side] is not None or
                          self.pinch_drag_active[side] or
                          self.active_action[side] is not None or
                          self.fist_action_menu_open[side])
            self._end_pinch_drag(side, status=False)
            if self.fist_started_at[side] is not None:
                self._reset_fist_gesture(side)
            self._clear_menu_pinch_sequence(side)
            self._clear_modifier_click(side)
            if self.fist_action_menu_open.get(side, False):
                self._set_pinch_menu_open(side, False, now)
            self.fist_action_menu_open[side] = False
            self.fist_hovered_action[side] = None
            self.pin_menu_ready_for_selection[side] = False
            self.pin_menu_selection_started_at[side] = None
            self.pin_menu_selection_kind[side] = None
            self.pin_menu_selection_option[side] = None
            self.active_action[side] = None
            self.action_stop_wait_open[side] = False
            self.action_stop_open_since[side] = None
            self.right_click_context_active[side] = False
            self.hand_missing_since[side] = None
            self.fist_release_required_for_pinch[side] = False
            self.fist_pose_confirmation_frames[side] = 0
            if self.gesture_owner == side:
                self.gesture_owner = None
            self.side_cursor_positions[side] = None
            self.smoothed_side_cursor_positions[side] = None
            self.hand_positions[side] = []
            if was_active:
                self.status = "Gesture cancelled — hand tracking lost"

        # The configured physical hand has first refusal for new gestures. Cancel a
        # pending uncommitted gesture from the other hand if the preferred hand appears.
        if preferred in cursor_points and self.gesture_owner == nonpreferred:
            other_is_committed = (self.active_action.get(nonpreferred) is not None or
                                  self.pinch_drag_active.get(nonpreferred, False) or
                                  self.fist_action_menu_open.get(nonpreferred, False) or
                                  (self.fist_hold_armed.get(nonpreferred, False) and
                                   self.fist_action.get(nonpreferred) in ("slider", "menu_drag", "canvas_drag")))
            if not other_is_committed:
                if self.fist_started_at[nonpreferred] is not None:
                    self._reset_fist_gesture(nonpreferred)
                self._clear_menu_pinch_sequence(nonpreferred)
                self.gesture_owner = None

        gesture_order = (preferred, nonpreferred)
        for side in gesture_order:
            entry = hands_by_side.get(side)
            if entry is None or side not in cursor_points or side not in poses:
                # A hand may have been intentionally suppressed for this frame (for
                # example, because it is outside the open glass panel). Keep rendering
                # independent from gesture dispatch and never index a missing pose.
                continue
            if not self.gesture_tracking_enabled.get(side, True) or side in gesture_tracking_just_toggled:
                # Per-hand thumbs-up lock: keep its camera landmarks/TLI alive, but do not
                # let this hand start or continue ordinary gestures. The other hand proceeds.
                continue
            cursor = cursor_points[side]
            pose = poses[side]
            is_pinch, is_fist, is_open = pose["pinch"], pose["fist"], pose["open"]
            clear_left_pinch = bool(pose.get("clear_left_pinch", False))
            # Fist/grip has priority over a transient pinch classification. Once a fist
            # candidate has started, a single noisy pinch frame continues that same hold
            # instead of resetting it or accidentally opening the Pinch Menu.
            if is_fist:
                is_pinch = False
            elif (self.fist_started_at[side] is not None and not is_open and is_pinch
                  and not clear_left_pinch):
                # Once a fist gesture has begun, a non-open pinch classification is noise;
                # keep the fist action in control until the user clearly opens the hand.
                is_fist, is_pinch = True, False

            # A moving held pinch is a native click-and-hold drag. Consume every frame
            # until release, so the same gesture cannot also fire a quick click or menu.
            if self.pinch_drag_active.get(side, False):
                if self.mouse_detached_by_side.get(side, False):
                    self._end_pinch_drag(side, status=False)
                    continue
                if is_pinch:
                    self.pinch_drag_last_points[side] = QPointF(cursor)
                    self.pinch_drag_release_candidate_since[side] = None
                    continue
                # Require a short stable release to avoid dropping an item on one bad
                # landmark frame. A genuine pinch release still drops promptly.
                # Release immediately on the first clear non-pinch frame. The previous
                # debounce made click-and-drag feel delayed compared with earlier builds.
                self._end_pinch_drag(side)
                continue

            # While LEFT fist is held as a modifier, the LEFT hand stops generating
            # gestures and quick RIGHT pinch/grip releases become reliable left-clicks.
            # This branch precedes gesture-owner arbitration so it works either cursor priority.
            if self.left_click_modifier_active:
                if self._handle_modifier_click_frame(side, is_pinch, is_fist, is_open, cursor, now):
                    continue

            # Direct fist-hold drag/scroll continues while the same hand is closed;
            # opening the hand ends the manipulation and returns to normal clicks.
            active = self.active_action.get(side)
            if active is not None:
                # While a right-fist scroll is active, MediaPipe can briefly label the
                # closed grip as a pinch/ambiguous pose. Preserve fist priority for every
                # non-open frame so scrolling neither pauses nor gets stolen by pinch
                # recognition; only a clearly open hand may release the active scroll.
                if active.get("mode") in ("drag_window", "scroll_window") and not is_open:
                    is_fist, is_pinch = True, False
                self.gesture_owner = side
                mode = active.get("mode")
                other_side = "left" if side == "right" else "right"
                if ((mode == "drag_window" or active.get("two_hand_clutch")) and
                        other_side in poses and not poses[other_side].get("fist", False)):
                    # Two-hand actions release as soon as either tracked hand opens.
                    if active.get("modifier_drag"):
                        self._clear_modifier_click(side)
                    self.active_action[side] = None
                    self.action_stop_wait_open[side] = True
                    self.action_stop_open_since[side] = now
                    self._reset_fist_gesture(side)
                    self.gesture_owner = None
                    self.status = "Window drag released — both hands must stay in fists"
                    continue
                if is_open and not is_pinch and not is_fist:
                    if active.get("open_since") is None:
                        active["open_since"] = now
                    # While waiting for a confirmed release, freeze movement and keep
                    # the anchor current so a false-open frame doesn't cause a jump.
                    active["last_cursor"] = QPointF(cursor)
                    release_delay = 0.14 if active.get("modifier_drag") else self.active_release_cooldown_seconds
                    if now-active["open_since"] >= release_delay:
                        if mode == "right_click":
                            self._activate_canvas_right_click(active.get("origin"), side, now)
                            self.status = "Right-click sent at the mode's starting TLI"
                        else:
                            self.status = f"{mode.replace('_', ' ').title()} finished — normal gestures restored"
                        self.active_action[side] = None
                        if active.get("modifier_drag"):
                            self._clear_modifier_click(side)
                        self.action_stop_wait_open[side] = True
                        self.action_stop_open_since[side] = now
                        self._reset_fist_gesture(side)
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                    continue

                # Briefly ambiguous landmarks should not end an active mode, but only
                # move/scroll when the hand is still classified as a fist.
                if is_fist:
                    active["open_since"] = None
                    active["ambiguous_since"] = None
                    self.active_push_sides.add(side)
                    previous = active.get("last_cursor")
                    if previous is None:
                        previous = cursor
                    dx = cursor.x()-previous.x()
                    dy = cursor.y()-previous.y()
                    if mode == "drag_window":
                        rect = active.get("rect") or self._native_window_rect(active.get("hwnd"))
                        gain = self.hand_tracking_sensitivity
                        move_x, move_y = dx*gain, dy*gain
                        if rect is not None and (abs(move_x) > 0.1 or abs(move_y) > 0.1):
                            left, top, right, bottom = rect
                            if self._move_native_window(active.get("hwnd"), left+move_x, top+move_y):
                                active["rect"] = (left+move_x, top+move_y, right+move_x, bottom+move_y)
                    elif mode == "scroll_window":
                        # Continuous grab-scroll: displacement is measured from the
                        # original fist/grab point. Holding the hand away from that point
                        # keeps scrolling; moving farther away increases the rate. Returning
                        # toward the origin slows/reverses the scroll naturally.
                        origin = active.get("origin", cursor)
                        displacement_x = float(cursor.x() - origin.x())
                        displacement_y = float(cursor.y() - origin.y())
                        displacement_x = max(-420.0, min(420.0, displacement_x))
                        displacement_y = max(-420.0, min(420.0, displacement_y))
                        smoothed_dx = smooth_scroll_delta(
                            active.get("scroll_smoothed_dx", 0.0), displacement_x,
                            previous_weight=0.55, max_delta=420.0,
                        )
                        smoothed_dy = smooth_scroll_delta(
                            active.get("scroll_smoothed_dy", 0.0), displacement_y,
                            previous_weight=0.55, max_delta=420.0,
                        )
                        active["scroll_smoothed_dx"] = smoothed_dx
                        active["scroll_smoothed_dy"] = smoothed_dy
                        deadzone = max(4.0, float(self.scroll_deadzone))
                        notch_spacing = 72.0
                        # Scroll activation uses circular distance from the grab anchor,
                        # so diagonal movement can drive both axes together. The existing
                        # deadzone/notch spacing is preserved: deadzone, +1 notch, +2 notches.
                        radial_distance = math.hypot(smoothed_dx, smoothed_dy)
                        deadzone = max(4.0, float(self.scroll_deadzone))
                        notch_spacing = 72.0
                        if radial_distance <= deadzone:
                            scroll_stage = 0
                        elif radial_distance < deadzone + notch_spacing:
                            scroll_stage = 1
                        elif radial_distance < deadzone + notch_spacing * 2.0:
                            scroll_stage = 2
                        else:
                            scroll_stage = 3
                        previous_stage = int(active.get("scroll_stage", 0))
                        active["scroll_stage"] = scroll_stage
                        if scroll_stage == 0:
                            active["scroll_nav_fired"] = False

                        horizontal_supported = active.get("horizontal_scroll_supported")
                        if horizontal_supported is None:
                            horizontal_supported = self._window_has_horizontal_scrollbar(active, cursor)
                            active["horizontal_scroll_supported"] = horizontal_supported
                        horizontal_supported = bool(horizontal_supported)

                        # Stage one gives a discrete nudge on each axis that has meaningful
                        # displacement; higher stages continuously scroll both components.
                        if scroll_stage == 1 and previous_stage == 0:
                            try:
                                if abs(smoothed_dy) > 0.5:
                                    self._scroll_target_window(active, 120 if smoothed_dy > 0 else -120,
                                                               cursor=cursor)
                                if horizontal_supported and abs(smoothed_dx) > 0.5:
                                    self._scroll_target_window(active, 120 if smoothed_dx > 0 else -120,
                                                               cursor=cursor, horizontal=True)
                            except Exception as exc:
                                self.status = f"Scroll failed: {exc}"

                        if scroll_stage < 2:
                            vertical_rate = horizontal_rate = 0.0
                        else:
                            if scroll_stage == 2:
                                radial_rate = 1.5
                            else:
                                radial_rate = max(1.2, min(24.0, float(self.scroll_acceleration) * 2.4))
                            # Preserve direction and blend by the radial vector, allowing
                            # simultaneous diagonal scrolling without either axis taking over.
                            vertical_rate = radial_rate * (smoothed_dy / radial_distance) if radial_distance else 0.0
                            horizontal_rate = radial_rate * (smoothed_dx / radial_distance) if radial_distance else 0.0
                        vertical_notches_per_second = vertical_rate * self.hand_tracking_sensitivity * 0.8
                        horizontal_notches_per_second = horizontal_rate * self.hand_tracking_sensitivity * 0.8
                        if not horizontal_supported:
                            horizontal_notches_per_second = 0.0
                            if scroll_stage == 3 and previous_stage != 3 and abs(smoothed_dx) > deadzone and not active.get("scroll_nav_fired", False):
                                direction = -1 if smoothed_dx < 0 else 1
                                if self.swap_drag_scroll_notches:
                                    direction = -direction
                                if self._send_browser_navigation(direction):
                                    self.status = "Back button sent" if direction < 0 else "Forward button sent"
                                active["scroll_nav_fired"] = True
                        # Edge auto-scroll: while the same fist remains closed near
                        # the top/bottom edge of its target window, scroll steadily even
                        # when the hand stops moving. The rate ramps gently with proximity.
                        hwnd = active.get("hwnd")
                        rect = self._native_window_rect(hwnd) if hwnd else None
                        if rect is not None:
                            left, top, right, bottom = rect
                            screen_cursor = self._virtual_point_to_global(cursor)
                            edge_band = 58.0
                            edge_rate = 0.0
                            if left <= screen_cursor.x() <= right:
                                if top <= screen_cursor.y() < top + edge_band:
                                    edge_rate = -3.0 * min(1.0, (top + edge_band - screen_cursor.y()) / edge_band)
                                elif bottom - edge_band < screen_cursor.y() <= bottom:
                                    edge_rate = 3.0 * min(1.0, (screen_cursor.y() - (bottom - edge_band)) / edge_band)
                            # Edge auto-scroll adds up to 3 notches/s near the window edge.
                            vertical_notches_per_second += edge_rate
                        # Independent accumulators allow horizontal and vertical wheel
                        # messages in the same frame, preserving diagonal scroll direction.
                        chunk = 40.0 if sys.platform.startswith("win") else 120.0
                        for axis_name, rate, horizontal in (
                                ("scroll_remainder", vertical_notches_per_second, False),
                                ("scroll_horizontal_remainder", horizontal_notches_per_second, True)):
                            active[axis_name] = float(active.get(axis_name, 0.0)) + rate * 120.0 * scroll_dt
                            wheel_units = int(active[axis_name] / chunk) * int(chunk)
                            if wheel_units:
                                try:
                                    self._scroll_target_window(active, wheel_units, cursor=cursor, horizontal=horizontal)
                                    active[axis_name] -= wheel_units
                                except Exception as exc:
                                    self.status = f"Scroll failed: {exc}"
                    active["last_cursor"] = QPointF(cursor)
                elif is_pinch:
                    active["open_since"] = None
                    active["ambiguous_since"] = None
                else:
                    active["open_since"] = None
                    if active.get("ambiguous_since") is None:
                        active["ambiguous_since"] = now
                    # Landmark flicker during a drag/scroll is tracking loss, not a release.
                    if now-active["ambiguous_since"] > self.tracking_loss_grace_seconds:
                        self.status = "Selected mode cancelled — hand pose lost"
                        self.active_action[side] = None
                        self.action_stop_wait_open[side] = True
                        self.action_stop_open_since[side] = None
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                continue

            # After stopping a drag/scroll, hold a neutral open hand briefly so the release
            # cannot be interpreted as the end of a quick fist-open click.
            if self.action_stop_wait_open[side]:
                if is_open:
                    if self.action_stop_open_since[side] is None:
                        self.action_stop_open_since[side] = now
                    elif now-self.action_stop_open_since[side] >= 0.14:
                        self.action_stop_wait_open[side] = False
                        self.action_stop_open_since[side] = None
                else:
                    self.action_stop_open_since[side] = None
                continue

            # Only one hand may own a gesture sequence at a time.
            if self.gesture_owner not in (None, side):
                continue
            if (not self.left_click_modifier_active and side != preferred and preferred in cursor_points and
                    not self.fist_action_menu_open.get(side, False)):
                # Always honour the configured physical-hand priority for new idle gestures.
                # A committed action or menu selection was handled above and remains locked.
                continue

            # Pinch Menu selection: release the opening pinch first, then either grab/fist
            # or make a fresh (short) pinch over an item to activate it.
            if self.fist_action_menu_open[side]:
                self.canvas_active_side = side
                hovered = self._action_menu_option_at(side, cursor)
                self.fist_hovered_action[side] = hovered
                panel, _menu_items = self._action_menu_geometry(side)
                point_inside_popup = panel is not None and panel.contains(cursor)

                # Once the opening pinch has been released, a new pinch/grip outside
                # the whole popup dismisses it. Consume the same pose so it cannot
                # fall through into a click, drag, or a fresh menu timer.
                if now >= self.pinch_menu_dismiss_block_until and should_dismiss_pinch_menu(
                        selection_ready=self.pin_menu_ready_for_selection[side],
                        is_fist=is_fist, is_pinch=is_pinch,
                        point_inside_popup=point_inside_popup):
                    self._dismiss_pinch_menu(side, now)
                    continue

                if not self.pin_menu_ready_for_selection[side]:
                    # Do not let the same long pinch that opened the menu select an item.
                    if is_open and not is_fist and not is_pinch:
                        self.pin_menu_ready_for_selection[side] = True
                        self._clear_menu_pinch_sequence(side)
                        self.status = "Pinch Menu ready — move over an item and grab or pinch to select"
                    self.pin_menu_selection_started_at[side] = None
                    self.pin_menu_selection_kind[side] = None
                    self.pin_menu_selection_option[side] = None
                    continue

                selection_kind = "grab" if is_fist else ("pinch" if is_pinch else None)
                if selection_kind is not None and hovered is not None:
                    if (self.pin_menu_selection_started_at[side] is None or
                            self.pin_menu_selection_kind[side] != selection_kind or
                            self.pin_menu_selection_option[side] != hovered):
                        self.pin_menu_selection_started_at[side] = now
                        self.pin_menu_selection_kind[side] = selection_kind
                        self.pin_menu_selection_option[side] = hovered
                    elif now-self.pin_menu_selection_started_at[side] >= self.pin_menu_selection_debounce_seconds:
                        self._begin_action_from_popup(side, hovered, cursor, now)
                else:
                    self.pin_menu_selection_started_at[side] = None
                    self.pin_menu_selection_kind[side] = None
                    self.pin_menu_selection_option[side] = None
                continue

            # A grab near a Settings slider takes priority over an armed window action:
            # it snaps to the slider knob and starts immediately, without the hold timer.
            slider_grab_candidate = self._canvas_slider_name_at(cursor, for_grab=True) if (is_fist or is_pinch) else None
            if is_pinch and (slider_grab_candidate is not None or
                             (self.fist_hold_armed.get(side, False) and self.fist_action.get(side) == "slider")):
                # Pinch on a slider is promoted to the same snapped/pinned slider drag as fist.
                is_fist, is_pinch = True, False

            if is_pinch:
                # The thumb/index pinch opens the Pinch Menu after a confirmation delay.
                # A quick pinch released before the long-hold threshold is a normal click.
                if is_fist or (self.fist_started_at[side] is not None and
                               self.fist_release_required_for_pinch[side]):
                    # Do not let a single noisy pinch frame flip a fist into a menu timer.
                    is_pinch = False
                else:
                    self.gesture_owner = side
                    self.canvas_active_side = side
                    self.active_push_sides.add(side)
                    if self.menu_pinch_started[side] is None:
                        self._start_menu_pinch_sequence(side, cursor, now)
                    elif self.menu_pinch_release_candidate_since[side] is not None:
                        # A re-pinched pulse gets its own drag origin, not the previous click point.
                        self.pinch_drag_origin_points[side] = QPointF(cursor)
                        self.pinch_drag_last_points[side] = QPointF(cursor)
                        release_started = self.menu_pinch_release_candidate_since[side]
                        if quick_pinch_reentry_eligible(
                                self.menu_pinch_started[side],
                                self.menu_pinch_pulse_started_at[side],
                                release_started, now,
                                self.menu_pinch_click_fired[side],
                                side in self.menu_pinch_latched,
                                self.pinch_buffer_seconds,
                                min_release_gap=0.0,
                                min_pulse_seconds=self.quick_pinch_click_min_seconds,
                                max_pulse_seconds=self.fist_quick_click_max_seconds):
                            # Dispatch the previous short pinch as the first click even
                            # when the next pinch returns before the 80ms release debounce.
                            click_point = self.menu_pinch_origin_points[side] or cursor
                            self._activate_quick_pinch_click(click_point, side, now)
                            if self.tli_calibration_active_side is not None:
                                self.pointer_side = None
                                self.active_push_sides.clear()
                                self.depth += (self.target_depth-self.depth)*0.16
                                self.update()
                                return
                            self.menu_pinch_click_fired[side] = True
                        if pinch_timer_gap_restarts_buffer(
                                self.menu_pinch_started[side], release_started, now,
                                self.pinch_buffer_seconds):
                            # A real pinch release during the 0.8s buffer invalidates that
                            # sequence; a quick re-pinch gets a new full buffer and no credit.
                            self._clear_menu_pinch_sequence(side)
                            self._start_menu_pinch_sequence(side, cursor, now)
                        else:
                            self.menu_pinch_accumulated_seconds[side] += pinch_timer_pulse_seconds(
                                self.menu_pinch_started[side],
                                self.menu_pinch_pulse_started_at[side],
                                release_started,
                                self.pinch_buffer_seconds,
                            )
                            self.menu_pinch_pulse_started_at[side] = now
                            self.menu_pinch_origin_points[side] = QPointF(cursor)
                            self.menu_pinch_click_fired[side] = False
                    self.menu_pinch_release_candidate_since[side] = None
                    held = self._menu_pinch_elapsed_seconds(side, now)
                    # A stationary pinch opens the Pinch Menu when the circle completes.
                    # Movement before that point means pick-up/drag instead: hold the
                    # native left mouse button so standard app drag/drop keeps working.
                    drag_origin = self.pinch_drag_origin_points.get(side)
                    # A drag needs a pinch that has lasted a moment: closing the hand into
                    # a thumbs up or fist sweeps a brief, moving pinch past the thumb, and
                    # that must not press the mouse button.
                    pulse_start = self.menu_pinch_pulse_started_at[side] or self.menu_pinch_started[side]
                    if (side not in self.menu_pinch_latched and
                            held < self.menu_pinch_hold_seconds and
                            not self.left_click_modifier_active and
                            now - pulse_start >= PINCH_DRAG_MIN_HOLD_SECONDS and
                            pinch_drag_should_start(
                                (drag_origin.x(), drag_origin.y()) if drag_origin is not None else None,
                                (cursor.x(), cursor.y()),
                                self.pinch_drag_start_distance)):
                        if self._begin_pinch_drag(side, cursor, now):
                            continue
                    if held >= self.menu_pinch_hold_seconds and side not in self.menu_pinch_latched:
                        self.menu_pinch_latched.add(side)
                        if self.right_click_mode_enabled or self.scroll_mode_enabled:
                            self.right_click_mode_enabled = False
                            self.scroll_mode_enabled = False
                            self.status = "Normal mode restored — release the pinch; pinch again to open the menu"
                        else:
                            self._set_pinch_menu_open(side, True, now, timer_completed=True)
                            self.fist_origin_points[side] = QPointF(cursor)
                            self.fist_arm_points[side] = QPointF(cursor)
                            self.fist_hovered_action[side] = self._action_menu_option_at(side, cursor)
                            self.pin_menu_ready_for_selection[side] = False
                            self.pin_menu_selection_started_at[side] = None
                            self.pin_menu_selection_kind[side] = None
                            self.pin_menu_selection_option[side] = None
                            self.status = "Pinch Menu open — release the pinch, then grab or pinch an item"
                    continue

            if self.menu_pinch_started[side] is not None:
                # Confirm a release before a standalone click. If the next pinch returns
                # before this debounce finishes, the re-entry path handles a valid click pair.
                if self.menu_pinch_release_candidate_since[side] is None:
                    self.menu_pinch_release_candidate_since[side] = now
                release_for = now-self.menu_pinch_release_candidate_since[side]
                if is_fist:
                    # A fist is not a pinch release/click; cancel once it remains a fist briefly.
                    if release_for >= 0.12:
                        self._clear_menu_pinch_sequence(side)
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                    continue
                if release_for >= 0.08:
                    if (not self.menu_pinch_click_fired[side] and side not in self.menu_pinch_latched and
                            pinch_timer_release_before_buffer(
                                self.menu_pinch_started[side],
                                self.menu_pinch_release_candidate_since[side],
                                self.pinch_buffer_seconds)):
                        pulse_start = self.menu_pinch_pulse_started_at[side] or self.menu_pinch_started[side]
                        verdict = "click"
                        if quick_pinch_click_eligible(
                                pulse_start,
                                self.menu_pinch_release_candidate_since[side],
                                self.quick_pinch_click_min_seconds,
                                self.fist_quick_click_max_seconds):
                            # Closing the hand (into a thumbs up, say) sweeps the index
                            # past the thumb: that brief pinch is not a click. Decide once
                            # the released hand shows whether it opened or closed.
                            verdict = pinch_release_click_verdict(
                                release_for, is_open, pose.get("fingers_closed", False))
                            if verdict == "wait":
                                continue
                        if verdict == "click" and quick_pinch_click_eligible(
                                pulse_start,
                                self.menu_pinch_release_candidate_since[side],
                                self.quick_pinch_click_min_seconds,
                                self.fist_quick_click_max_seconds):
                            click_point = self.menu_pinch_origin_points[side] or cursor
                            self._activate_quick_pinch_click(click_point, side, now)
                            if self.tli_calibration_active_side is not None:
                                self.pointer_side = None
                                self.active_push_sides.clear()
                                self.depth += (self.target_depth-self.depth)*0.16
                                self.update()
                                return
                            self.menu_pinch_click_fired[side] = True
                    # A release before confirmation expires cancels the sequence once stable.
                    # Quick pinches cannot pool against one old wall-clock start time.
                    if pinch_timer_release_before_buffer(
                            self.menu_pinch_started[side], self.menu_pinch_release_candidate_since[side],
                            self.pinch_buffer_seconds):
                        self._clear_menu_pinch_sequence(side)
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                        continue
                # After the initial buffer has completed, short release gaps may preserve the
                # fill sequence; only actual down-time is accumulated, never the gap itself.
                if release_for <= self.menu_pinch_sequence_gap_seconds:
                    continue
                self._clear_menu_pinch_sequence(side)
                if self.gesture_owner == side:
                    self.gesture_owner = None
                continue

            if is_fist:
                self.gesture_owner = side
                self.canvas_active_side = side
                self.active_push_sides.add(side)
                if self.fist_started_at[side] is None:
                    self.fist_started_at[side] = now
                    self.fist_release_candidate_since[side] = None
                    self.fist_open_candidate_since[side] = None
                    self.fist_origin_points[side] = QPointF(cursor)
                    self.fist_arm_points[side] = None
                    self.fist_hold_armed[side] = False
                    self.fist_action[side] = None
                    self.fist_last_points[side] = QPointF(cursor)
                    self.fist_start_controls[side] = slider_grab_candidate or self._canvas_slider_name_at(cursor)
                    self.fist_slider_offset_x[side] = 0.0
                    self.fist_scroll_remainder[side] = 0.0
                    self.fist_target_window[side] = None
                    self.fist_target_window_resolved[side] = False
                    self.fist_started_on_taskbar[side] = False
                    self.fist_start_hit_test[side] = None
                    self.fist_shake_samples[side] = []
                    self.fist_hold_ready_notified[side] = False
                    raw_anchor_start = self.raw_hand_anchor(entry[0], side)
                    shake_start = self.interactive_visual_point(*raw_anchor_start)
                    self.fist_shake_samples[side].append((now, shake_start.x(), shake_start.y()))
                    self.fist_action_menu_open[side] = False
                    self.fist_hovered_action[side] = None
                    if self.fist_start_controls[side]:
                        # Snap the TLI to the current knob, preserve its value at pickup,
                        # and begin dragging immediately; the hold timer is bypassed.
                        geometry = self._canvas_slider_geometry(self.fist_start_controls[side])
                        if geometry is not None:
                            _left, _right, _y, knob = geometry
                            self.fist_slider_offset_x[side] = knob.x()-cursor.x()
                            self.fist_arm_points[side] = QPointF(knob)
                            self.fist_hold_armed[side] = True
                            self.fist_action[side] = "slider"
                            self.fist_last_points[side] = QPointF(cursor)
                            self.slider_last_good_cursor[side] = QPointF(cursor)
                            self.slider_cursor_jump_candidate[side] = None
                            self.tli_overrides[side] = ("slider", knob)
                            self.status = "Slider snapped — move hand to adjust; open hand to release"
                # Also capture a slider entered during the initial fist approach. This avoids
                # letting the hold timer fire just because the fist began a few pixels away.
                if (not self.fist_hold_armed[side] and self.fist_start_controls[side] is None
                        and slider_grab_candidate is not None):
                    geometry = self._canvas_slider_geometry(slider_grab_candidate)
                    if geometry is not None:
                        _left, _right, _y, knob = geometry
                        self.fist_start_controls[side] = slider_grab_candidate
                        self.fist_slider_offset_x[side] = knob.x()-cursor.x()
                        self.fist_arm_points[side] = QPointF(knob)
                        self.fist_hold_armed[side] = True
                        self.fist_action[side] = "slider"
                        self.fist_last_points[side] = QPointF(cursor)
                        self.slider_last_good_cursor[side] = QPointF(cursor)
                        self.slider_cursor_jump_candidate[side] = None
                        self.tli_overrides[side] = ("slider", knob)
                        self.status = "Slider snapped — move hand to adjust; open hand to release"
                self.fist_release_candidate_since[side] = None
                self.fist_open_candidate_since[side] = None
                # Measure this specific fist hold for quick-click release and the separate
                # two-hand drag clutch. Right-only scroll itself has no hold-time requirement.
                current_fist_held = now-self.fist_started_at[side]
                origin = self.fist_origin_points[side] or cursor
                dx_from_origin = cursor.x()-origin.x()
                dy_from_origin = cursor.y()-origin.y()

                # Track an unbounded mapped anchor so horizontal shakes remain detectable
                # even if the visible cursor is clamped at the screen edge.
                raw_anchor = self.raw_hand_anchor(entry[0], side)
                shake_point = self.interactive_visual_point(*raw_anchor)
                shake_samples = self.fist_shake_samples[side]
                shake_samples.append((now, shake_point.x(), shake_point.y()))
                while shake_samples and now-shake_samples[0][0] > 1.05:
                    shake_samples.pop(0)

                # Resolve the window only once the confirmation delay proves this isn't
                # a quick click. Avoid Win32 enumeration and hit-testing on every click.
                if (current_fist_held >= self.gesture_timer_delay_seconds and
                        not self.fist_hold_armed[side] and
                        not self.fist_start_controls[side] and
                        not self.fist_target_window_resolved[side]):
                    self.fist_target_window_resolved[side] = True
                    self.fist_started_on_taskbar[side] = (
                        self.desktop_mode == "os" and self._is_native_taskbar_point(origin))
                    if self.fist_started_on_taskbar[side]:
                        self.fist_start_hit_test[side] = -100
                    else:
                        self.fist_target_window[side] = self._find_native_window_at(origin)
                        self.fist_start_hit_test[side] = self._native_nonclient_hit_test(
                            self.fist_target_window[side], origin)

                # A RIGHT-only fist is a movement scroll gesture: no circle timer or
                # 0.8s wait. A deliberate vertical displacement over client/scrollbar
                # content arms the already-smoothed scroll path. The LEFT fist remains
                # reserved as the modifier / two-hand drag clutch.
                right_only_fist = (side == "right" and
                                   not poses.get("left", {}).get("fist", False) and
                                   not self.left_click_modifier_active)
                scroll_candidate = False
                if right_only_fist and not self.fist_hold_armed[side]:
                    proposed = should_start_right_fist_scroll(
                        side=side, is_fist=is_fist,
                        left_fist_active=poses.get("left", {}).get("fist", False),
                        modifier_active=self.left_click_modifier_active,
                        already_armed=self.fist_hold_armed[side],
                        dx=dx_from_origin, dy=dy_from_origin,
                        hit_test=self.fist_start_hit_test.get(side),
                    )
                    if proposed and not self.fist_target_window_resolved[side]:
                        # Resolve the scroll target only once the hand actually moves,
                        # not during ordinary quick fist-open clicks.
                        self.fist_target_window[side] = self._find_native_window_at(origin)
                        self.fist_start_hit_test[side] = self._native_nonclient_hit_test(
                            self.fist_target_window[side], origin)
                        self.fist_target_window_resolved[side] = True
                    if proposed:
                        scroll_candidate = should_start_right_fist_scroll(
                            side=side, is_fist=is_fist,
                            left_fist_active=poses.get("left", {}).get("fist", False),
                            modifier_active=self.left_click_modifier_active,
                            already_armed=self.fist_hold_armed[side],
                            dx=dx_from_origin, dy=dy_from_origin,
                            hit_test=self.fist_start_hit_test.get(side),
                        )
                if scroll_candidate:
                    self._arm_fist_hold_action(side, cursor, now, automatic_motion=True,
                                               force_mode="scroll_window")
                elif (side == "right" and not self.fist_hold_armed[side] and
                      current_fist_held >= self.gesture_timer_delay_seconds and
                      poses.get("left", {}).get("fist", False) and
                      poses.get("right", {}).get("fist", False) and
                      "left" in cursor_points and "right" in cursor_points):
                    # Left fist is the stationary clutch/anchor; right fist is the drag hand.
                    left_start = self.fist_origin_points.get("left") or cursor_points["left"]
                    right_start = self.fist_origin_points.get("right") or cursor
                    left_travel = math.hypot(cursor_points["left"].x()-left_start.x(), cursor_points["left"].y()-left_start.y())
                    right_travel = math.hypot(cursor.x()-right_start.x(), cursor.y()-right_start.y())
                    if left_travel <= 24.0 and right_travel >= 34.0:
                        # With both fists held, the left hand is the stationary clutch
                        # and the right hand is the drag hand. If the right TLI started
                        # on the glass menu, explicitly route this two-fist drag to the
                        # menu panel rather than looking for an OS window underneath it.
                        if (self.menu_open and self.hub_visible and
                                self._canvas_menu_panel_contains(right_start)):
                            self.fist_target_window["right"] = None
                            clutch_mode = "scroll_window" if self.scroll_mode_enabled else "drag_window"
                            self._arm_fist_hold_action(
                                "right", cursor, now, automatic_motion=False,
                                force_mode=clutch_mode)
                            active_clutch = self.active_action.get("right")
                            if active_clutch is not None:
                                active_clutch["two_hand_clutch"] = True
                            if clutch_mode == "scroll_window":
                                self.status = "Two-fist content scroll active — move the right fist; open either hand to stop"
                            else:
                                self.status = "Glass menu drag active — move the right fist; open to release"
                        else:
                            self.fist_target_window["right"] = self._find_native_window_at(right_start)
                            # If the topmost hit is a screen-sized host surface, prefer
                            # the smallest visible application window containing the TLI.
                            self.fist_target_window["right"] = self._find_native_window_at(
                                right_start, prefer_smallest=True) or self.fist_target_window["right"]
                            clutch_mode = "scroll_window" if self.scroll_mode_enabled else "drag_window"
                            self._arm_fist_hold_action("right", cursor, now, automatic_motion=False,
                                                       force_mode=clutch_mode)
                            active_clutch = self.active_action.get("right")
                            if active_clutch is not None:
                                active_clutch["two_hand_clutch"] = True
                            if clutch_mode == "scroll_window":
                                self.status = "Two-fist content scroll active — move the right fist; open either hand to stop"
                        self.fist_shake_samples["right"] = []
                if self.fist_hold_armed[side]:
                    # Direct manipulation freezes only the action choice, not pointer tracking.
                    action = self.fist_action[side]
                    previous = self.fist_last_points[side]
                    if previous is None:
                        previous = cursor
                    step_x, step_y = cursor.x()-previous.x(), cursor.y()-previous.y()
                    if action == "slider":
                        slider_name = self.fist_start_controls[side]
                        # Use a slider-specific stable cursor. A single malformed landmark
                        # frame can move the cursor hundreds of pixels; do not let that slam
                        # the smoothing value to an endpoint. A new position must be stable
                        # for three frames before it is accepted.
                        last_good = self.slider_last_good_cursor.get(side) or QPointF(previous)
                        candidate_jump = math.hypot(cursor.x()-last_good.x(), cursor.y()-last_good.y())
                        jump_limit = max(58.0, min(100.0, self.width()*0.045))
                        accept_cursor = True
                        if candidate_jump > jump_limit:
                            candidate = self.slider_cursor_jump_candidate.get(side)
                            candidate_tolerance = max(34.0, self.width()*0.02)
                            if candidate is not None and math.hypot(cursor.x()-candidate[0].x(), cursor.y()-candidate[0].y()) <= candidate_tolerance:
                                count = candidate[1]+1
                            else:
                                count = 1
                            self.slider_cursor_jump_candidate[side] = (QPointF(cursor), count)
                            accept_cursor = count >= 3
                        else:
                            self.slider_cursor_jump_candidate[side] = None
                        if not accept_cursor:
                            # Keep the knob and TLI pinned to the last safe point while the
                            # hand tracker confirms whether the new location is real.
                            drag_point = QPointF(last_good.x()+self.fist_slider_offset_x.get(side, 0.0), last_good.y())
                            reticle = self._update_canvas_slider(slider_name, drag_point)
                            self.tli_overrides[side] = ("slider", reticle)
                            self.fist_last_points[side] = QPointF(cursor)
                            continue
                        effective_dx = cursor.x()-last_good.x()
                        self.fist_slider_offset_x[side] -= effective_dx * (1.0-self.slider_drag_gain)
                        self.slider_last_good_cursor[side] = QPointF(cursor)
                        self.slider_cursor_jump_candidate[side] = None
                        drag_point = QPointF(cursor.x()+self.fist_slider_offset_x.get(side, 0.0), cursor.y())
                        reticle = self._update_canvas_slider(slider_name, drag_point)
                        self.tli_overrides[side] = ("slider", reticle)
                    elif action in ("menu_drag", "canvas_drag"):
                        self.hub_pos = QPointF(
                            min(0.82, max(0.18, self.hub_pos.x()+step_x*self.hand_tracking_sensitivity/max(1, self.width()))),
                            min(0.82, max(0.18, self.hub_pos.y()+step_y*self.hand_tracking_sensitivity/max(1, self.height()))),
                        )
                    self.fist_last_points[side] = QPointF(cursor)
                continue

            # Stable opening ends a quick fist click, or confirms a choice/cancels a hold.
            if self.fist_started_at[side] is not None:
                elapsed = now-self.fist_started_at[side]
                origin = self.fist_origin_points[side]
                if is_open:
                    self.fist_release_candidate_since[side] = None
                    if self.fist_open_candidate_since[side] is None:
                        self.fist_open_candidate_since[side] = now
                    if now-self.fist_open_candidate_since[side] < 0.07:
                        continue
                    # Use the first observed open frame as release time; don't make the
                    # 70ms release stabiliser accidentally turn a quick grasp into a hold.
                    release_elapsed = self.fist_open_candidate_since[side]-self.fist_started_at[side]
                    if (not self.fist_hold_armed[side] and
                            self.click_debounce_seconds <= release_elapsed <= self.fist_quick_click_max_seconds):
                        click_point = origin if origin is not None else cursor
                        in_glass_ui = self._canvas_menu_panel_contains(click_point)
                        if self.right_click_mode_enabled and not in_glass_ui:
                            self._activate_canvas_right_click(click_point, side, now)
                            self.last_quick_grasp[side] = None
                        else:
                            # Keep double-click natural: two ordinary quick grasps at
                            # nearly the same TLI point generate two native click events.
                            # The operating system applies its own double-click timing.
                            native_target = self.desktop_mode == "os" and not in_glass_ui
                            previous_click = self.last_quick_grasp.get(side)
                            if (native_target and previous_click is not None and
                                    now-previous_click[0] <= self.quick_grasp_double_click_window and
                                    math.hypot(click_point.x()-previous_click[1].x(),
                                               click_point.y()-previous_click[1].y()) <= 52):
                                click_point = QPointF(previous_click[1])
                                self.last_quick_grasp[side] = None
                            elif native_target:
                                self.last_quick_grasp[side] = (now, QPointF(click_point))
                            else:
                                self.last_quick_grasp[side] = None
                            self._activate_canvas_click(click_point, side, now)
                            if self.tli_calibration_active_side is not None:
                                self.pointer_side = None
                                self.active_push_sides.clear()
                                self.depth += (self.target_depth-self.depth)*0.16
                                self.update()
                                return
                    # A long fist has already started manipulation; it cannot also click on release.
                    self._reset_fist_gesture(side)
                    if self.gesture_owner == side:
                        self.gesture_owner = None
                else:
                    self.fist_open_candidate_since[side] = None
                    if self.fist_release_candidate_since[side] is None:
                        self.fist_release_candidate_since[side] = now
                    elif now-self.fist_release_candidate_since[side] > 0.14:
                        self._reset_fist_gesture(side)
                        if self.gesture_owner == side:
                            self.gesture_owner = None

        # Right hand has priority for idle cursor lock. During an already-started
        # interaction, keep its owner (including the 1.5s tracking-loss grace).
        if self.gesture_owner in cursor_points:
            self.pointer_side = self.gesture_owner
        elif self.gesture_owner is not None and self._needs_tracking_loss_grace(self.gesture_owner):
            self.pointer_side = self.gesture_owner
        elif self.cursor_lock_priority in cursor_points:
            self.pointer_side = self.cursor_lock_priority
        elif ("right" if self.cursor_lock_priority == "left" else "left") in cursor_points:
            self.pointer_side = "right" if self.cursor_lock_priority == "left" else "left"
        else:
            self.pointer_side = None
        if self.pointer_side in cursor_points:
            self.canvas_active_side = self.pointer_side if self.canvas_active_side not in cursor_points else self.canvas_active_side
        if self.desktop_mode == "os":
            mouse_side = self.pointer_side
            if mouse_side not in cursor_points or self.mouse_detached_by_side.get(mouse_side, False):
                alternate = "left" if mouse_side == "right" else "right"
                mouse_side = alternate if (alternate in cursor_points and
                                           not self.mouse_detached_by_side.get(alternate, False)) else None
            if mouse_side in cursor_points:
                self._set_os_pointer(cursor_points[mouse_side])

        self.depth += (self.target_depth-self.depth)*0.16
        self._request_scene_repaint(now)

    def closeEvent(self, event):
        if getattr(self, "hand_calibration_window", None) is not None:
            self._cancel_hand_onboarding(announce=False)
        if getattr(self, "tli_calibration_window", None) is not None:
            self._cancel_tli_calibration(announce=False)
        if self.voice_selection_mode == "text" and pyautogui is not None:
            try:
                pyautogui.mouseUp(button="left", _pause=False)
            except Exception:
                pass
        self.voice_selection_mode = None
        self.voice_lasso_start = None
        self.focus_dictation_enabled = False
        self.focus_dictation_committed_words = []
        if getattr(self, "search_hand_overlay", None) is not None:
            self.search_hand_overlay.sync_timer.stop()
            self.search_hand_overlay.hide()
        if self.search_dialog is not None and self.search_dialog.isVisible():
            self.search_dialog.close()
        for side in ("left", "right"):
            self._end_pinch_drag(side, status=False)
            self._release_fist_mouse(side)
            self.active_action[side] = None
        self.speech.stop()
        # Do not leave a CPU-heavy local llama-server running after the app exits.
        try:
            stop_d1_server()
        except Exception:
            pass
        self.face_welcome.close()
        self.tts.close()
        self.tracker.close()
        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def mouseMoveEvent(self, event):
        # This overlay is gesture-controlled; physical mouse movement has no action.
        event.accept()

    def _custom_key_name_from_event(self, event):
        """Convert a Qt key event to a pyautogui-compatible shortcut token."""
        key = int(event.key())
        modifiers = event.modifiers()
        modifier_only = {int(Qt.Key_Control), int(Qt.Key_Shift), int(Qt.Key_Alt), int(Qt.Key_Meta),
                         int(Qt.Key_AltGr)}
        if key in modifier_only:
            return None
        parts = []
        if modifiers & Qt.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.AltModifier:
            parts.append("alt")
        if modifiers & Qt.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.MetaModifier:
            parts.append("win")
        if int(Qt.Key_A) <= key <= int(Qt.Key_Z):
            name = chr(key).lower()
        elif int(Qt.Key_0) <= key <= int(Qt.Key_9):
            name = chr(key)
        elif int(Qt.Key_F1) <= key <= int(Qt.Key_F35):
            name = "f" + str(key-int(Qt.Key_F1)+1)
        else:
            names = {
                int(Qt.Key_Space): "space", int(Qt.Key_Tab): "tab",
                int(Qt.Key_Backtab): "tab", int(Qt.Key_Return): "enter",
                int(Qt.Key_Enter): "enter", int(Qt.Key_Backspace): "backspace",
                int(Qt.Key_Delete): "delete", int(Qt.Key_Insert): "insert",
                int(Qt.Key_Home): "home", int(Qt.Key_End): "end",
                int(Qt.Key_PageUp): "pageup", int(Qt.Key_PageDown): "pagedown",
                int(Qt.Key_Left): "left", int(Qt.Key_Right): "right",
                int(Qt.Key_Up): "up", int(Qt.Key_Down): "down",
                int(Qt.Key_Escape): "esc", int(Qt.Key_Print): "printscreen",
                int(Qt.Key_Plus): "plus", int(Qt.Key_Minus): "minus",
                int(Qt.Key_Comma): "comma", int(Qt.Key_Period): "period",
                int(Qt.Key_Slash): "slash", int(Qt.Key_Backslash): "backslash",
                int(Qt.Key_Semicolon): "semicolon", int(Qt.Key_Apostrophe): "quote",
                int(Qt.Key_BracketLeft): "[", int(Qt.Key_BracketRight): "]",
                int(Qt.Key_Equal): "=", int(Qt.Key_QuoteLeft): "`",
                int(Qt.Key_CapsLock): "capslock", int(Qt.Key_NumLock): "numlock",
                int(Qt.Key_ScrollLock): "scrolllock", int(Qt.Key_Pause): "pause",
            }
            name = names.get(key)
            if name is None:
                text = (event.text() or "").strip()
                if len(text) == 1 and text.isprintable():
                    name = text.lower()
                else:
                    return None
        parts.append(name)
        return "+".join(parts)

    def keyPressEvent(self, event):
        capture_index = self.custom_gesture_key_capture
        if capture_index is not None:
            if event.key() == Qt.Key_Escape and not event.modifiers():
                self.custom_gesture_key_capture = None
                self.custom_gesture_message = "Custom key capture cancelled."
                event.accept()
                self.update()
                return
            if event.isAutoRepeat():
                event.accept()
                return
            combo = self._custom_key_name_from_event(event)
            if combo:
                if 0 <= capture_index < len(self.custom_gestures):
                    gesture = self.custom_gestures[capture_index]
                    gesture["action"] = "custom_key"
                    gesture["key_combo"] = "+".join(
                        part.upper() for part in combo.split("+")
                    )
                    self._save_custom_gestures_now()
                    self.custom_gesture_message = (f"Saved shortcut {gesture['key_combo']} for "
                                                   f"{gesture.get('label', 'gesture')}.")
                self.custom_gesture_key_capture = None
                event.accept()
                self.update()
                return
            event.accept()
            return
        if event.key() == Qt.Key_Space and event.modifiers() & Qt.ControlModifier:
            if self.search_dialog is not None and self.search_dialog.isVisible():
                self.search_dialog.close()
            else:
                self.open_search_bar(from_voice=False)
            return
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key_PageDown:
            self.target_depth = min(2, self.target_depth+1)
        elif event.key() == Qt.Key_PageUp:
            self.target_depth = max(0, self.target_depth-1)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        """Paint one frame; the painter is always released, even if drawing fails.

        If an exception escaped mid-frame, the QPainter stayed active on the window,
        and every later repaint then failed ("Painter not active" / "one painter at
        a time"), leaving stale frames stacked on screen. Re-entrant paints (possible
        while native windows are being moved) are skipped for the same reason.
        """
        if getattr(self, "_in_paint_event", False):
            return
        self._in_paint_event = True
        self._frame_painter = None
        try:
            self._paint_frame(event)
        except Exception:
            if not getattr(self, "_paint_error_reported", False):
                self._paint_error_reported = True
                import traceback
                print("Spatial Desktop: a frame failed to paint (later failures are not repeated):")
                traceback.print_exc()
        finally:
            painter = self._frame_painter
            self._frame_painter = None
            if painter is not None and painter.isActive():
                painter.end()
            self._in_paint_event = False

    def _paint_frame(self, event):
        # Native video windows can temporarily invalidate the transparent overlay surface
        # while Windows is moving/resizing them. Skip that frame instead of issuing
        # follow-on drawing calls against an inactive painter.
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter()
        self._frame_painter = painter
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()
        now = time.monotonic()
        notification_stack_opacity = self._notification_stack_opacity(now)
        # Each notification anchor owns a vertical stack. Multiple active bars may share
        # bottom-centre, but their rectangles are allocated cumulatively with a clear gap.
        notification_stack_offsets = {}
        if self.desktop_mode == "virtual":
            painter.fillRect(self.rect(), QColor(8, 12, 22))
            spacing = 64
            for x in range(0, width, spacing):
                painter.setPen(QPen(QColor(60, 105, 155, 28), 1))
                painter.drawLine(x, 0, x, height)
            for y in range(0, height, spacing):
                painter.drawLine(0, y, width, y)

        # Paint scroll-circle backdrops before hand geometry so the virtual hand always
        # renders in front of the translucent background.
        for marker_side in ("right", "left"):
            marker = self.fist_scroll_markers.get(marker_side)
            if not marker:
                continue
            action = self.active_action.get(marker_side) or {}
            scrolling = action.get("mode") == "scroll_window"
            if not scrolling and now > float(marker.get("until", 0.0)):
                continue
            point = marker.get("point")
            if point is None:
                continue
            fade = 1.0 if scrolling else max(0.0, min(1.0, (float(marker.get("until", 0.0))-now)/1.25))
            deadzone_mark = max(4.0, float(self.scroll_deadzone))
            backdrop_radius = deadzone_mark + 72.0 * 2.0 + 24.0
            backdrop_alpha = int(255.0 * max(0.0, min(100.0, self.scroll_backdrop_opacity)) / 100.0 * fade)
            grey_level = int(max(0.0, min(255.0, self.scroll_backdrop_hue)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(grey_level, grey_level, grey_level, backdrop_alpha)))
            painter.drawEllipse(point, backdrop_radius, backdrop_radius)

        cx, cy = self.hub_pos.x()*width, self.hub_pos.y()*height
        scale = self.menu_ui_scale()
        if self.menu_open and self.hub_visible:
            fade = min(1.0, max(0.0, (now-self.menu_transition_started)/self.menu_transition_duration))
            fade = fade*fade*(3-2*fade)
            painter.save()
            painter.setOpacity(fade)
            panel_width, panel_height = self._menu_panel_dimensions()
            page_progress = self._settings_page_transition_progress(now) if self.settings_open else 1.0
            target_page = self.settings_page_pending
            if (self.settings_open and target_page is not None and page_progress >= 0.5):
                self.settings_page = target_page
                self.settings_page_pending = None
                # Recalculate target dimensions now that the incoming page is active.
                panel_width, panel_height = self._menu_panel_dimensions()
            elif target_page is not None:
                panel_width, panel_height = self._menu_panel_dimensions(page=target_page)
            if self.settings_open and self.settings_page_transition_from_size is not None and page_progress < 1.0:
                old_w, old_h = self.settings_page_transition_from_size
                # Gentle cubic ease: no sudden snap, with a small elastic-like settle.
                t = page_progress
                eased = 1.0 - (1.0-t)**3 + 0.08*((1.0-t)**2)*t
                panel_width = old_w + (panel_width-old_w)*eased
                panel_height = old_h + (panel_height-old_h)*eased
            elif page_progress >= 1.0:
                self.settings_page_transition_from_size = None
            panel_width *= scale
            panel_h = panel_height*scale
            panel = QRectF(cx-panel_width/2, cy-panel_h/2, panel_width, panel_h)
            # Frosted-glass softness: blur the backdrop locally, then add a darker tint so
            # labels remain legible over both bright and busy desktop content.
            self._paint_frosted_backdrop(painter, panel, 28*scale, opacity=0.15)
            for expansion, alpha, pen_width in ((14*scale, 13, 12*scale), (7*scale, 22, 7*scale)):
                painter.setPen(QPen(QColor(120, 225, 245, alpha), pen_width))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(panel.adjusted(-expansion, -expansion, expansion, expansion),
                                        28*scale+expansion*0.5, 28*scale+expansion*0.5)
            gradient = QLinearGradient(panel.topLeft(), panel.bottomRight())
            gradient.setColorAt(0.0, QColor(105, 220, 235, 217))
            gradient.setColorAt(0.45, QColor(12, 48, 67, 217))
            gradient.setColorAt(1.0, QColor(4, 12, 28, 217))
            painter.setPen(QPen(QColor(175, 250, 255, 180), 2))
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(panel, 28*scale, 28*scale)
            if self.settings_open:
                # A soft translucent inner wash makes Settings read as frosted glass.
                inset = panel.adjusted(9*scale, 9*scale, -9*scale, -9*scale)
                painter.setPen(QPen(QColor(190, 245, 255, 65), 1.0*scale))
                painter.setBrush(QBrush(QColor(155, 225, 240, 34)))
                painter.drawRoundedRect(inset, 24*scale, 24*scale)

            # Keep the frosted panel continuously visible. Only the page content
            # briefly soft-fades during navigation, then returns to full opacity.
            if self.settings_open and self.settings_page_transition_started:
                tp = self._settings_page_transition_progress(now)
                if tp < 0.5:
                    content_opacity = 1.0 - (tp / 0.5)
                else:
                    content_opacity = min(1.0, (tp-0.5) / 0.5)
                content_opacity = content_opacity*content_opacity*(3.0-2.0*content_opacity)
                painter.setOpacity(fade * content_opacity)
            else:
                painter.setOpacity(fade)

            if not self.settings_open:
                close_banner = self._main_menu_close_rect(cx, cy, scale)
                # Red Close button; brighter red while either hand's TLI hovers over it.
                # Uses the same transformed TLI points that click hit-testing uses.
                close_hovered = any(
                    point is not None and close_banner.contains(point)
                    for point in self.side_cursor_positions.values()
                )
                close_gradient = QLinearGradient(close_banner.topLeft(), close_banner.bottomRight())
                if close_hovered:
                    for expansion, alpha in ((7*scale, 45), (3.5*scale, 90)):
                        painter.setPen(QPen(QColor(255, 90, 90, alpha), 3*scale))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRoundedRect(close_banner.adjusted(-expansion, -expansion, expansion, expansion),
                                                12*scale+expansion, 12*scale+expansion)
                    close_gradient.setColorAt(0.0, QColor(255, 105, 100, 245))
                    close_gradient.setColorAt(1.0, QColor(215, 30, 40, 250))
                    painter.setPen(QPen(QColor(255, 215, 210, 250), 2.0*scale))
                else:
                    close_gradient.setColorAt(0.0, QColor(200, 55, 60, 215))
                    close_gradient.setColorAt(1.0, QColor(95, 12, 22, 235))
                    painter.setPen(QPen(QColor(255, 150, 150, 210), 1.5*scale))
                painter.setBrush(QBrush(close_gradient))
                painter.drawRoundedRect(close_banner, 12*scale, 12*scale)
                painter.setPen(QColor(240, 255, 255))
                painter.setFont(self._ui_font(max(8, int(10*scale)), QFont.DemiBold))
                painter.drawText(close_banner, Qt.AlignCenter, "CLOSE")

                labels = [
                    (("OS DESKTOP" if self.desktop_mode == "virtual" else "VIRTUAL DESKTOP"), cx-116*scale, cy-101*scale, "▣"),
                    ("GESTURE MAP", cx, cy-101*scale, "✋"),
                    ("SPEECH COMMANDS", cx+116*scale, cy-101*scale, "≡"),
                    (("VOICE ON" if self.stt_enabled else "VOICE OFF"), cx-116*scale, cy+79*scale, "🎤"),
                    ("SETTINGS", cx, cy+79*scale, "⚙"),
                    ("SEARCH", cx+116*scale, cy+79*scale, "⌕"),
                ]
                for label, ox, oy, glyph in labels:
                    rect = QRectF(ox-49*scale, oy-42*scale, 98*scale, 84*scale)
                    # Match the Close button's responsive highlight on every glass-menu
                    # tile. Use the same transformed TLI positions used for hit-testing.
                    hovered = any(
                        point is not None and rect.contains(point)
                        for point in self.side_cursor_positions.values()
                    )
                    tile = QLinearGradient(rect.topLeft(), rect.bottomRight())
                    if hovered:
                        for expansion, alpha in ((7*scale, 38), (3.5*scale, 78)):
                            painter.setPen(QPen(QColor(100, 255, 255, alpha), 3*scale))
                            painter.setBrush(Qt.NoBrush)
                            painter.drawRoundedRect(rect.adjusted(-expansion, -expansion,
                                                                  expansion, expansion),
                                                    18*scale+expansion, 18*scale+expansion)
                        tile.setColorAt(0.0, QColor(165, 255, 255, 225))
                        tile.setColorAt(1.0, QColor(20, 100, 135, 235))
                        painter.setPen(QPen(QColor(220, 255, 255, 250), 2.0*scale))
                    else:
                        if label in ("VOICE ON", "VOICE OFF"):
                            tile.setColorAt(0.0, QColor(55, 170, 95, 205) if self.stt_enabled else QColor(82, 88, 98, 180))
                            tile.setColorAt(1.0, QColor(18, 92, 48, 225) if self.stt_enabled else QColor(42, 47, 57, 220))
                            painter.setPen(QPen(QColor(155, 245, 180, 220) if self.stt_enabled else QColor(165, 172, 182, 210), 1.5*scale))
                        else:
                            tile.setColorAt(0.0, QColor(125, 245, 250, 145))
                            tile.setColorAt(1.0, QColor(7, 28, 48, 195))
                            painter.setPen(QPen(QColor(155, 250, 250, 220), 1.5*scale))
                    painter.setBrush(QBrush(tile))
                    painter.drawRoundedRect(rect, 18*scale, 18*scale)
                    painter.setPen(QColor(230, 255, 255))
                    icon_rect = QRectF(ox-46*scale, oy-34*scale, 92*scale, 30*scale)
                    label_rect = QRectF(ox-47*scale, oy-1*scale, 94*scale, 35*scale)
                    if label == "SPEECH COMMANDS":
                        # A command-list icon is visually distinct from the live voice/STT waves.
                        pen = QPen(QColor(230, 255, 255), 2.0*scale, Qt.SolidLine, Qt.RoundCap)
                        painter.setPen(pen)
                        for line_i, line_w in enumerate((19, 25, 15)):
                            yy = oy-27*scale + line_i*7*scale
                            painter.drawEllipse(QPointF(ox-13*scale, yy), 1.2*scale, 1.2*scale)
                            painter.drawLine(QPointF(ox-7*scale, yy), QPointF(ox+(line_w-7)*scale, yy))
                    elif label == "SEARCH":
                        painter.setPen(QPen(QColor(230, 255, 255), 2.4*scale, Qt.SolidLine, Qt.RoundCap))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(QRectF(ox-12*scale, oy-31*scale, 17*scale, 17*scale))
                        painter.drawLine(QPointF(ox+1*scale, oy-18*scale), QPointF(ox+10*scale, oy-9*scale))
                    else:
                        painter.setFont(self._ui_font(20, QFont.DemiBold))
                        painter.drawText(icon_rect, Qt.AlignCenter, glyph)
                    painter.setPen(QColor(230, 255, 255))
                    button_label = label.replace("SPEECH COMMANDS", "SPEECH\nCOMMANDS").replace("VIRTUAL DESKTOP", "VIRTUAL\nDESKTOP")
                    self._draw_fitted_button_label(painter, label_rect, button_label,
                                                   base_size=9, min_size=7, inset=2)
            else:
                # The four Settings tab captions already name their pages; retain a heading
                # only for the two reference pages, which have no tab buttons of their own.
                if self.settings_page in ("gesture_map", "speech_commands"):
                    page_titles = {"gesture_map": "GESTURE MAP", "speech_commands": "SPEECH COMMANDS"}
                    painter.setPen(QColor(235, 255, 255))
                    painter.setFont(self._ui_font(16, QFont.DemiBold))
                    title_top = -self._settings_layout()["height"]*0.5 + 26.0
                    painter.drawText(QRectF(cx-200*scale, cy+title_top*scale, 400*scale, 32*scale),
                                     Qt.AlignCenter | Qt.AlignVCenter, page_titles[self.settings_page])
                back = self._settings_back_rect(cx, cy, scale)
                painter.setBrush(QBrush(QColor(35, 105, 125, 225)))
                painter.setPen(QPen(QColor(175, 250, 255, 220), 1.5))
                painter.drawRoundedRect(back, 12*scale, 12*scale)
                painter.setPen(QColor(235, 255, 255))
                painter.setFont(self._ui_font(10, QFont.DemiBold))
                painter.drawText(back, Qt.AlignCenter, "← BACK")

                tab_labels = {"interaction": "INTERACTION", "hands": "HANDS", "tli": "TLI",
                              "notifications": "NOTIFY", "gestures": "GESTURES", "voice": "VOICE",
                              "d1": "ENGINE", "update": "UPDATE"}
                info_pages = ("gesture_map", "speech_commands")
                if self.settings_page not in info_pages:
                    for page_name, tab in self._settings_tab_rects(cx, cy, scale).items():
                        selected = page_name == self.settings_page
                        painter.setBrush(QBrush(QColor(55, 145, 82, 225) if selected else QColor(75, 80, 89, 205)))
                        painter.setPen(QPen(QColor(155, 245, 180, 235) if selected else QColor(155, 162, 172, 170), 1.5))
                        painter.drawRoundedRect(tab, 10*scale, 10*scale)
                        painter.setPen(QColor(245, 255, 255))
                        painter.setFont(self._ui_font(7, QFont.DemiBold))
                        # Explicit, balanced interior padding prevents labels from looking
                        # cramped against the rounded tabs. The tab itself remains the hit area.
                        text_rect = tab.adjusted(6*scale, 8*scale, -6*scale, -8*scale)
                        painter.drawText(text_rect, Qt.AlignCenter, tab_labels[page_name])

                if self.settings_page == "gesture_map":
                    columns = [
                        [
                            ("POINTER & QUICK ACTIONS", True),
                            ("Move the TLI with your palm anchor.", False),
                            ("Quick pinch → release: left click.", False),
                            ("Quick grip → open: left click.", False),
                            ("Clicks use the gesture start point.", False),
                            ("", False),
                            ("LEFT-HAND MODIFIER", True),
                            ("Hold LEFT fist to enable the modifier.", False),
                            ("RIGHT pinch → release: right-click.", False),
                            ("RIGHT grip: grab the window under TLI.", False),
                            ("Move RIGHT hand; open it to release.", False),
                            ("Open LEFT fist to leave modifier mode.", False),
                            ("", False),
                            ("PEACE-SIGN TRACKING TOGGLE", True),
                            ("Hold a V-sign for 1.5 seconds.", False),
                            ("Ring counts down to the toggle.", False),
                            ("Only that hand's tracking is toggled.", False),
                            ("Voice confirms enabled or disabled.", False),
                            ("", False),
                            ("LONG GESTURES", True),
                            ("Move held pinch before timer: drag/drop.", False),
                            ("Stationary pinch opens Pinch Menu.", False),
                            ("Ring fills after the pinch buffer.", False),
                        ],
                        [
                            ("PINCH MENU", True),
                            ("Hold pinch until the ring fills.", False),
                            ("Release, then pinch/grip an item.", False),
                            ("Pinch/grip outside closes the popup.", False),
                            ("", False),
                            ("WINDOW / SCROLL", True),
                            ("Modifier: left fist + right grip drag.", False),
                            ("Normal drag: make both hands fists,", False),
                            ("keep one still, then move the other.", False),
                            ("A vertical fist movement can scroll", False),
                            ("inside content or a scrollbar.", False),
                            ("Open hand to end a normal drag/scroll.", False),
                            ("", False),
                            ("SETTINGS", True),
                            ("Pinch/grip near a slider to adjust it.", False),
                            ("TLI is biased slightly toward the thumb.", False),
                            ("", False),
                            ("VOICE CONTROL", True),
                            ("Say 'show menu' or 'show context'.", False),
                            ("Say 'pause tracking' / 'resume tracking'.", False),
                            ("Use Speech Commands for the full list.", False),
                        ],
                    ]
                    self._paint_reference_columns(painter, columns, cx, cy, scale)

                elif self.settings_page == "speech_commands":
                    columns = [
                        [
                            ("MENU / TRACKING", True),
                            ("show menu · show context", False),
                            ("pause tracking · resume tracking", False),
                            ("search bar · open search", False),
                            ("close search · open top result", False),
                            ("begin dictation · end dictation", False),
                            ("pause dictation · resume dictation", False),
                            ("", False),
                            ("WINDOW MANAGEMENT", True),
                            ("close window · minimize window", False),
                            ("maximize window · restore window", False),
                            ("switch window · show desktop", False),
                            ("snap window left · snap window right", False),
                            ("fullscreen", False),
                            ("", False),
                            ("FILES / APPS", True),
                            ("open file explorer · open user folder", False),
                            ("open downloads · open documents", False),
                            ("open desktop folder · open pictures", False),
                            ("open videos · open music folder", False),
                            ("open notepad · open calculator", False),
                        ],
                        [
                            ("EDITING / SELECTION", True),
                            ("copy this · copy selection · cut this", False),
                            ("paste · paste here · select all", False),
                            ("undo · redo", False),
                            ("start text selection", False),
                            ("finish selection · cancel selection", False),
                            ("copy image · start image lasso", False),
                            ("copy this to finish a lasso", False),
                            ("", False),
                            ("MEDIA / AUDIO", True),
                            ("play pause · next track", False),
                            ("previous track · volume up/down", False),
                            ("mute · unmute", False),
                            ("", False),
                            ("FILE / APP NAVIGATION", True),
                            ("new folder · new tab · close tab", False),
                            ("go back · go forward · go up", False),
                            ("refresh · rename selected", False),
                            ("delete selected", False),
                            ("", False),
                            ("Commands are local; availability can", False),
                            ("depend on your OS and active app.", False),
                        ],
                    ]
                    self._paint_reference_columns(painter, columns, cx, cy, scale)

                elif self.settings_page in ("interaction", "hands", "tli", "gestures",
                                            "notifications", "voice", "update", "d1"):
                    # Every control below is placed by _settings_layout(), the same
                    # geometry used for click hit-testing and hover/press feedback.
                    controls = self._settings_control_rects(cx, cy, scale)
                    text_rects = self._settings_text_rects(cx, cy, scale)
                    label_size = max(7, int(8*scale))
                    if self.settings_page == "gestures":
                        self._paint_custom_gestures_page(painter, cx, cy, scale, now)
                    self._paint_settings_sliders(painter, cx, cy, scale)

                    if self.settings_page == "interaction":
                        scroll = controls["scroll_toggle"]
                        swap_nav = controls["swap_drag_scroll_toggle"]
                        priority = controls["priority_toggle"]
                        for rect, enabled in ((scroll, self.invert_scroll),
                                              (swap_nav, self.swap_drag_scroll_notches)):
                            painter.setBrush(QBrush(QColor(55, 145, 82, 225) if enabled else QColor(75, 80, 89, 205)))
                            painter.setPen(QPen(QColor(155, 245, 180, 220) if enabled else QColor(155, 162, 172, 200), 1.5*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                        painter.setBrush(QBrush(QColor(10, 37, 55, 215)))
                        painter.setPen(QPen(QColor(125, 235, 245, 200), 1.5*scale))
                        painter.drawRoundedRect(priority, 10*scale, 10*scale)
                        painter.setPen(QColor(225, 250, 250))
                        self._draw_fitted_button_label(painter, scroll, "SCROLL  ·  "+("INVERTED" if self.invert_scroll else "NORMAL"), base_size=label_size, min_size=6, inset=12*scale)
                        self._draw_fitted_button_label(painter, swap_nav, "DRAG NAV  ·  "+("BACK/FORWARD SWAPPED" if self.swap_drag_scroll_notches else "NORMAL"), base_size=label_size, min_size=6, inset=12*scale)
                        self._draw_fitted_button_label(painter, priority, "CURSOR PRIORITY  ·  "+self.cursor_lock_priority.upper(), base_size=label_size, min_size=6, inset=12*scale)

                    elif self.settings_page == "tli":
                        glow_toggle = controls["glow_toggle"]
                        painter.setBrush(QBrush(QColor(55, 145, 82, 225) if self.tli_glow_enabled else QColor(75, 80, 89, 205)))
                        painter.setPen(QPen(QColor(155, 245, 180, 220) if self.tli_glow_enabled else QColor(155, 162, 172, 200), 1.5*scale))
                        painter.drawRoundedRect(glow_toggle, 10*scale, 10*scale)
                        painter.setPen(QColor(225, 250, 250))
                        self._draw_fitted_button_label(painter, glow_toggle, "TLI GLOW  ·  "+("ON" if self.tli_glow_enabled else "OFF"), base_size=label_size, min_size=6, inset=12*scale)

                        # Independent left/right TLI colour swatches are grouped here.
                        for label, key, color in (("LEFT TLI", "swatch_left", self.left_click_color),
                                                  ("RIGHT TLI", "swatch_right", self.right_click_color)):
                            swatch = controls[key]
                            centre = QPointF(swatch.center().x(), swatch.top()+24*scale)
                            painter.setBrush(QBrush(color))
                            painter.setPen(QPen(QColor(225, 250, 255, 230), 2*scale))
                            painter.drawEllipse(centre, 14*scale, 14*scale)
                            painter.setPen(QColor(210, 245, 240))
                            self._draw_fitted_button_label(
                                painter, QRectF(swatch.left(), swatch.top()+42*scale, swatch.width(), 20*scale),
                                label, base_size=label_size, min_size=6, inset=2)

                    elif self.settings_page == "hands":
                        # Three-way RENDER STYLE switch: SKELETON | FLAT | 3D.
                        switch = controls["render_style"]
                        painter.setBrush(QBrush(QColor(10, 37, 55, 225)))
                        painter.setPen(QPen(QColor(125, 235, 245, 210), 1.5*scale))
                        painter.drawRoundedRect(switch, 10*scale, 10*scale)
                        caption_w = switch.width()*self.RENDER_SWITCH_CAPTION
                        painter.setPen(QColor(200, 240, 248))
                        self._draw_fitted_button_label(painter, QRectF(switch.left(), switch.top(), caption_w, switch.height()),
                                                       "RENDER", base_size=label_size, min_size=6, inset=8*scale)
                        for seg_index, (seg_style, seg) in enumerate(self._render_style_segments(switch, scale)):
                            selected = seg_style == self.hand_render_style
                            if selected:
                                painter.setBrush(QBrush(QColor(55, 145, 82, 235)))
                                painter.setPen(QPen(QColor(155, 245, 180, 230), 1.2*scale))
                                painter.drawRoundedRect(seg.adjusted(1*scale, 0, -1*scale, 0), 8*scale, 8*scale)
                            elif seg_index > 0 and HAND_RENDER_STYLES[seg_index-1] != self.hand_render_style:
                                painter.setPen(QPen(QColor(125, 235, 245, 90), 1.0*scale))
                                painter.drawLine(QPointF(seg.left(), seg.top()+6*scale), QPointF(seg.left(), seg.bottom()-6*scale))
                            painter.setPen(QColor(245, 255, 250) if selected else QColor(175, 210, 220))
                            self._draw_fitted_button_label(painter, seg, HAND_RENDER_STYLE_LABELS[seg_style],
                                                           base_size=label_size, min_size=6, inset=4*scale)

                        tint_toggle = controls["skin_tint"]
                        painter.setBrush(QBrush(QColor(55, 145, 82, 225) if self.skin_use_skeleton_color else QColor(75, 80, 89, 205)))
                        painter.setPen(QPen(QColor(125, 235, 245, 210), 1.5*scale))
                        painter.drawRoundedRect(tint_toggle, 10*scale, 10*scale)
                        painter.setPen(QColor(235, 255, 255))
                        self._draw_fitted_button_label(painter, tint_toggle, "SKIN COLOUR  ·  "+("SKELETON COLOUR" if self.skin_use_skeleton_color else "NATURAL"), base_size=label_size, min_size=6, inset=12*scale)

                        # Each hand's independent calibration opens a temporary target window.
                        for calibration_side, rect in self._tli_calibration_button_rects(cx, cy, scale).items():
                            pending = self.tli_calibration_pending.get(calibration_side, False)
                            calibrated = self.tli_calibrations.get(calibration_side) is not None
                            if pending:
                                if self.tli_calibration_wait_release.get(calibration_side, False):
                                    label = f"{calibration_side.upper()} · RELEASE FIRST"
                                else:
                                    count = len(self.tli_calibration_samples.get(calibration_side, []))
                                    label = (f"{calibration_side.upper()} · PINCH IN WINDOW" if count == 0 else
                                             f"{calibration_side.upper()} · {count}/{self.tli_calibration_sample_count}")
                                fill = QColor(100, 130, 45, 235)
                            elif calibrated and self.tli_calibration_persisted.get(calibration_side, False):
                                label = f"{calibration_side.upper()} TLI · CALIBRATED"
                                fill = QColor(20, 104, 82, 225)
                            elif calibrated:
                                label = f"{calibration_side.upper()} · SESSION ONLY"
                                fill = QColor(125, 78, 28, 225)
                            else:
                                label = f"CALIBRATE {calibration_side.upper()} TLI"
                                fill = QColor(10, 47, 67, 225)
                            painter.setBrush(QBrush(fill))
                            painter.setPen(QPen(QColor(125, 235, 245, 210), 1.5*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                            painter.setPen(QColor(235, 255, 255))
                            self._draw_fitted_button_label(painter, rect, label, base_size=label_size, min_size=6, inset=12*scale)

                        hand_setup_rect = controls["hand_setup"]
                        painter.setBrush(QBrush(QColor(18, 80, 93, 235)))
                        painter.setPen(QPen(QColor(115, 242, 250, 220), 1.4*scale))
                        painter.drawRoundedRect(hand_setup_rect, 10*scale, 10*scale)
                        painter.setPen(QColor(235, 255, 255))
                        self._draw_fitted_button_label(painter, hand_setup_rect, "RUN HAND SCALE & PROXIMITY CALIBRATION",
                                                       base_size=label_size, min_size=6, inset=12*scale)

                    elif self.settings_page == "notifications":
                        for key, label, default in NOTIFICATION_SETTING_ROWS:
                            rect = controls["notify:" + key]
                            painter.setBrush(QBrush(QColor(10, 47, 67, 225)))
                            painter.setPen(QPen(QColor(140, 230, 240, 210), 1.5*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                            position = self.ui_preferences.get(key, default)
                            painter.setPen(QColor(237, 252, 255))
                            self._draw_fitted_button_label(painter, rect, f"{label}  ·  {position.upper().replace('-', ' ')}",
                                                           base_size=max(7, int(8.5*scale)), min_size=6, inset=12*scale)
                        painter.setPen(QColor(185, 225, 235))
                        self._draw_wrapped_fitted_text(painter, text_rects["help"],
                                                       "Select a row to cycle through six screen positions.\nChanges save automatically.",
                                                       base_size=max(7, int(8*scale)), min_size=6)

                    elif self.settings_page == "voice":
                        tts_toggle = controls["tts_toggle"]
                        voice_button = controls["voice_cycle"]
                        painter.setBrush(QBrush(QColor(55, 145, 82, 225) if self.tts.enabled else QColor(75, 80, 89, 205)))
                        painter.setPen(QPen(QColor(155, 245, 180, 220) if self.tts.enabled else QColor(155, 162, 172, 200), 1.5*scale))
                        painter.drawRoundedRect(tts_toggle, 10*scale, 10*scale)
                        painter.setBrush(QBrush(QColor(25, 95, 115, 225)))
                        painter.setPen(QPen(QColor(140, 230, 240, 215), 1.5*scale))
                        painter.drawRoundedRect(voice_button, 10*scale, 10*scale)
                        painter.setPen(QColor(240, 253, 255))
                        self._draw_fitted_button_label(painter, tts_toggle, "LOCAL TTS  ·  "+("ON" if self.tts.enabled else "OFF"), base_size=label_size, min_size=6, inset=12*scale)
                        self._draw_fitted_button_label(painter, voice_button, "VOICE  ·  "+str(self.tts.settings.get("voice", "bf_emma")).upper()+"  ·  CLICK TO CHANGE", base_size=label_size, min_size=6, inset=12*scale)

                        test_button = controls["test_voice"]
                        face_button = controls["face_welcome"]
                        streamtts_config_button = controls["streamtts_config"]
                        for rect in (test_button, face_button, streamtts_config_button):
                            painter.setBrush(QBrush(QColor(10, 47, 67, 225)))
                            painter.setPen(QPen(QColor(140, 230, 240, 210), 1.5*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                        painter.setPen(QColor(240, 253, 255))
                        self._draw_fitted_button_label(painter, test_button, "TEST VOICE", base_size=label_size, min_size=6, inset=12*scale)
                        face_label = ("FACE WELCOME  ·  " + self.face_welcome.profile_name() + "  ·  MANAGE" if self.face_welcome.has_profile()
                                      else "ENROL FACE FOR WELCOME-BACK")
                        self._draw_fitted_button_label(painter, face_button, face_label, base_size=label_size, min_size=6, inset=12*scale)
                        self._draw_fitted_button_label(painter, streamtts_config_button, "OPEN STREAMTTS CONFIG", base_size=label_size, min_size=6, inset=12*scale)
                        painter.setPen(QColor(170, 215, 225, 230))
                        voice_status = f"{self.tts.status}  ·  {self.face_enrollment_status}"
                        self._draw_wrapped_fitted_text(painter, text_rects["status"], voice_status[:150],
                                                       base_size=max(6, int(7*scale)), min_size=6)

                    elif self.settings_page == "update":
                        action_button = controls["update_action"]
                        downloading = self.update_action_running and self.update_action_mode == "download"
                        if self.update_action_running:
                            action_label = "INSTALLING…" if self.update_action_mode == "install" else ""
                        elif self.update_installed:
                            action_label = "RESTART"
                        elif self.update_download_path:
                            action_label = "INSTALL"
                        else:
                            action_label = "UPDATE"
                        painter.setBrush(QBrush(QColor(10, 47, 67, 225)))
                        painter.setPen(QPen(QColor(140, 230, 240, 210), 1.5*scale))
                        painter.drawRoundedRect(action_button, 10*scale, 10*scale)
                        if downloading:
                            inner = action_button.adjusted(3*scale, 3*scale, -3*scale, -3*scale)
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QBrush(QColor(48, 132, 76, 245)))
                            if self.update_progress_total > 0:
                                fraction = min(1.0, max(0.0, self.update_progress_current / self.update_progress_total))
                                fill = QRectF(inner.x(), inner.y(), inner.width()*fraction, inner.height())
                                painter.drawRoundedRect(fill, 7*scale, 7*scale)
                            painter.setPen(QColor(240, 253, 255))
                            progress_text = (f"{min(100, int(100*self.update_progress_current/self.update_progress_total))}%"
                                             if self.update_progress_total > 0 else "DOWNLOADING…")
                            self._draw_fitted_button_label(painter, action_button, progress_text, base_size=max(7, int(9*scale)), min_size=6, inset=12*scale)
                        else:
                            painter.setPen(QColor(240, 253, 255))
                            self._draw_fitted_button_label(painter, action_button, action_label, base_size=max(7, int(9*scale)), min_size=6, inset=12*scale)
                        painter.setPen(QColor(185, 225, 232))
                        self._draw_wrapped_fitted_text(painter, text_rects["status"], self.update_status,
                                                       base_size=max(7, int(8*scale)), min_size=6,
                                                       align=Qt.AlignHCenter | Qt.AlignTop)
                        self._draw_wrapped_fitted_text(painter, text_rects["about"],
                                                       "Checks GitHub version.txt, downloads a newer release, then lets you install it while preserving settings and calibration. Restart after installation.",
                                                       base_size=max(6, int(7*scale)), min_size=6,
                                                       align=Qt.AlignHCenter | Qt.AlignTop)

                    elif self.settings_page == "d1":
                        info = d1_status()
                        model_key = selected_d1_model()
                        model_button = controls["d1_model"]
                        runtime_button = controls["d1_start"]
                        stop_button = controls["d1_stop"]
                        retry_button = controls["d1_retry"]
                        running = bool(info.get("ready") or info.get("loading"))
                        model_active = bool(info.get("model_exists"))
                        retry_active = bool(info.get("error") or info.get("downloading") or not model_active)
                        button_states = (
                            (model_button, model_active),
                            (runtime_button, running),
                            (stop_button, running),
                            (retry_button, retry_active),
                        )
                        for rect, active in button_states:
                            fill = QColor(48, 132, 76, 230) if active else QColor(72, 78, 88, 210)
                            edge = QColor(155, 245, 180, 230) if active else QColor(155, 165, 178, 190)
                            painter.setBrush(QBrush(fill))
                            painter.setPen(QPen(edge, 1.5*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                        painter.setPen(QColor(240, 253, 255))
                        self._draw_fitted_button_label(painter, model_button,
                                                       "MODEL  ·  " + D1_MODELS[model_key]["label"].upper() + "  ·  DEFAULT / ENSURE",
                                                       base_size=label_size, min_size=6, inset=10*scale)
                        self._draw_fitted_button_label(painter, runtime_button, "START / ENSURE RUNNING", base_size=label_size, min_size=6, inset=8*scale)
                        self._draw_fitted_button_label(painter, stop_button, "STOP LOCAL SERVER", base_size=label_size, min_size=6, inset=8*scale)
                        self._draw_fitted_button_label(painter, retry_button, "DOWNLOAD / RETRY / START", base_size=label_size, min_size=6, inset=10*scale)
                        painter.setPen(QColor(185, 225, 232))
                        if info.get("downloading"):
                            state = "DOWNLOADING MODEL FILES…"
                        elif info.get("loading"):
                            state = "STARTING LLAMA.CPP SERVER…"
                        elif info.get("ready"):
                            state = "LOCAL SERVER: RUNNING"
                        elif info.get("error"):
                            state = "STATUS: " + str(info.get("error"))[:90]
                        else:
                            state = "LOCAL SERVER: STOPPED / NOT READY"
                        self._draw_wrapped_fitted_text(painter, text_rects["state"], state,
                                                       base_size=max(7, int(8*scale)), min_size=6)
                        self._draw_wrapped_fitted_text(painter, text_rects["about"],
                                                       "Runs locally through llama.cpp. No separate terminal is needed. "
                                                       "The larger model downloads only after you select it.",
                                                       base_size=max(6, int(7*scale)), min_size=6)
                        # Compact in-panel event log beneath the status text.
                        log_rect = text_rects["log"]
                        painter.setBrush(QBrush(QColor(3, 10, 20, 220)))
                        painter.setPen(QPen(QColor(90, 170, 190, 190), 1.0*scale))
                        painter.drawRoundedRect(log_rect, 8*scale, 8*scale)
                        signature = (bool(info.get("ready")), bool(info.get("loading")),
                                     bool(info.get("downloading")), str(info.get("error") or ""),
                                     bool(info.get("model_exists")))
                        if signature != self._last_d1_log_signature:
                            self._last_d1_log_signature = signature
                            if info.get("error"):
                                self._log_d1_ui("ERROR: " + str(info.get("error"))[:120])
                            elif info.get("downloading"):
                                self._log_d1_ui("Downloading model files…")
                            elif info.get("loading"):
                                self._log_d1_ui("Starting local runtime…")
                            elif info.get("ready"):
                                self._log_d1_ui("Local server ready")
                            elif info.get("model_exists"):
                                self._log_d1_ui("Model files available; server stopped")
                            else:
                                self._log_d1_ui("Model files missing; use Download / Retry / Start")
                        painter.setPen(QColor(185, 225, 232))
                        painter.setFont(self._ui_font(7, QFont.DemiBold))
                        painter.drawText(log_rect.adjusted(9*scale, 4*scale, -9*scale, -4*scale),
                                         Qt.AlignLeft | Qt.AlignTop, "ENGINE LOG")
                        log_lines = self.d1_ui_log[-5:]
                        log_text = "\n".join(log_lines)
                        painter.save()
                        painter.setClipRect(log_rect.adjusted(2*scale, 2*scale, -2*scale, -2*scale))
                        painter.setFont(self._ui_font(7, QFont.Normal))
                        painter.drawText(log_rect.adjusted(9*scale, 21*scale, -9*scale, -5*scale),
                                         Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, log_text)
                        painter.restore()

                    if self.settings_page == "gestures" and self.custom_gesture_dropdown is not None:
                        # Painted after the sliders so the open list sits on top of them.
                        self._paint_custom_gesture_dropdown(painter, cx, cy, scale)
            painter.restore()

        # When move mode is engaged, animate waves and sparkles from the TLI/hand anchor.
        # This is the only window-drag feedback: the old outline around the grabbed
        # window was removed (it re-queried native window bounds on every frame and
        # left stacked outlines over video players such as PotPlayer).
        # This visual feedback appears only after the fist shake has been recognised.
        for side, active in self.active_action.items():
            if not active or active.get("mode") != "drag_window":
                continue
            cursor = self.side_cursor_positions.get(side)
            if cursor is None:
                continue
            cursor = self._visual_tli_position(side, cursor)
            for wave_index in range(3):
                phase = (now*1.25 + wave_index/3.0) % 1.0
                radius = 14.0 + phase*48.0
                alpha = int(215*(1.0-phase))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(80, 235, 255, max(10, alpha)), 2.2))
                painter.drawEllipse(cursor, radius, radius)
            painter.setPen(QPen(QColor(205, 255, 255, 235), 2.0, Qt.SolidLine, Qt.RoundCap))
            for sparkle_index in range(8):
                angle = now*2.4 + sparkle_index*math.pi/4.0
                radius = 23.0 + 5.0*math.sin(now*4.2 + sparkle_index)
                x = cursor.x() + math.cos(angle)*radius
                y = cursor.y() + math.sin(angle)*radius
                painter.drawLine(QPointF(x-3, y), QPointF(x+3, y))
                painter.drawLine(QPointF(x, y-3), QPointF(x, y+3))

        # When Search is open, the dedicated topmost SearchHandOverlay is the single
        # owner of hands/TLIs/timers/Pinch Menu. Painting the same layer here too causes
        # offset duplicate/ghost hands, often noticed around the screen centre.
        search_overlay = getattr(self, "search_hand_overlay", None)
        search_dialog = getattr(self, "search_dialog", None)
        search_overlay_draws_hand_layer = bool(
            search_overlay is not None and search_overlay.isVisible() and
            search_dialog is not None and search_dialog.isVisible()
        )
        calibration_window_owns_hand_layer = (
            (self.tli_calibration_active_side is not None and
             self.tli_calibration_window is not None and self.tli_calibration_window.isVisible()) or
            (self.hand_calibration_active and self.hand_calibration_window is not None and
             self.hand_calibration_window.isVisible())
        )
        if not search_overlay_draws_hand_layer and not calibration_window_owns_hand_layer:
            # The floating Pinch Menu (and the completed gesture ring fading away as it
            # fades in) is painted first, so the virtual hand and its TLI draw on top of it.
            for side in ("left", "right"):
                self._paint_gesture_timer_fade(painter, side, now)
                self._paint_pinch_menu(painter, side, now)

            # Reticle and skeleton use the fixed hand-relative (optionally calibrated) TLI anchor.
            # During TLI calibration the dedicated target window is the sole visual owner;
            # painting here as well produced a second, differently transformed skeleton.
            for side in ("left", "right"):
                points = self.hand_positions.get(side, [])
                if not points:
                    continue
                color = QColor(self.left_click_color if side == "left" else self.right_click_color)
                # The skeleton is mapped without cursor clamping; joints may leave the
                # camera-to-screen active rectangle and be clipped by the display itself.
                # Clamping every joint individually made the skeleton squash against an
                # artificial wall near the top and bottom of the screen.
                self._paint_hand_skeleton(painter, points, side)
                # Keep the virtual skeleton visible when a hand is disabled, but remove
                # its moving TLI/cursor until thumbs-up re-enables tracking.
                if self.gesture_tracking_enabled.get(side, True):
                    cursor = self.side_cursor_positions.get(side)
                    if cursor is None:
                        cursor = self._scale_hand_tli_point(self.interactive_hand_anchor(points, side), points, side)
                    cursor = self._visual_tli_position(side, cursor, points)
                    engaged = side in self.active_push_sides or (self.click_side == side and now < self.click_flash_until)
                    self._paint_tli(painter, cursor, side, engaged)

            # Thumbs-up gesture-tracking toggle has a visible two-second confirmation ring.
            for side in ("left", "right"):
                thumbs_started = self.thumbs_up_hold_started.get(side)
                if thumbs_started is None or self.thumbs_up_toggle_latched.get(side, False):
                    continue
                cursor = self.side_cursor_positions.get(side)
                if cursor is None:
                    # Tracking-off hands still need a visible re-enable countdown, but
                    # this visual anchor is not fed back into cursor/gesture processing.
                    points = self.hand_positions.get(side, [])
                    if not points:
                        continue
                    cursor = self._scale_hand_tli_point(self.interactive_hand_anchor(points, side), points, side)
                cursor = self._visual_tli_position(side, cursor)
                elapsed = max(0.0, now-thumbs_started)
                progress = min(1.0, elapsed/max(0.01, self.gesture_tracking_toggle_hold_seconds))
                remaining = max(0.0, self.gesture_tracking_toggle_hold_seconds-elapsed)
                ring = QRectF(cursor.x()-38, cursor.y()-38, 76, 76)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(105, 240, 255, 95), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(105, 240, 255, 245), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(220, 255, 255, 245))
                painter.setFont(self._ui_font(8, QFont.DemiBold))
                painter.drawText(QRectF(cursor.x()-24, cursor.y()+27, 48, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")

            # Peace-sign tracking toggles use an immediate visual state notice; no toggle TTS.
            for side in ("left", "right"):
                peace_started = self.peace_hold_started.get(side)
                if peace_started is None or self.peace_toggle_latched.get(side, False):
                    continue
                cursor = self.side_cursor_positions.get(side)
                if cursor is None:
                    continue
                cursor = self._visual_tli_position(side, cursor)
                elapsed = max(0.0, now-peace_started)
                progress = min(1.0, elapsed/max(0.01, self.peace_toggle_hold_seconds))
                remaining = max(0.0, self.peace_toggle_hold_seconds-elapsed)
                ring = QRectF(cursor.x()-32, cursor.y()-32, 64, 64)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(255, 220, 110, 100), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(255, 220, 110, 248), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(255, 242, 205, 245))
                painter.setFont(self._ui_font(8, QFont.DemiBold))
                painter.drawText(QRectF(cursor.x()-22, cursor.y()+23, 44, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")

            # Custom gestures with their circle timer on show a filling ring at the TLI
            # (midway between both TLIs for the overlap gesture) until the gesture fires.
            for key, (label, progress, remaining) in list(getattr(self, "custom_gesture_rings", {}).items()):
                if key == "overlap":
                    a, b = self.side_cursor_positions.get("left"), self.side_cursor_positions.get("right")
                    if a is None or b is None:
                        continue
                    cursor = QPointF((a.x()+b.x())*0.5, (a.y()+b.y())*0.5)
                else:
                    cursor = self.side_cursor_positions.get(key)
                    if cursor is None:
                        continue
                    cursor = self._visual_tli_position(key, cursor)
                ring = QRectF(cursor.x()-41, cursor.y()-41, 82, 82)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(110, 255, 170, 85), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(110, 255, 170, 245), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(235, 255, 240, 245))
                painter.setFont(self._ui_font(8, QFont.DemiBold))
                painter.drawText(QRectF(cursor.x()-50, cursor.y()-60, 100, 16), Qt.AlignCenter, label)
                painter.drawText(QRectF(cursor.x()-22, cursor.y()+23, 44, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")

            # Only timed pinches show a progress circle. A right fist scrolls on
            # deliberate vertical movement and does not wait for or draw a fist timer.
            for side in ("left", "right"):
                if self.fist_action_menu_open.get(side):
                    continue
                started = self.menu_pinch_started.get(side)
                if started is None or self.menu_pinch_release_candidate_since.get(side) is not None:
                    continue
                cursor = self.side_cursor_positions.get(side)
                if cursor is None:
                    continue
                cursor = self._visual_tli_position(side, cursor)
                elapsed = self._menu_pinch_elapsed_seconds(side, now)
                duration = self.menu_pinch_hold_seconds
                total_duration = self.pinch_buffer_seconds + duration
                progress = min(1.0, max(0.0, elapsed/max(0.01, duration)))
                buffer_remaining = max(0.0, self.pinch_buffer_seconds-(now-started))
                # During the initial buffer the ring stays empty and the label shows only
                # the 0.8s confirmation countdown; the hold countdown begins afterward.
                remaining = buffer_remaining if buffer_remaining > 0.0 else max(0.0, duration-elapsed)
                color = self.left_click_color if side == "left" else self.right_click_color
                ring = QRectF(cursor.x()-41, cursor.y()-41, 82, 82)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 85), 3))
                painter.drawEllipse(ring)
                painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 245), 4))
                painter.drawArc(ring, 90*16, -int(progress*360*16))
                painter.setPen(QColor(245, 251, 255, 245))
                painter.setFont(self._ui_font(8, QFont.DemiBold))
                painter.drawText(QRectF(cursor.x()-22, cursor.y()+23, 44, 16), Qt.AlignCenter,
                                 f"{remaining:.1f}s")


        if self.menu_open and self.hub_visible:
            # Keep the normal gesture legend visible while the glass menu is open,
            # including on Settings pages. Share the notification stack with the
            # optional modifier banner so the two messages never overlap.
            gesture_legend_position = self.ui_preferences.get("paused_notice_position", "bottom-center")
            gesture_legend = self._stacked_notification_rect(
                gesture_legend_position, min(850.0, width-24.0), 34,
                notification_stack_offsets, margin=18)
            painter.setPen(QPen(QColor(100, 235, 250, 220), 1.5))
            painter.setBrush(QBrush(QColor(4, 22, 36, 158)))
            painter.drawRoundedRect(gesture_legend, 12, 12)
            painter.setPen(QColor(235, 253, 255))
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            painter.drawText(
                gesture_legend, Qt.AlignCenter,
                "LEFT HAND FIST: MODIFIER  ·  LEFT OR RIGHT HAND PINCH: CLICK  ·  GRAB & DRAG: SCROLL PAGE")

        if self.voice_selection_mode is not None and self.voice_lasso_start is not None:
            end_global = self._current_tli_global_position()
            start_local = self.mapFromGlobal(QPoint(round(self.voice_lasso_start.x()), round(self.voice_lasso_start.y())))
            end_local = self.mapFromGlobal(end_global)
            selection_rect = QRectF(start_local, end_local).normalized()
            painter.setPen(QPen(QColor(100, 230, 255, 235), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(40, 170, 220, 38)) if self.voice_selection_mode == "image" else Qt.NoBrush)
            if self.voice_selection_mode == "image":
                painter.drawRect(selection_rect)
                hint = "IMAGE LASSO · SAY 'COPY THIS'"
            else:
                painter.drawLine(start_local, end_local)
                hint = "TEXT SELECTION · SAY 'COPY THIS'"
            label = QRectF(selection_rect.left(), max(4, selection_rect.top()-28), 245, 24)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(5, 20, 33, 220)))
            painter.drawRoundedRect(label, 6, 6)
            painter.setPen(QColor(210, 250, 255))
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            painter.drawText(label.adjusted(7, 0, -5, 0), Qt.AlignVCenter | Qt.AlignLeft, hint)

        # Keep the active Tap Location Indicator above the Pinch Menu glass and buttons,
        # so users can always see precisely which item is under their hand. When Search's
        # topmost overlay owns the hand layer, it already paints this final TLI; repeating it
        # on the main window can create a fixed-centre ghost ring during menu gestures.
        if not search_overlay_draws_hand_layer:
            for side in ("right", "left"):
                if not self.fist_action_menu_open.get(side):
                    continue
                cursor = self.side_cursor_positions.get(side)
                if cursor is None:
                    continue
                cursor = self._visual_tli_position(side, cursor)
                color = QColor(self.left_click_color if side == "left" else self.right_click_color)
                engaged = side in self.active_push_sides or (self.click_side == side and now < self.click_flash_until)
                fill = QColor(color.red(), color.green(), color.blue(), 82)
                self._paint_tli(painter, cursor, side, engaged)

        # A fixed origin marker makes the centre point of recognised fist-drag scrolling
        # visible without following the hand/TLI. It remains briefly after release.
        for marker_side in ("right", "left"):
            marker = self.fist_scroll_markers.get(marker_side)
            if not marker:
                continue
            action = self.active_action.get(marker_side) or {}
            scrolling = action.get("mode") == "scroll_window"
            if not scrolling and now > float(marker.get("until", 0.0)):
                self.fist_scroll_markers[marker_side] = None
                continue
            point = marker.get("point")
            if point is None:
                continue
            fade = 1.0 if scrolling else max(0.0, min(1.0, (float(marker.get("until", 0.0))-now)/1.25))
            alpha = int(220 * fade)
            marker_color = QColor(105, 235, 255, alpha)
            painter.save()
            # Soft neutral backdrop behind the scroll anchor and both notch axes.
            # Keep it subtle and fade it with the marker so it doesn't obscure content.
            deadzone_mark = max(4.0, float(self.scroll_deadzone))
            notch_spacing = 72.0
            backdrop_radius = deadzone_mark + notch_spacing * 2.0 + 24.0
            # Backdrop is painted in the pre-pass so hands stay in front of it.
            painter.setPen(QPen(marker_color, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(point, 10.0, 10.0)
            painter.drawLine(QPointF(point.x()-15, point.y()), QPointF(point.x()-5, point.y()))
            painter.drawLine(QPointF(point.x()+5, point.y()), QPointF(point.x()+15, point.y()))
            painter.drawLine(QPointF(point.x(), point.y()-15), QPointF(point.x(), point.y()-5))
            painter.drawLine(QPointF(point.x(), point.y()+5), QPointF(point.x(), point.y()+15))
            # Show both control rulers simultaneously: up/down scrolling above/below
            # the anchor, and horizontal scrolling or Back/Forward at its sides.
            try:
                smoothed_dx = float(action.get("scroll_smoothed_dx", 0.0)) if scrolling else 0.0
                smoothed_dy = float(action.get("scroll_smoothed_dy", 0.0)) if scrolling else 0.0
            except (TypeError, ValueError):
                smoothed_dx = smoothed_dy = 0.0
            def notch_stage(delta):
                distance = abs(delta)
                if distance <= deadzone_mark:
                    return 0
                if distance < deadzone_mark + notch_spacing:
                    return 1
                if distance < deadzone_mark + notch_spacing * 2.0:
                    return 2
                return 3
            vertical_stage = notch_stage(smoothed_dy)
            horizontal_stage = notch_stage(smoothed_dx)
            horizontal_fallback = bool(action.get("horizontal_scroll_supported") is False) if scrolling else False
            for notch_index in range(3):
                offset = deadzone_mark + notch_spacing*notch_index
                half_width = 10.0 + notch_index*2.0
                # Vertical ruler: white means that notch's distance stage is active.
                vertical_active = vertical_stage == notch_index+1
                vertical_color = QColor(255, 255, 255, alpha) if vertical_active else QColor.fromHsv(int(self.scroll_notch_hue) % 360, 190, 255, alpha)
                if vertical_active:
                    glow = QColor(255, 255, 255, int(alpha * 0.42))
                    painter.setPen(QPen(glow, 11.0, Qt.SolidLine, Qt.RoundCap))
                    for sign in (-1.0, 1.0):
                        notch_y = point.y() + sign*offset
                        if notch_index == 2 and scrolling:
                            direction = sign
                            painter.drawLine(QPointF(point.x()-7, notch_y-direction*7),
                                             QPointF(point.x(), notch_y+direction*7))
                            painter.drawLine(QPointF(point.x(), notch_y+direction*7),
                                             QPointF(point.x()+7, notch_y-direction*7))
                        else:
                            painter.drawLine(QPointF(point.x()-half_width, notch_y), QPointF(point.x()+half_width, notch_y))
                painter.setPen(QPen(vertical_color, 4.0, Qt.SolidLine, Qt.RoundCap))
                for sign in (-1.0, 1.0):
                    notch_y = point.y() + sign*offset
                    if notch_index == 2 and scrolling:
                        direction = sign
                        painter.drawLine(QPointF(point.x()-7, notch_y-direction*7),
                                         QPointF(point.x(), notch_y+direction*7))
                        painter.drawLine(QPointF(point.x(), notch_y+direction*7),
                                         QPointF(point.x()+7, notch_y-direction*7))
                    else:
                        painter.drawLine(QPointF(point.x()-half_width, notch_y),
                                         QPointF(point.x()+half_width, notch_y))
                # Horizontal ruler: if the target has no horizontal scroll range, the
                # outer marks become left/right navigation arrows instead.
                horizontal_active = horizontal_stage == notch_index+1
                horizontal_color = QColor(255, 255, 255, alpha) if horizontal_active else QColor.fromHsv(int(self.scroll_notch_hue) % 360, 190, 255, alpha)
                if horizontal_active:
                    glow = QColor(255, 255, 255, int(alpha * 0.42))
                    painter.setPen(QPen(glow, 11.0, Qt.SolidLine, Qt.RoundCap))
                    for sign in (-1.0, 1.0):
                        notch_x = point.x() + sign*offset
                        if horizontal_fallback and notch_index == 2:
                            direction = sign * (-1.0 if self.swap_drag_scroll_notches else 1.0)
                            painter.drawLine(QPointF(notch_x-direction*7, point.y()-8), QPointF(notch_x+direction*7, point.y()))
                            painter.drawLine(QPointF(notch_x+direction*7, point.y()), QPointF(notch_x-direction*7, point.y()+8))
                        else:
                            painter.drawLine(QPointF(notch_x, point.y()-half_width), QPointF(notch_x, point.y()+half_width))
                painter.setPen(QPen(horizontal_color, 4.0, Qt.SolidLine, Qt.RoundCap))
                for sign in (-1.0, 1.0):
                    notch_x = point.x() + sign*offset
                    if horizontal_fallback and notch_index == 2:
                        direction = sign * (-1.0 if self.swap_drag_scroll_notches else 1.0)
                        painter.drawLine(QPointF(notch_x-direction*7, point.y()-8),
                                         QPointF(notch_x+direction*7, point.y()))
                        painter.drawLine(QPointF(notch_x+direction*7, point.y()),
                                         QPointF(notch_x-direction*7, point.y()+8))
                    else:
                        painter.drawLine(QPointF(notch_x, point.y()-half_width),
                                         QPointF(notch_x, point.y()+half_width))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(180, 250, 255, alpha)))
            painter.drawEllipse(point, 2.3, 2.3)
            painter.restore()

        # Small persistent on-state badges for dictation and speech recognition.
        # Their screen anchors are independently configurable on the Notifications page.
        search_dictation_active = bool(self.search_dialog is not None and
                                       self.search_dialog.isVisible() and
                                       self.search_dialog.dictation_enabled)
        dictation_active = bool(self.focus_dictation_enabled or search_dictation_active)
        active_indicators = []
        if dictation_active:
            active_indicators.append(("🎤>_", "dictation_indicator_position", QColor(104, 235, 180, 255)))
        if self.stt_enabled:
            active_indicators.append(("🎤", "stt_indicator_position", QColor(110, 220, 250, 255)))
        for indicator_text, position_key, accent in active_indicators:
            indicator_rect = self._notification_rect(
                self.ui_preferences.get(position_key, "top-left" if position_key == "dictation_indicator_position" else "top-right"),
                72 if indicator_text == "🎤" else 82, 28, margin=18)
            painter.save()
            # These are critical live-state indicators: always fully opaque, even
            # when the rest of the notification stack fades into the background.
            painter.setPen(QPen(accent, 1.2))
            bg = QColor(5, 18, 32, 205)
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(indicator_rect, 12, 12)
            painter.setPen(QColor(235, 252, 255, 250))
            painter.setFont(self._ui_font(8, QFont.DemiBold))
            painter.drawText(indicator_rect, Qt.AlignCenter, indicator_text)
            painter.restore()

        if self.stt_enabled:
            if self.heard_popup_mode == "unrecognised":
                transcript = self.heard_popup_message or "Unrecognised Command"
            else:
                transcript = self.speech_partial or (f"Heard: {self.last_speech}" if self.last_speech else "Listening locally…")
            pill_w = min(width-48, max(300, min(760, 18 + len(transcript)*8)))
            heard_position = self.ui_preferences.get("heard_bar_position", "bottom-center")
            pill = self._stacked_notification_rect(heard_position, pill_w, 34,
                                                   notification_stack_offsets, margin=18)
            now_mono = time.monotonic()
            if self.heard_popup_mode == "unrecognised" and now_mono < self.heard_popup_shake_until:
                # math is imported at module level. A local "import math" here made
                # math a local name for this whole method, so the window-drag sparkle
                # animation earlier in the frame raised UnboundLocalError mid-paint.
                shake = int(math.sin(now_mono * 48.0) * 4.0)
                pill.translate(shake, 0)
            painter.save()
            # The Heard / transcript pill is a live STT indicator: keep it fully opaque
            # while speech recognition is enabled, independent of notification fading.
            if self.heard_popup_mode == "unrecognised":
                painter.setPen(QPen(QColor(255, 86, 92, 245), 1.5))
                painter.setBrush(QBrush(QColor(74, 12, 20, 225)))
                painter.setPen(QColor(255, 225, 226))
            else:
                painter.setPen(QPen(QColor(115, 220, 235, 205), 1.2))
                painter.setBrush(QBrush(QColor(5, 18, 32, 148)))
                painter.setPen(QColor(224, 250, 255))
            painter.drawRoundedRect(pill, 14, 14)
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            painter.drawText(pill.adjusted(12, 0, -12, 0), Qt.AlignVCenter | Qt.AlignLeft, transcript)
            painter.restore()

        # Persistent per-hand status cards. Keep them near the screen edges so they
        # don't obscure the centred controls banner; the modifier badge occupies the
        # centre only while the left-fist modifier is active.
        notice_opacity = self._notification_stack_opacity(now)
        card_height = 58.0
        margin_x = 22.0
        side_width = min(285.0, max(170.0, width * 0.205))
        modifier_width = min(430.0, max(230.0, width * 0.34))
        # On narrower windows, shrink the three slots together to prevent overlap.
        available_width = max(0.0, width - 2.0 * margin_x)
        if 2.0 * side_width + modifier_width > available_width:
            shrink = available_width / max(1.0, 2.0 * side_width + modifier_width)
            side_width *= shrink
            modifier_width *= shrink
        panel_y = height - 76.0
        modifier_active = bool(self.left_click_modifier_active)
        # Keep bottom-centred timed notices above the persistent status row.
        notification_stack_offsets["bottom-center"] = max(
            float(notification_stack_offsets.get("bottom-center", 0.0)), 84.0)

        status_positions = {
            "left": self.ui_preferences.get("left_status_position", "bottom-left"),
            "right": self.ui_preferences.get("right_status_position", "bottom-right"),
        }
        for side in ("left", "right"):
            status_position = status_positions[side]
            card = self._notification_rect(status_position, side_width, card_height, margin=margin_x)
            detached = bool(self.mouse_detached_by_side.get(side, False))
            gestures_on = bool(self.gesture_tracking_enabled.get(side, True))
            fist_active = bool(self.fist_started_at.get(side) is not None or
                               self.fist_hold_armed.get(side, False) or
                               ((self.active_action.get(side) or {}).get("mode") == "scroll_window"))
            signature = (detached, gestures_on, fist_active)
            if signature != self._status_card_signatures.get(side):
                self._status_card_signatures[side] = signature
                self._status_card_full_until[side] = now + 5.0
            # A content change briefly restores full brightness; the left card also
            # stays bright while its fist modifier is active.
            card_opacity = 1.0 if now < self._status_card_full_until.get(side, 0.0) else notice_opacity
            if side == "left" and modifier_active:
                card_opacity = 1.0
            if side == "right" and fist_active:
                card_opacity = 1.0

            painter.save()
            painter.setOpacity(painter.opacity() * card_opacity)
            painter.setPen(QPen(QColor(105, 220, 240, 190), 1.3))
            painter.setBrush(QBrush(QColor(5, 18, 32, 170)))
            painter.drawRoundedRect(card, 11, 11)
            painter.setPen(QColor(145, 225, 245, 235))
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            painter.drawText(QRectF(card.left()+10, card.top()+4, card.width()-20, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, f"{side.upper()}")
            mouse_label = "MOUSE DETACHED" if detached else "MOUSE ATTACHED"
            gesture_label = "GESTURES DISABLED" if not gestures_on else "GESTURES ENABLED"
            mouse_color = QColor(255, 205, 125, 245) if detached else QColor(165, 245, 195, 245)
            gesture_color = QColor(255, 205, 125, 245) if not gestures_on else QColor(165, 245, 195, 245)
            painter.setFont(self._ui_font(8, QFont.DemiBold))
            painter.setPen(mouse_color)
            painter.drawText(QRectF(card.left()+10, card.top()+21, card.width()-20, 14),
                             Qt.AlignLeft | Qt.AlignVCenter, mouse_label)
            painter.setPen(gesture_color)
            painter.drawText(QRectF(card.left()+10, card.top()+38, card.width()-20, 14),
                             Qt.AlignLeft | Qt.AlignVCenter, gesture_label)
            painter.restore()

        # Modifier is entirely hidden when inactive. While active, keep it fully lit.
        if modifier_active and modifier_width > 0:
            modifier_badge = QRectF((width-modifier_width)/2.0, height-76.0,
                                    modifier_width, card_height)
            painter.save()
            painter.setOpacity(painter.opacity())
            painter.setPen(QPen(QColor(100, 235, 250, 225), 1.5))
            painter.setBrush(QBrush(QColor(4, 22, 36, 220)))
            painter.drawRoundedRect(modifier_badge, 11, 11)
            painter.setPen(QColor(220, 252, 255, 255))
            modifier_text = "LEFT FIST MODIFIER ACTIVE · PINCH: RIGHT CLICK · GRIP: DRAG"
            # Use the shared UI font scale, and fit the longer status line inside its
            # badge so it scales with the rest of the app without clipping.
            self._draw_fitted_button_label(
                painter, modifier_badge, modifier_text,
                base_size=10, min_size=6, inset=7)
            painter.restore()

        if self.hand_tracking_paused:
            painter.save()
            painter.setOpacity(painter.opacity()*notification_stack_opacity)
            pause_position = self.ui_preferences.get("paused_notice_position", "bottom-center")
            pause_rect = self._stacked_notification_rect(
                pause_position, 368, 38, notification_stack_offsets, margin=18)
            painter.setPen(QPen(QColor(255, 190, 92, 215), 1.5))
            painter.setBrush(QBrush(QColor(38, 27, 12, 158)))
            painter.drawRoundedRect(pause_rect, 12, 12)
            painter.setPen(QColor(255, 222, 165))
            painter.setFont(self._ui_font(10, QFont.DemiBold))
            painter.drawText(pause_rect, Qt.AlignCenter, "HAND TRACKING PAUSED  ·  SAY 'RESUME TRACKING'")
            painter.restore()

        if self.desktop_mode == "virtual":
            painter.setPen(QColor(150, 190, 205, 190))
            painter.setFont(self._ui_font(10))
            painter.drawText(22, 28, "SPATIAL DESKTOP  ·  FLOATING CANVAS")
        # Apply the same responsive cyan hover treatment used by the glass-menu tiles
        # to every Settings navigation tab, Back button, and page button. Outline-only
        # painting keeps labels and the existing button hit targets unchanged.
        if getattr(self, "menu_open", False) and getattr(self, "settings_open", False):
            scale = self.menu_ui_scale()
            # The glass panel can be moved, so feedback geometry must use the same
            # hub anchor as the actual controls rather than the screen centre.
            cx, cy = self.hub_pos.x()*self.width(), self.hub_pos.y()*self.height()
            hovered_settings_rects = [self._settings_back_rect(cx, cy, scale)]
            if self.settings_page not in ("gesture_map", "speech_commands"):
                hovered_settings_rects.extend(self._settings_tab_rects(cx, cy, scale).values())
            page = self.settings_page
            if page not in ("gesture_map", "speech_commands"):
                # Same rectangles the page paints and hit-tests (see _settings_layout).
                hovered_settings_rects.extend(self._settings_control_rects(cx, cy, scale).values())
            for rect in hovered_settings_rects:
                if any(point is not None and rect.contains(point)
                       for point in self.side_cursor_positions.values()):
                    for expansion, alpha in ((7*scale, 38), (3.5*scale, 78)):
                        painter.setPen(QPen(QColor(100, 255, 255, alpha), 3*scale))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRoundedRect(rect.adjusted(-expansion, -expansion, expansion, expansion),
                                                12*scale+expansion, 12*scale+expansion)
                    painter.setPen(QPen(QColor(220, 255, 255, 250), 2.0*scale))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(rect, 10*scale, 10*scale)

        # Brief cyan pressed glow for every custom-painted glass-menu/settings button.
        pressed_point = getattr(self, "_glass_pressed_point", None)
        pressed_age = now - float(getattr(self, "_glass_pressed_at", 0.0))
        if pressed_point is not None and 0.0 <= pressed_age < 0.30 and self.menu_open and self.hub_visible:
            # Match pressed feedback to the live menu position, including when
            # the user has dragged the glass panel away from screen centre.
            pcx, pcy = self.hub_pos.x()*self.width(), self.hub_pos.y()*self.height()
            pscale = self.menu_ui_scale()
            press_rects = []
            if not self.settings_open:
                press_rects.append(self._main_menu_close_rect(pcx, pcy, pscale))
                press_rects.extend([QRectF(pcx-166*pscale, pcy-145*pscale, 100*pscale, 88*pscale),
                    QRectF(pcx-50*pscale, pcy-145*pscale, 100*pscale, 88*pscale),
                    QRectF(pcx+66*pscale, pcy-145*pscale, 100*pscale, 88*pscale),
                    QRectF(pcx-166*pscale, pcy+35*pscale, 100*pscale, 88*pscale),
                    QRectF(pcx-50*pscale, pcy+35*pscale, 100*pscale, 88*pscale),
                    QRectF(pcx+66*pscale, pcy+35*pscale, 100*pscale, 88*pscale)])
            else:
                press_rects.append(self._settings_back_rect(pcx, pcy, pscale))
                if self.settings_page not in ("gesture_map", "speech_commands"):
                    press_rects.extend(self._settings_tab_rects(pcx, pcy, pscale).values())
                page = self.settings_page
                if page not in ("gesture_map", "speech_commands"):
                    press_rects.extend(self._settings_control_rects(pcx, pcy, pscale).values())
                if page == "gestures":
                    layout = self._custom_gesture_layout(pcx, pcy, pscale)
                    press_rects.extend([layout["record_left"], layout["record_right"]])
                    for row in layout["rows"]:
                        press_rects.extend([row["row"], row["side"], row["timer"], row["label"], row["action"], row["delete"]])
            pressed_rect = next((r for r in press_rects if r.contains(pressed_point)), None)
            if pressed_rect is not None:
                fade_alpha = int(235 * (1.0 - pressed_age/0.30))
                painter.save()
                painter.setPen(QPen(QColor(190, 255, 255, fade_alpha), 2.5*pscale))
                painter.setBrush(QBrush(QColor(90, 235, 255, int(45*(1.0-pressed_age/0.30)))))
                painter.drawRoundedRect(pressed_rect.adjusted(1.5*pscale, 1.5*pscale, -1.5*pscale, -1.5*pscale),
                                        12*pscale, 12*pscale)
                painter.restore()

        painter.end()


def main():
    ensure_models()
    app = QApplication(sys.argv)
    # Shared tactile pressed feedback for every native button in the application.
    app.setStyleSheet("QPushButton:pressed, QToolButton:pressed { background-color: rgba(75, 205, 235, 210); border: 2px solid #b9f5ff; padding-top: 2px; padding-left: 2px; }")
    # The main overlay uses Qt.Tool and the search launcher is a child QDialog;
    # closing the launcher must never terminate the entire Spatial Desktop process.
    app.setQuitOnLastWindowClosed(False)
    window = SpatialDesktop()
    window.showFullScreen()
    window._sync_linux_fullscreen_geometry()
    # Some XLibre/Qtile configurations finalize the native geometry just after the
    # first show; repeat once the first event-loop pass has applied the fullscreen state.
    QTimer.singleShot(0, window._sync_linux_fullscreen_geometry)
    QTimer.singleShot(250, window._sync_linux_fullscreen_geometry)
    window.apply_desktop_mode()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        # On each fresh app launch, clean up a leftover app-managed llama-server
        # before requesting a new runtime. Ordinary retries do not force a restart.
        start_d1_loading(fresh=True)
    except Exception:
        pass
    main()
