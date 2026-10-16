#!/usr/bin/env python3
"""Spatial Desktop — hand input, offline voice, local search and optional face welcome (v61).

The webcam supplies hand landmarks. TPI calibration is hand-relative (not camera-depth
calibration): it stores a pinch point in a palm-local coordinate frame. Clicks, drag/scroll,
right-click, and menu toggle are recognised by pose transitions and timed holds.
"""
from __future__ import annotations

import sys
import os
import json
import base64
import time
import math
import random
import threading
import queue
import urllib.request
import re
import ctypes
import subprocess
import getpass
import shutil
import hashlib
import tempfile
import uuid
from datetime import datetime
from ctypes import wintypes
from pathlib import Path
from linux_screen_geometry import resolve_fullscreen_bounds
from garden_weather import fetch_live_weather, calculate_solar_state, weather_kind, WEATHER_REFRESH_SECONDS

from gesture_rules import (
    should_dismiss_pinch_menu,
    detect_fist_shake,
    should_start_right_fist_scroll,
    active_fist_scroll_release_confirmed,
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
    thumb_axis_is_vertical_up,
    thumb_middle_pinch_pose,
    opposite_diagonal_corner_assignment,
    classify_middle_pinch_window_action,
    resize_rect_from_corner_deltas,
    resize_rect_from_distance_scale,
)
from voice_services import Speech, KokoroTTS
from hand_occlusion import TuckSmoother
from hand_contact import contact_offsets as hand_contact_offsets
from custom_gestures import (
    ACTION_IDS as CUSTOM_GESTURE_ACTION_IDS, ACTION_LABELS as CUSTOM_GESTURE_ACTION_LABELS,
    ACTION_GROUPS as CUSTOM_GESTURE_ACTION_GROUPS,
    MAX_GESTURES as CUSTOM_GESTURE_MAX, INSTANT_HOLD_SECONDS as CUSTOM_GESTURE_INSTANT_HOLD, HoldTrigger, hand_features, mean_features, describe as describe_pose,
    builtin_conflict, pinch_ratio as custom_pinch_ratio, best_pose_match, tli_overlap,
    load_custom_gestures, save_custom_gestures, pyautogui_key_combo, display_key_combo,
    WindowsShortcutCapture, X11ShortcutCapture,
)
from voice_pc_commands import match_pc_command, aliases_for_help, parse_navigation_command, is_navigation_intent, has_explicit_create_prefix
from voice_assistant_commands import live_answer, file_query
from d1_decision import (classify_voice_command, start_loading as start_d1_loading,
                         selected_model as selected_d1_model, select_model as select_d1_model,
                         status as d1_status, MODELS as D1_MODELS, stop_server as stop_d1_server,
                         start_loading as start_d1_server)
from search_launcher import SearchLauncherDialog
from face_welcome import FaceWelcomeService
from tli_calibration import encode_relative_tli, resolve_relative_tli, mean_relative_tli
from tpi_screen_calibration import (
    TARGET_COUNT as TPI_CALIBRATION_TARGET_COUNT, TARGET_NAMES as TPI_CALIBRATION_TARGET_NAMES,
    TARGET_FRACTIONS as TPI_CALIBRATION_TARGET_FRACTIONS,
    load_calibrations as load_tpi_screen_calibrations,
    save_calibrations as save_tpi_screen_calibrations,
    calibration_from_samples as make_tpi_screen_calibration,
    map_point as map_tpi_screen_point,
)
from updater import VERSION_FILE, compare_versions, read_version, fetch_remote_version, download_update, install_update
from windows_virtual_desktop import pin_window_on_all_virtual_desktops
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

from PySide6.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QRectF, QObject, Signal, QMimeData, QBuffer, QIODevice
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetricsF, QCursor, QLinearGradient, QImage, QPainterPath, QPainterPathStroker, QRadialGradient, QPolygonF, QDrag, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QInputDialog

from user_paths import user_data_dir, user_file

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
USER_DATA_DIR = user_data_dir()
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
# User-owned state lives outside the install directory so application updates cannot overwrite it.
GESTURES_FILE = user_file("gestures.txt", ROOT)
UI_SETTINGS_FILE = user_file("ui_settings.json", ROOT)
TLI_CALIBRATION_FILE = user_file("tli_calibration.json", ROOT)  # Legacy palm-local TLI anchor calibration
TPI_SCREEN_CALIBRATION_FILE = user_file("tpi_screen_calibration.json", ROOT)
HAND_CALIBRATION_FILE = user_file("hand_calibration.json", ROOT)
CUSTOM_GESTURES_FILE = user_file("custom_gestures.json", ROOT)
VIRTUAL_DESKTOP_DATA_FILE = user_file("virtual_desktop.json", ROOT)
VIRTUAL_BACKGROUND_STYLES = (
    {"name": "MIDNIGHT BLUE", "background": "#080c16", "grid": "#4b9be8"},
    {"name": "NEON PURPLE", "background": "#10091b", "grid": "#a26bff"},
    {"name": "DEEP TEAL", "background": "#061719", "grid": "#37d9cd"},
    {"name": "AMBER GRID", "background": "#171108", "grid": "#f4b95f"},
)
VIRTUAL_PEN_PALETTE = (
    ("WHITE", "#f7fbff"), ("CYAN", "#43ddff"), ("RED", "#ff555d"),
    ("ORANGE", "#ffad45"), ("GREEN", "#65e58d"), ("PURPLE", "#c58bff"),
    ("YELLOW", "#ffe66b"),
)
VIRTUAL_SCENES = ("grid", "garden", "space")
VIRTUAL_SCENE_LABELS = {"grid": "GRID", "garden": "GARDEN", "space": "SPACE"}
VIRTUAL_DESKTOP_DEFAULTS = {
    "background": {
        "grid_enabled": True, "garden_mode": False, "scene": "grid", "style": 0, "brightness": 78.0,
        "glow": 55.0, "spacing": 64.0, "thickness": 1.25,
    },
    "strokes": [],
}


def _finite_clamped(value, low, high, default):
    try:
        number = float(value)
        if not math.isfinite(number):
            number = float(default)
    except (TypeError, ValueError):
        number = float(default)
    return min(float(high), max(float(low), number))


def normalise_virtual_desktop_data(value):
    """Validate persistent grid settings and normalized pen strokes."""
    source = value if isinstance(value, dict) else {}
    background_source = source.get("background", {})
    if not isinstance(background_source, dict):
        background_source = {}
    defaults = VIRTUAL_DESKTOP_DEFAULTS["background"]
    try:
        style = int(background_source.get("style", defaults["style"]))
    except (TypeError, ValueError):
        style = defaults["style"]
    # Older saves stored only garden_mode; never lose the selected Garden scene.
    raw_scene = background_source.get("scene")
    scene = (raw_scene if raw_scene in VIRTUAL_SCENES else
             "garden" if background_source.get("garden_mode") is True else "grid")
    background = {
        "scene": scene,
        "grid_enabled": (background_source.get("grid_enabled")
                         if isinstance(background_source.get("grid_enabled"), bool)
                         else defaults["grid_enabled"]),
        "garden_mode": (background_source.get("garden_mode")
                        if isinstance(background_source.get("garden_mode"), bool)
                        else defaults["garden_mode"]),
        "style": max(0, min(len(VIRTUAL_BACKGROUND_STYLES)-1, style)),
        "brightness": _finite_clamped(background_source.get("brightness"), 10, 100, defaults["brightness"]),
        "glow": _finite_clamped(background_source.get("glow"), 0, 100, defaults["glow"]),
        "spacing": _finite_clamped(background_source.get("spacing"), 24, 160, defaults["spacing"]),
        "thickness": _finite_clamped(background_source.get("thickness"), 0.5, 3.0, defaults["thickness"]),
    }
    strokes = []
    raw_strokes = source.get("strokes", [])
    total_points = 0
    if isinstance(raw_strokes, list):
        for raw in raw_strokes[-2000:]:
            if not isinstance(raw, dict):
                continue
            erase = bool(raw.get("erase", False))
            color = str(raw.get("color", "#43ddff"))
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = "#43ddff"
            raw_points = raw.get("points", [])
            if not isinstance(raw_points, list):
                continue
            points = []
            for point in raw_points[:5000]:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                points.append([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
            if not points:
                continue
            if total_points + len(points) > 150000:
                break
            total_points += len(points)
            strokes.append({
                "color": color.lower(), "erase": erase,
                "width": _finite_clamped(raw.get("width"), 0.0005, 0.06,
                                          0.018 if erase else 0.0038),
                "points": points,
            })
    constellations = []
    for item in source.get("constellations", []) if isinstance(source.get("constellations"), list) else []:
        if not isinstance(item, dict):
            continue
        raw_stars = item.get("stars", [])
        if not isinstance(raw_stars, list):
            continue
        indices = []
        for star in raw_stars[:72]:
            if type(star) is int and 0 <= star < 120 and (not indices or star != indices[-1]):
                indices.append(star)
        if len(indices) >= 2:
            constellations.append({"stars": indices})
        if len(constellations) >= 120:
            break
    return {"background": background, "strokes": strokes, "constellations": constellations}


def load_virtual_desktop_data(path=VIRTUAL_DESKTOP_DATA_FILE):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raw = {}
    return normalise_virtual_desktop_data(raw)


def save_virtual_desktop_data(value, path=VIRTUAL_DESKTOP_DATA_FILE):
    clean = normalise_virtual_desktop_data(value)
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return True
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        print("Could not save virtual desktop:", exc)
        return False


NOTIFICATION_POSITIONS = ("top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right")
SOUND_SETTING_ROWS = (
    ("sound_pinch", "PINCH"),
    ("sound_pinch_menu", "PINCH MENU"),
    ("sound_gesture_timer_during", "GESTURE TIMER DURING"),
    ("sound_gesture_timer", "GESTURE TIMER COMPLETION"),
    ("sound_glass_buttons", "GLASS MENU BUTTONS"),
    ("sound_glass_toggle", "GLASS MENU TOGGLE"),
)
SOUND_EXTENSIONS = {".wav", ".ogg", ".mp3", ".flac", ".aiff", ".aif"}

def available_ui_sounds():
    folder = ROOT / "assets" / "sounds"
    try:
        return [p.name for p in sorted(folder.iterdir(), key=lambda item: item.name.casefold())
                if p.is_file() and p.suffix.lower() in SOUND_EXTENSIONS]
    except OSError:
        return []

def play_ui_sound(filename, *, stop_event=None):
    """Play bundled UI audio. An optional event lets gesture timers cancel it early."""
    if not filename or filename == "None":
        return
    folder = (ROOT / "assets" / "sounds").resolve()
    path = (folder / str(filename)).resolve()
    if path.parent != folder or not path.is_file() or path.suffix.lower() not in SOUND_EXTENSIONS:
        return
    try:
        if os.name == "nt" and path.suffix.lower() in {".mp3", ".wav"}:
            # Windows does not normally ship with paplay/aplay/ffplay. Use its
            # native MCI playback asynchronously so gesture recognition never
            # blocks while the chosen timer sound plays.
            def _play_windows_mci():
                alias = "spatial_ui_" + uuid.uuid4().hex[:12]
                mci = ctypes.windll.winmm.mciSendStringW
                try:
                    media_type = "mpegvideo" if path.suffix.lower() == ".mp3" else "waveaudio"
                    if mci(f'open "{path}" type {media_type} alias {alias}', None, 0, None) != 0:
                        return
                    if stop_event is None:
                        mci(f"play {alias} wait", None, 0, None)
                    else:
                        # MCI `play ... wait` cannot be interrupted safely by the UI.
                        # Use nonblocking playback and poll the stop event instead.
                        if stop_event.is_set():
                            return
                        if mci(f"play {alias}", None, 0, None) != 0:
                            return
                        while not stop_event.wait(0.035):
                            mode = ctypes.create_unicode_buffer(32)
                            if mci(f"status {alias} mode", mode, len(mode), None) != 0:
                                break
                            if mode.value.strip().lower() != "playing":
                                break
                        if stop_event.is_set():
                            mci(f"stop {alias}", None, 0, None)
                except (OSError, ValueError, AttributeError):
                    pass
                finally:
                    try:
                        mci(f"close {alias}", None, 0, None)
                    except (OSError, ValueError, AttributeError):
                        pass
            threading.Thread(target=_play_windows_mci, daemon=True).start()
        else:
            command = None
            if shutil.which("paplay"):
                command = ["paplay", str(path)]
            elif shutil.which("aplay") and path.suffix.lower() == ".wav":
                command = ["aplay", "-q", str(path)]
            elif shutil.which("ffplay"):
                command = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
            if command is not None and (stop_event is None or not stop_event.is_set()):
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if stop_event is not None:
                    def _stop_process_on_cancel():
                        # Wait for exit *or* cancellation without blocking the scene timer.
                        while process.poll() is None:
                            if stop_event.wait(0.035):
                                if process.poll() is None:
                                    try:
                                        process.terminate()
                                    except OSError:
                                        pass
                                break
                    threading.Thread(target=_stop_process_on_cancel, daemon=True).start()
    except (OSError, ValueError):
        pass
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
        "show_gesture_disabled_indicator": True,
        **{key: "None" for key, _label in SOUND_SETTING_ROWS},
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
            elif key == "show_gesture_disabled_indicator":
                if isinstance(value, bool):
                    result[key] = value
            elif key in dict(SOUND_SETTING_ROWS):
                allowed = {"None", *available_ui_sounds()}
                if value in allowed:
                    result[key] = value
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
        elif key == "show_gesture_disabled_indicator":
            if isinstance(value, bool):
                current[key] = value
        elif key in dict(SOUND_SETTING_ROWS):
            allowed = {"None", *available_ui_sounds()}
            if value in allowed:
                current[key] = value
        elif value in NOTIFICATION_POSITIONS:
            current[key] = value
    try:
        UI_SETTINGS_FILE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print("Could not save UI preferences:", exc)


def load_tli_calibrations() -> dict:
    """Load optional palm-relative TPI coordinates for each physical hand."""
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
        print("Could not save TPI calibration:", exc)
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
        "# POINTER/TPI: saved palm-relative calibration point when calibrated; otherwise thumb/index fingertip midpoint.",
        "# QUICK CLICK: quick fist/grip -> open OR quick thumb/index pinch -> release clicks at the starting TPI point.",
        "# LEFT MODIFIER: hold LEFT fist; quick RIGHT pinch/release = right-click/context menu; held RIGHT grip = drag a native window (OS) or virtual canvas/app (3D).",
        "# Modifier gestures suppress the regular Pinch Menu timer; open RIGHT hand to release a modifier drag.",
        "# DOUBLE-CLICK: two quick thumb/index pinches near the same point are snapped to one target for a native double-click.",
        "# PINCH MENU: hold thumb/index pinch continuously for pinch_buffer_seconds (default 0.8s); then the circle fills for the hold timer.",
        "# PINCH MENU: while open, a fresh pinch/grab outside the popup closes it without clicking through.",
        "# TWO-HAND PINCH ZOOM: pinch thumb to middle fingertip with BOTH hands and hold through pinch_buffer_seconds; move hands apart/together to zoom app content with Ctrl/Command+wheel.",
        "# WINDOW RESIZE: begin the same two-hand pinch with TPIs at opposite diagonal corners of one application window; keep pinching and move the corners to resize it.",
        "# FIST MOVEMENT: a moving RIGHT-only fist starts smooth vertical page scrolling without a fist timer; left fist remains the modifier, and two-fist window drag remains available.",
        "# TIMERS: Pinch Menu, two-hand thumb-middle pinch zoom, and peace-sign gestures show countdowns. Right-fist scrolling has no circle timer.",
        "# A release before 0.8s cancels that timer sequence. Quick/double pinches start a fresh buffer; after confirmation only held-pinch time counts.",
        "# A fist/grip has priority over pinch recognition; a fist alone cannot start the Pinch Menu timer.",
        "# PEACE SIGN: hold an index+middle V-sign for 1.5 seconds to detach/reattach that hand from native OS mouse movement; glass and Pinch Menu gestures and live TPI tracking remain active.",
        "# A detached hand cannot drag-scroll; reattach it to scroll with a fist.",
        "# FIST: fingers closed AND thumb tucked in. With the thumb sticking out it is not a fist (thumbs up/down are free for custom gestures).",
        "# TPI OVERLAP: bring both TPIs together and hold for the gesture hold time to toggle the glass menu.",
        "# MODES: select Right-click or Scroll Window in the Pinch Menu; a timed pinch while a mode is active returns to normal.",
        "# FIST: a quick fist-open clicks. A RIGHT-only fist starts scrolling when moved vertically over page content; make both fists for the separate two-hand window drag.",
        "# SCROLL: RIGHT-only fist + deliberate vertical movement over page content starts gentle scrolling immediately; there is no right-fist circle timer.",
        "# EDGE INPUT: gesture recognition is slightly more tolerant near screen edges; skeleton joints are not individually clamped.",
        "# RELEASE COOLDOWN: one-hand fist scrolling requires a clear open hand for 0.48 seconds; other active drag/scroll actions retain the 0.20-second release. Timed pinches need a continuous 0.8-second confirmation hold.",
        "# SLIDER SNAP: fist OR thumb/index pinch near a settings slider snaps the TPI to its knob and starts dragging; no hold timer.",
        "# SLIDER STABILITY: isolated large cursor jumps are ignored; a new slider cursor position must persist for three frames.",
        "# HAND DETECTION: conservative MediaPipe confidence, three consistent detections, no cached-landmark display; side-on geometry requires high confidence and depth support.",
        "# PHYSICAL SIDES: camera handedness labels are normalized at the tracker boundary before cursor priority and colour assignment.",
        "# GLASS UI: menu and Settings panels use stronger Gaussian backdrop blur, a darker frosted tint, and soft halo edges.",
        "# PINCH MENU ITEMS: RIGHT-CLICK MODE, SCROLL WINDOW, and GLASS MENU. Grab or pinch an item to activate it.",
        "# RIGHT-CLICK MODE makes quick fist-open grabs right-click at the original TPI point.",
        "# DRAG WINDOW: left fist + right grip drags a native window in OS mode or the virtual canvas/app in 3D mode; open the right hand to release.",
        "# NORMAL TWO-HAND DRAG: make both hands fists; keep one still and move the other. Opening either releases.",
        "# SCROLL WINDOW mode makes a held fist scroll vertically inside the target window. A timed pinch exits the mode.",
        "# CAMERA OVERSCAN: the central 60% × 68% camera area maps to the full screen; outside it clamps to the nearest edge.",
        "# All gesture recognition uses hand shape and time. Relative camera depth is not used for activation.",
        "#",
        "# SETTINGS",
        "# hand_visual_scale: skeleton/TPI scale, 0.1 to 1.0.",
        "# hand_display_fps: visual hand refresh cap, 5 to 30 FPS; default 15. Hand detection runs separately.",
        "# hand_contact_band_ratio: palm attraction distance beyond touch, in palm widths; 0.0 to 0.9; default 0.02.",
        "# palm_attract_acceleration: speed multiplier for the virtual hands blending toward shared contact alignment; 0.2 to 5.0x; default 0.33x.",
        "# skin_tone_darkness: 0 to 100 percent extra warm darkening of the skin-like hand mesh.",
        "# finger_thickness: rounded finger segment thickness multiplier, 0.8 to 10.0; default 4.24.",
        "# palm_width_scale: palm contour width multiplier, 0.65 to 1.80; default 1.26.",
        "# hand_render_style: skeleton (lines), flat (painted 2D hand) or 3d (rigged 3D hand mesh).",
        "# hand_skin_effect: kept for older versions; true unless hand_render_style is skeleton.",
        "# TPI calibration: Hand Settings -> Calibrate Left/Right TPI; the live hand and transformed TPI are redrawn above the target and hit-tested at the same point.",
        "# Each TPI calibration stores the pinch midpoint in normalized palm-local 3D coordinates in tli_calibration.json.",
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
        "# tli_glow_intensity: TPI halo intensity multiplier, 0.0 to 2.5.",
        "# tli_colour_shift_degrees: hue shift applied to both TPI colours, 0 to 360 degrees.",
        "# tli_border_thickness: TPI outline thickness, 1.5 to 8.0 pixels.",
        "# tli_diameter: TPI outer diameter, 12 to 96 pixels; default 30 pixels.",
        "# skeleton_line_thickness: skeleton stroke multiplier, 0.5 to 4.0; default 1.5.",
        "# click_debounce_seconds: minimum fist duration accepted as a click, 0.04 to 0.25 seconds.",
        "# default_gesture_hold_seconds: shared activation duration for peace sign, thumbs up, and Pinch Menu, 0.5 to 5.0 seconds; default 0.5.",
        "# invert_scroll: true reverses vertical scroll direction; false is normal.",
        "# swap_drag_scroll_notches: true swaps the drag-scroll Back/Forward navigation notch directions.",
        "# cursor_lock_priority: right or left; selected physical hand gets first refusal for new gestures when both are visible.",
        "# Quick pinch clicks require a discrete short pulse before the 0.8s timer buffer; two quick pinches at one point form a native double-click.",
        "# Settings slider drag uses a reduced gain (0.42x) for finer control; initial pickup preserves the value.",
        "# Default TPI colours: LEFT red, RIGHT blue. Swatches can still cycle independently.",
        f"hand_visual_scale={clamped('hand_visual_scale', 0.1, 1.0, 0.3):.2f}",
        f"hand_display_fps={clamped('hand_display_fps', 5.0, 30.0, 15.0):.0f}",
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
        "hand_display_fps": 15.0,
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
        "hand_display_fps": (5.0, 30.0),
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
    """Require a clearly curled four-finger grip before scrolling or dragging."""
    if landmarks is None or len(landmarks) < 21:
        return False
    straight = _finger_straightness_values(landmarks)
    # Three curled fingers plus a half-open index was triggering scroll while
    # pointing or pinching. Require all four to curl, including at screen edges.
    index_limit = 0.79 if edge_tolerant else 0.77
    other_limit = 0.83 if edge_tolerant else 0.81
    fingers_closed = straight[0] <= index_limit and all(
        value <= other_limit for value in straight[1:])
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


def is_thumb_middle_zoom_pinch(landmarks, edge_tolerant=False):
    """True for a thumb-to-middle fingertip pinch, distinct from the Pinch Menu."""
    if landmarks is None or len(landmarks) < 21:
        return False
    palm = max(1e-5, distance3d(landmarks, 0, 9))
    middle_gap = distance3d(landmarks, 4, 12) / palm
    index_gap = distance3d(landmarks, 4, 8) / palm
    return thumb_middle_pinch_pose(
        normalized_middle_gap=middle_gap,
        normalized_index_gap=index_gap,
        finger_straightness=_finger_straightness_values(landmarks),
        edge_tolerant=edge_tolerant,
    )


class _HandVisualMotionFilter:
    """Suppress single-frame joint distortion without delaying whole-hand movement.

    This filter operates ONLY on the painted hand. Raw MediaPipe landmarks are kept
    for gesture recognition, hand contact/scale and the calibrated TLI. The previous
    hand is first carried along with the present palm's translation, rotation and
    scale, then each joint is eased toward its new measurement. A bad fingertip
    therefore cannot drag the rest of the hand into a twisted pose.
    """

    _PALM = (0, 5, 9, 13, 17)
    _CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
               (13, 14, 15, 16), (17, 18, 19, 20))
    _Y_TO_X = 9.0 / 16.0

    def __init__(self):
        self._previous = {}

    def update(self, side, landmarks, now):
        """Return 21 stabilised (x, y, z) tuples, keeping the wrist live."""
        if side not in ('left', 'right') or len(landmarks) < 21:
            return list(landmarks)
        try:
            current = np.array([(float(p[0]), float(p[1])*self._Y_TO_X,
                                 float(p[2]) if len(p) > 2 else 0.0)
                                for p in landmarks[:21]], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            self._previous.pop(side, None)
            return list(landmarks)
        if not np.isfinite(current).all():
            self._previous.pop(side, None)
            return list(landmarks)

        previous_entry = self._previous.get(side)
        if previous_entry is None or not 0.0 < float(now)-previous_entry[1] < 0.30:
            output = current
        else:
            previous, last_time = previous_entry
            dt = min(0.10, max(0.001, float(now)-last_time))
            # Carry the previous shape with the *current palm* so rapid sweeping,
            # turning or moving toward the camera does not twist the fingers.
            a = previous[list(self._PALM), :2]
            b = current[list(self._PALM), :2]
            ac = a.mean(axis=0)
            bc = b.mean(axis=0)
            aa = a-ac
            bb = b-bc
            spread = float(np.sum(aa*aa))
            if spread <= 1e-9:
                output = current
            else:
                u, singular, vt = np.linalg.svd(aa.T @ bb)
                correction = np.eye(2)
                if np.linalg.det(u @ vt) < 0:
                    correction[1, 1] = -1.0
                rotation = u @ correction @ vt
                ratio = float(np.sum(singular * np.diag(correction)) / spread)
                ratio = max(0.68, min(1.48, ratio))
                predicted = previous.copy()
                predicted[:, :2] = (previous[:, :2]-ac) @ rotation * ratio + bc
                predicted[:, 2] = (previous[:, 2]-previous[0, 2])*ratio + current[0, 2]

                palm_size = max(0.009, float(np.linalg.norm(current[9, :2]-current[0, :2])),
                                float(np.linalg.norm(current[5, :2]-current[17, :2]))*0.82)
                # Follow genuine articulation quickly but limit one-frame spikes.
                response = 1.0-math.exp(-25.0*dt)
                difference = current-predicted
                # Palm joints should remain mutually coherent; distal joints can
                # change farther in a frame when the person opens/closes a hand.
                for index in range(21):
                    reach = palm_size * (0.27 if index in self._PALM else 0.57)
                    xy_len = float(np.linalg.norm(difference[index, :2]))
                    if xy_len > reach:
                        difference[index, :2] *= reach/xy_len
                    z_limit = palm_size * (0.32 if index in self._PALM else 0.68)
                    difference[index, 2] = max(-z_limit, min(z_limit, difference[index, 2]))
                output = predicted + response*difference

                # Guard impossible one-frame bone stretching. This is generous
                # enough for foreshortening and naturally opening the fingers.
                for chain in self._CHAINS:
                    for parent, child in zip(chain, chain[1:]):
                        old_length = float(np.linalg.norm(previous[child]-previous[parent]))
                        new_bone = output[child]-output[parent]
                        new_length = float(np.linalg.norm(new_bone))
                        limit = max(0.27*palm_size, old_length*ratio*1.75)
                        if new_length > limit and new_length > 1e-9:
                            output[child] = output[parent] + new_bone*(limit/new_length)

                # No hand-position lag: anchor the *whole visual shape* to the
                # current wrist, matching the cursor's existing draw offset.
                output += current[0]-output[0]

        self._previous[side] = (output.copy(), float(now))
        return [(float(p[0]), float(p[1])/self._Y_TO_X, float(p[2])) for p in output]


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
        # Inference is the largest ongoing CPU cost. Preserve a responsive active
        # tracking cadence, but avoid processing excess camera frames with no hands.
        # Gesture countdown timing is wall-clock based and still updated by the UI.
        frame_interval = 1.0 / 15.0
        idle_frame_interval = 1.0 / 6.0
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
            # Active inference is capped at 15 FPS to reduce sustained CPU load. A much
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
    """Click-through overlay that keeps hands/TPI visible above the search launcher."""
    def __init__(self, owner):
        super().__init__(None)
        self.owner = owner
        self.setWindowTitle("Spatial Desktop Hand Indicators")
        # On Windows this is a real, independent top-level window rather than a Qt.Tool.
        # The shell's virtual-desktop pinning service needs a stable native window view;
        # Keep Windows as a regular top-level view for virtual-desktop pinning and
        # Task View integration; Linux keeps the established Qt.Tool behaviour unchanged.
        overlay_kind = Qt.Window if sys.platform.startswith("win") else Qt.Tool
        self.setWindowFlags(overlay_kind | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._last_raise = 0.0
        self._last_native_raise = 0.0
        self._native_style_ready = False
        self._global_hand_layer_active = False
        self._windows_virtual_desktop_pinned = False
        self._windows_virtual_desktop_pin_last_attempt = 0.0
        self._windows_virtual_desktop_pin_warning_reported = False
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(24)
        self.sync_timer.timeout.connect(self.sync_to_launcher)
        self.sync_timer.start()

    def sync_to_launcher(self):
        owner = self.owner
        dialog = owner.search_dialog
        search_visible = bool(dialog is not None and dialog.isVisible() and
                              not owner.hand_tracking_paused)
        # Windows needs a separate transparent hand layer which stays pinned across
        # virtual desktops. A QCursor position follows the active desktop on its own,
        # but hands painted into the original fullscreen window do not reliably follow
        # Win+Tab/Task View. This independent layer is click-through and owns the hands
        # on Windows; Linux keeps the original Search-only overlay behaviour.
        windows_global_layer = bool(
            sys.platform.startswith("win") and not owner.hand_tracking_paused and
            not getattr(owner, "hand_calibration_active", False) and
            getattr(owner, "tli_calibration_active_side", None) is None
        )
        self._global_hand_layer_active = windows_global_layer
        should_show = bool(search_visible or windows_global_layer)
        # A permanently visible transparent Windows hand layer previously repainted
        # the entire screen ~25 times/second even with no hands. Keep it responsive
        # while there are visible hand landmarks or running countdowns, but let the
        # empty layer fall to a low cadence. The 180 ms poll also detects new hands.
        active_visuals = bool(
            any(owner.hand_positions.get(side) for side in ("left", "right")) or
            any(owner.peace_hold_started.get(side) is not None for side in ("left", "right")) or
            any(owner.menu_pinch_started.get(side) is not None for side in ("left", "right")) or
            bool(getattr(owner, "custom_gesture_rings", {})) or
            any(owner.fist_action_menu_open.get(side, False) for side in ("left", "right"))
        )
        # User-configurable hand *display* FPS. This does not change camera
        # inference, the gesture state machine, or pointer/TPI input cadence.
        hand_frame_ms = max(1, round(1000.0 / max(5, min(30, owner.hand_display_fps))))
        wanted_interval = hand_frame_ms if active_visuals else 180
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
        if sys.platform.startswith("win"):
            # Pin this actual hand-rendering HWND as well as the main fullscreen owner.
            # Retry until Explorer confirms pinning; the Windows shell API is private
            # and may be unavailable on some builds.
            if (not self._windows_virtual_desktop_pinned and
                    now-self._windows_virtual_desktop_pin_last_attempt >= 2.0):
                self._windows_virtual_desktop_pin_last_attempt = now
                try:
                    pinned = pin_window_on_all_virtual_desktops(int(self.winId()))
                    if pinned:
                        self._windows_virtual_desktop_pinned = True
                    elif not self._windows_virtual_desktop_pin_warning_reported:
                        print("Spatial Desktop: Windows could not pin the hand-visual overlay to all virtual desktops; retrying.")
                        self._windows_virtual_desktop_pin_warning_reported = True
                except Exception as exc:
                    if not self._windows_virtual_desktop_pin_warning_reported:
                        print("Spatial Desktop: hand-visual virtual-desktop pinning unavailable:", exc)
                        self._windows_virtual_desktop_pin_warning_reported = True
            if not self._native_style_ready:
                try:
                    hwnd = wintypes.HWND(int(self.winId()))
                    user32 = ctypes.windll.user32
                    # HWND is pointer-sized on 64-bit Windows. Explicit signatures prevent
                    # ctypes from truncating it through its default 32-bit integer ABI.
                    get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
                    set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
                    get_long.argtypes = [wintypes.HWND, ctypes.c_int]
                    get_long.restype = ctypes.c_ssize_t
                    set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
                    set_long.restype = ctypes.c_ssize_t
                    style = int(get_long(hwnd, -20))
                    # Transparent input, layered composition, and no activation. Keep this
                    # as a normal top-level view: WS_EX_TOOLWINDOW explicitly excludes it
                    # from Task View/Alt+Tab and can interfere with Explorer's virtual-
                    # desktop view association. The overlay is still click-through and
                    # does not accept focus.
                    style |= 0x00000020 | 0x00080000 | 0x08000000
                    set_long(hwnd, -20, style)
                    self._native_style_ready = True
                    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                                    ctypes.c_int, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_int, wintypes.UINT]
                    user32.SetWindowPos.restype = wintypes.BOOL
                    # SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_SHOWWINDOW|SWP_FRAMECHANGED.
                    user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0,
                                        0x0001 | 0x0002 | 0x0010 | 0x0040 | 0x0020)
                except Exception:
                    pass
        # Reassert the topmost position without activation. This orders the overlay
        # above ordinary desktop windows; immersive Windows shell surfaces such as
        # Task View may still be in a higher system z-order band.
        raise_interval = 0.12 if active_visuals else 0.65
        if now-self._last_raise > raise_interval:
            self.raise_()
            self._last_raise = now
        if sys.platform.startswith("win") and now-self._last_native_raise > raise_interval:
            try:
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                # HWND_TOPMOST, SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_SHOWWINDOW.
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
        search_visible = bool(owner.search_dialog is not None and owner.search_dialog.isVisible())
        if not (search_visible or getattr(self, "_global_hand_layer_active", False)):
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
        owner._paint_floating_clipboard(painter, now)
        owner._paint_middle_pinch_zoom_timer(painter, now)
        # This overlay is the sole hand-visual owner while Search is visible; the main
        # window skips the identical layer to avoid double-painted ghost silhouettes.
        for side in ("left", "right"):
            points = owner.hand_positions.get(side, [])
            if not points:
                continue
            color = QColor(owner.left_click_color if side == "left" else owner.right_click_color)
            owner._paint_hand_skeleton(painter, points, side)
            owner._paint_gesture_disabled_indicator(painter, points, side)
            cursor = owner.side_cursor_positions.get(side)
            peace_counting = (owner.peace_hold_started.get(side) is not None and
                              not owner.peace_toggle_latched.get(side, False))
            if cursor is None and peace_counting:
                cursor = owner._scale_hand_tli_point(
                    owner.interactive_hand_anchor(points, side), points, side)
            if cursor is None:
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

        # Keep the thumbs-up tracking-toggle countdown visible in this dedicated layer.
        # The main window does not paint hands while this overlay owns the Windows hand
        # layer, so its confirmation ring must be reproduced here alongside custom rings.
        # For a hand whose tracking is currently disabled, use the same visual anchor
        # fallback as the main window so the ring still teaches how to re-enable it.
        for side in ("left", "right"):
            thumbs_started = getattr(owner, "thumbs_up_hold_started", {}).get(side)
            if (thumbs_started is None or
                    getattr(owner, "thumbs_up_toggle_latched", {}).get(side, False)):
                continue
            points = owner.hand_positions.get(side, [])
            cursor = owner.side_cursor_positions.get(side)
            if cursor is None:
                if not points:
                    continue
                cursor = owner._scale_hand_tli_point(
                    owner.interactive_hand_anchor(points, side), points, side)
            cursor = owner._visual_tli_position(side, cursor, points)
            elapsed = max(0.0, now-thumbs_started)
            duration = max(0.01, owner.gesture_tracking_toggle_hold_seconds)
            progress = min(1.0, elapsed/duration)
            remaining = max(0.0, duration-elapsed)
            ring = QRectF(cursor.x()-38, cursor.y()-38, 76, 76)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(105, 240, 255, 95), 3))
            painter.drawEllipse(ring)
            painter.setPen(QPen(QColor(105, 240, 255, 245), 4))
            painter.drawArc(ring, 90*16, -int(progress*360*16))
            painter.setPen(QColor(220, 255, 255, 245))
            painter.setFont(owner._ui_font(8, QFont.DemiBold))
            painter.drawText(QRectF(cursor.x()-24, cursor.y()+27, 48, 16), Qt.AlignCenter,
                             f"{remaining:.1f}s")

        # The main window suppresses its hand layer while this dedicated overlay owns
        # rendering. Keep recorded custom-gesture countdown rings in this layer too;
        # otherwise they disappear whenever the Windows overlay is active.
        for key, (label, progress, remaining) in list(getattr(owner, "custom_gesture_rings", {}).items()):
            if key == "overlap":
                a = owner.side_cursor_positions.get("left")
                b = owner.side_cursor_positions.get("right")
                if a is None or b is None:
                    continue
                cursor = QPointF((a.x()+b.x())*0.5, (a.y()+b.y())*0.5)
            else:
                cursor = owner.side_cursor_positions.get(key)
                if cursor is None:
                    continue
                cursor = owner._visual_tli_position(key, cursor, owner.hand_positions.get(key, []))
            ring = QRectF(cursor.x()-41, cursor.y()-41, 82, 82)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(110, 255, 170, 85), 3))
            painter.drawEllipse(ring)
            painter.setPen(QPen(QColor(110, 255, 170, 245), 4))
            painter.drawArc(ring, 90*16, -int(progress*360*16))
            painter.setPen(QColor(235, 255, 240, 245))
            painter.setFont(owner._ui_font(8, QFont.DemiBold))
            painter.drawText(QRectF(cursor.x()-50, cursor.y()-60, 100, 16), Qt.AlignCenter, label)
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
        # Windows uses this layer in place of main-window hand rendering. Paint
        # peace timers AFTER all cursors so an off hand still gets its circle.
        for side in ("left", "right"):
            owner._paint_peace_sign_timer(painter, side, now)
        painter.end()


class TliCalibrationWindow(QWidget):
    """Fullscreen, nine-point TPI calibration target; hand input is modal at app level."""
    cancel_requested = Signal()

    def __init__(self, side):
        super().__init__(None)
        self.side = side
        self.progress = 0
        self.sample_count = 12
        self.target_index = 0
        self.phase_text = "Release the pinch used to open this window"
        self.setWindowTitle(f"{side.title()} TPI Calibration")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.live_overlay = {"hands": [], "tli": []}
        screen = QApplication.primaryScreen()
        geometry = screen.geometry() if screen is not None else QRectF(0, 0, 1280, 800).toRect()
        self.setGeometry(geometry)
        self.target_positions = [QPointF(x, y) for x, y in
                                 ((fx * max(1, self.width()-1), fy * max(1, self.height()-1))
                                  for fx, fy in TPI_CALIBRATION_TARGET_FRACTIONS)]

    def set_full_screen_geometry(self, geometry):
        """Resize calibration to the whole native screen, including Qtile panels."""
        if geometry is None:
            return
        self.setGeometry(geometry)
        width = max(1, self.width() - 1)
        height = max(1, self.height() - 1)
        self.target_positions = [QPointF(fx * width, fy * height)
                                 for fx, fy in TPI_CALIBRATION_TARGET_FRACTIONS]
        self.update()

    def target_local_position(self):
        """Current target centre in fullscreen-window-local screen pixel coordinates."""
        index = min(TPI_CALIBRATION_TARGET_COUNT-1, max(0, int(self.target_index)))
        return QPointF(self.target_positions[index])

    def target_name(self):
        return TPI_CALIBRATION_TARGET_NAMES[min(TPI_CALIBRATION_TARGET_COUNT-1,
                                                max(0, int(self.target_index)))]

    def set_target_index(self, index):
        self.target_index = min(TPI_CALIBRATION_TARGET_COUNT-1, max(0, int(index)))
        self.progress = 0
        self.update()

    def pinch_area_global_rect(self):
        """A forgiving target hit area centred on the currently highlighted point."""
        centre = self.target_local_position()
        radius = max(38.0, min(62.0, min(self.width(), self.height()) * 0.055))
        global_centre = self.mapToGlobal(QPoint(round(centre.x()), round(centre.y())))
        return QRectF(float(global_centre.x()-radius), float(global_centre.y()-radius),
                      radius*2.0, radius*2.0)

    def set_phase(self, text, progress=None):
        self.phase_text = str(text)
        if progress is not None:
            self.progress = min(self.sample_count, max(0, int(progress)))
        self.update()

    def set_hand_overlay(self, hands, tli_points):
        """Paint live tracked hand and TPI above the calibration target."""
        self.live_overlay = {"hands": list(hands or []), "tli": list(tli_points or [])}
        self.update()

    def paintEvent(self, event):
        """Paint the fullscreen 3x3 target mesh and the live hand overlay safely."""
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
                print("Freehand: a calibration frame failed to paint (later failures are not repeated):")
                traceback.print_exc()
        finally:
            painter = self._frame_painter
            self._frame_painter = None
            if painter is not None and painter.isActive():
                painter.end()
            self._in_paint_event = False

    def _paint_frame(self, _event):
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter()
        self._frame_painter = painter
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        active = min(TPI_CALIBRATION_TARGET_COUNT-1, max(0, int(self.target_index)))
        # Small unobtrusive HUD plates leave the screen, app window, and hand visible.
        hud = QRectF(self.width()*0.5-280.0, 18.0, 560.0, 76.0)
        footer = QRectF(self.width()*0.5-280.0, self.height()-56.0, 560.0, 42.0)
        painter.setPen(QPen(QColor(90, 235, 255, 215), 1.5))
        painter.setBrush(QBrush(QColor(3, 14, 27, 212)))
        painter.drawRoundedRect(hud, 18, 18)
        painter.drawRoundedRect(footer, 14, 14)
        painter.setPen(QColor(235, 252, 255, 250))
        painter.setFont(QFont("Sans", 14, QFont.Bold))
        painter.drawText(QRectF(hud.left()+14, hud.top()+7, hud.width()-28, 27), Qt.AlignCenter,
                         f"{self.side.upper()} TPI CALIBRATION  ·  POINT {active+1}/9")
        painter.setPen(QColor(180, 223, 236, 245))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(QRectF(hud.left()+14, hud.top()+35, hud.width()-28, 25), Qt.AlignCenter,
                         "Move the TPI onto the highlighted target, then pinch and hold steady")

        # Show all nine positions faintly; the current target is large and unambiguous.
        for index, point in enumerate(self.target_positions):
            if index == active:
                continue
            painter.setPen(QPen(QColor(100, 232, 247, 105), 1.5))
            painter.setBrush(QBrush(QColor(5, 30, 44, 70)))
            painter.drawEllipse(point, 8.0, 8.0)
        centre = self.target_local_position()
        radius = max(38.0, min(62.0, min(self.width(), self.height()) * 0.055))
        painter.setPen(QPen(QColor(255, 255, 255, 225), 1.6))
        painter.setBrush(QBrush(QColor(245, 255, 255, 24)))
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(QPen(QColor(30, 240, 255, 245), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, 26.0, 26.0)
        painter.setPen(QPen(QColor(255, 255, 255, 250), 2.0))
        painter.drawLine(QPointF(centre.x()-13, centre.y()), QPointF(centre.x()+13, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y()-13), QPointF(centre.x(), centre.y()+13))
        # Name the active point without hiding targets near the lower edge.
        label_y = centre.y()+radius+9.0
        if label_y+32.0 > self.height()-62.0:
            label_y = centre.y()-radius-39.0
        label = QRectF(centre.x()-95.0, label_y, 190.0, 30.0)
        painter.setPen(QPen(QColor(30, 220, 242, 205), 1.0))
        painter.setBrush(QBrush(QColor(3, 18, 32, 220)))
        painter.drawRoundedRect(label, 10, 10)
        painter.setPen(QColor(230, 252, 255, 250))
        painter.setFont(QFont("Sans", 9, QFont.Bold))
        painter.drawText(label, Qt.AlignCenter, self.target_name())

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
        painter.drawText(QRectF(footer.left()+10, footer.top()+2, footer.width()-20, 17), Qt.AlignCenter,
                         self.phase_text)
        track = QRectF(footer.left()+42, footer.top()+27, footer.width()-84, 5)
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
        overlay_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        # Keep the main fullscreen tracking overlay unmanaged by Qtile/XLibre.
        # This stops Qtile from tiling, decorating, grouping, or otherwise handling
        # Spatial Desktop as a normal client window. Other dialogs remain managed.
        if sys.platform.startswith("linux"):
            overlay_flags |= Qt.X11BypassWindowManagerHint
        self.setWindowFlags(overlay_flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        settings = load_gesture_settings()
        self.hand_visual_scale = settings["hand_visual_scale"]
        self.hand_display_fps = int(min(30, max(5, round(settings.get("hand_display_fps", 15)))))
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
        self.sound_dropdown_key = None
        self.custom_gesture_key_capture = None  # gesture row awaiting a physical keyboard shortcut
        self._native_shortcut_capture = None
        self.custom_gesture_triggers = {"left": HoldTrigger(), "right": HoldTrigger(), "overlap": HoldTrigger()}
        # Track which timer starts have already played the delayed timer-during sound.
        self._timer_during_sound_played_at = {}
        self._timer_during_sound_stop_events = {}
        self._timer_during_sound_seen_keys = set()
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
        self.show_gesture_disabled_indicator = bool(
            self.ui_preferences.get("show_gesture_disabled_indicator", True))
        self.virtual_desktop_data = load_virtual_desktop_data()
        virtual_bg = self.virtual_desktop_data["background"]
        self.virtual_grid_enabled = bool(virtual_bg["grid_enabled"])
        self.virtual_scene = virtual_bg.get("scene", "garden" if virtual_bg.get("garden_mode") else "grid")
        if self.virtual_scene not in VIRTUAL_SCENES:
            self.virtual_scene = "grid"
        self.virtual_garden_mode = self.virtual_scene == "garden"
        self.virtual_space_mode = self.virtual_scene == "space"
        self.space_constellations = list(self.virtual_desktop_data.get("constellations", []))
        # Play-based onboarding is contextual, not a modal calibration wizard.
        self._scene_tutorial_steps = set()
        self._scene_arrow_pinch_previous = {"left": False, "right": False}
        self._scene_arrow_pinch_latched = {"left": False, "right": False}
        self._scene_transition = None
        self._space_constellation_revision = 0
        self._garden_constellation_cache = None
        self._garden_constellation_cache_key = None
        self.space_drawing_by_side = {"left": None, "right": None, "mouse": None}
        self.space_clear_press_latch = {"left": False, "right": False}
        self._space_stars = self._make_space_stars()
        self._space_background_cache = None
        self._space_background_size = None
        self.virtual_background_style = int(virtual_bg["style"])
        self.virtual_grid_brightness = float(virtual_bg["brightness"])
        self.virtual_grid_glow = float(virtual_bg["glow"])
        self.virtual_grid_spacing = float(virtual_bg["spacing"])
        self.virtual_grid_thickness = float(virtual_bg["thickness"])
        self.virtual_canvas_strokes = self.virtual_desktop_data["strokes"]
        self.virtual_canvas_layer = None
        self.virtual_canvas_layer_size = None
        self.virtual_canvas_layer_dirty = True
        self.virtual_dock_open = True
        self.virtual_pen_popup_open = False
        self.virtual_pen_active = False
        self.virtual_pen_color = VIRTUAL_PEN_PALETTE[1][1]
        self.virtual_pen_erase = False
        self.virtual_pen_active_stroke = None
        self.virtual_pen_last_status_at = 0.0
        # Garden weather is refreshed only while Garden Mode is visible. Network work runs
        # off the GUI thread and uses city-level IP geolocation, never webcam/hand data.
        self.garden_weather = {
            "source": "local-time", "city": "Local area", "country": "",
            "weather_code": 0, "temperature": float("nan"), "cloud_cover": 0.0,
            "wind_speed": 4.0, "precipitation": 0.0, "rain": 0.0, "snowfall": 0.0,
            "sunrise": None, "sunset": None, "timezone": "", "observed_at": "",
        }
        self.garden_weather_location = None
        self._garden_weather_lock = threading.Lock()
        self._garden_weather_fetching = False
        self._garden_weather_last_attempt = 0.0
        self._garden_background_cache = None
        self._garden_background_cache_key = None
        self._garden_grass_specs_size = None
        self._garden_grass_specs = None
        # Lightweight interactive Garden props; coordinates are normalised to the scene.
        self._garden_props = {
            "ball": {"x": 0.70, "y": 0.82, "vx": 0.0, "vy": 0.0, "r": 0.027, "held": None},
            "sundial": {"x": 0.30, "y": 0.77, "held": None},
        }
        # Garden interactions are local to the scene, never native OS mouse drags.
        # The projection is a click-to-toggle, not an expiring animation.
        self._garden_sundial_projection_active = False
        self._garden_sundial_hand_pinch = {"left": False, "right": False}
        self._garden_sundial_hand_owned = {"left": False, "right": False}
        self._garden_bird_seed = 19.0
        self._garden_bird_last_spawn = 0.0
        self._garden_birds = []
        self._garden_prop_last_tick = time.monotonic()
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
        # Floating clipboard history is populated from the real system clipboard. Entries
        # are kept in memory for this session and can be selected or dragged with a hand.
        self.floating_clipboard_visible = False
        self.floating_clipboard_entries = []
        self.floating_clipboard_max_entries = 36
        self.floating_clipboard_scroll_offset = 0
        # Optional screen-local top-left override; None retains the default right-side position.
        self.floating_clipboard_position = None
        self.floating_clipboard_drag_state = {"left": None, "right": None}
        self.floating_clipboard_drag_running = False
        self.floating_clipboard_temp_dir = Path(tempfile.gettempdir()) / "FreehandClipboard"
        self.floating_clipboard_temp_dir.mkdir(parents=True, exist_ok=True)
        self._clipboard_last_signature = None
        self._clipboard_last_signature_at = 0.0
        self._clipboard_capture_suppression_until = 0.0
        self.status = f"Ready — offline Vosk STT · Kokoro TTS: {self.tts.status}"

        self.desktop_mode = "os"  # start in OS Desktop mode; the menu can switch to virtual/3D mode
        self._applied_desktop_mode = None
        self._applied_menu_state = None
        self._last_native_topmost_reassert = 0.0
        self._windows_virtual_desktop_pinned = False
        self._windows_virtual_desktop_pin_last_attempt = 0.0
        self._windows_virtual_desktop_pin_warning_reported = False
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
        self.tli_calibrations = load_tli_calibrations()  # Legacy palm-local anchor calibration
        self.tpi_screen_calibrations = load_tpi_screen_calibrations(TPI_SCREEN_CALIBRATION_FILE)
        self.tli_calibration_persisted = {side: self.tpi_screen_calibrations.get(side) is not None
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
        self.tli_calibration_samples = {"left": [], "right": []}  # Samples for the current screen target
        self.tpi_calibration_draft_points = {"left": [], "right": []}
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
        # Countdown visuals must survive removal of the disabled hand's native cursor.
        self.peace_timer_anchor_by_side = {"left": None, "right": None}
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
        # Display-only: raw landmark geometry still drives all interaction logic.
        self._hand_visual_motion_filter = _HandVisualMotionFilter()
        self.hand_stable_positions = {"left": None, "right": None}
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
        # Two-hand thumb-to-middle pinch zoom. The existing Pinch Menu buffer is reused;
        # once the hold is confirmed, hand separation drives content zoom or corner resize.
        self.middle_pinch_zoom = {
            "started_at": None, "last_both_seen_at": None, "active": False,
            "mode": None, "hwnd": None, "rect": None, "corners": None,
            "start_points": None, "start_distance": None, "last_distance": None,
            "zoom_remainder": 0.0,
            "last_resize_rect": None, "last_resize_at": 0.0,
        }
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
        self.virtual_desktop_save_timer = QTimer(self)
        self.virtual_desktop_save_timer.setSingleShot(True)
        self.virtual_desktop_save_timer.timeout.connect(self._save_virtual_desktop_data)
        # A mouse-transparent topmost visual layer keeps the skeleton, TLI and gesture
        # progress visible over the search launcher without blocking its controls.
        self.search_hand_overlay = SearchHandOverlay(self)
        self._system_clipboard = QApplication.clipboard()
        try:
            self._system_clipboard.dataChanged.connect(self._capture_system_clipboard)
            QTimer.singleShot(350, self._capture_system_clipboard)
        except Exception:
            pass
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(50)
        self.custom_key_capture_poll_timer = QTimer(self)
        self.custom_key_capture_poll_timer.setInterval(20)
        self.custom_key_capture_poll_timer.timeout.connect(self._poll_custom_key_capture)
        # Open with the existing glass menu; hand calibration stays available in Hands settings.
        QTimer.singleShot(2400, self.speak_startup_greeting)

    def schedule_gesture_settings_save(self):
        self.settings_save_timer.start(450)

    def persist_gesture_settings(self):
        save_gesture_settings({
            "hand_visual_scale": self.hand_visual_scale,
            "hand_display_fps": self.hand_display_fps,
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
        grid = QPainter(image)
        self._paint_virtual_background(grid, width, height)
        grid.end()
        self._glass_virtual_backdrop = self._blur_glass_image(image, sigma=36.0)
        self._glass_virtual_backdrop_size = size
        return self._glass_virtual_backdrop

    def _virtual_background_style_data(self):
        index = max(0, min(len(VIRTUAL_BACKGROUND_STYLES)-1, int(self.virtual_background_style)))
        return VIRTUAL_BACKGROUND_STYLES[index]

    def _paint_virtual_background(self, painter, width, height):
        """Paint the active Virtual desktop scene (grid canvas or separate Garden scene)."""
        slide = getattr(self, "_scene_transition", None)
        if slide is not None:
            elapsed = max(0.0, time.monotonic()-slide["started"])
            amount = min(1.0, elapsed / slide["duration"])
            if amount >= 1.0:
                self._scene_transition = None
            else:
                # Smooth movement of two *cached* scene images; avoid re-rendering
                # the whole Garden and Space scenes on every transition frame.
                eased = amount * amount * (3.0-2.0*amount)
                travel = float(height) * eased
                direction = -1.0 if slide["from"] == "garden" else 1.0
                painter.save()
                try:
                    painter.setClipRect(QRectF(0, 0, width, height))
                    painter.drawImage(QPointF(0, direction*travel), slide["source"])
                    painter.drawImage(QPointF(0, direction*(travel-height)), slide["target"])
                finally:
                    painter.restore()
                return
        if getattr(self, "virtual_garden_mode", False):
            self._paint_virtual_garden(painter, width, height)
            return
        if getattr(self, "virtual_space_mode", False):
            self._paint_virtual_space(painter, width, height)
            return
        style = self._virtual_background_style_data()
        painter.fillRect(QRectF(0, 0, width, height), QColor(style["background"]))
        if not self.virtual_grid_enabled:
            return
        base = QColor(style["grid"])
        brightness = min(1.0, max(0.0, float(self.virtual_grid_brightness)/100.0))
        glow = min(1.0, max(0.0, float(self.virtual_grid_glow)/100.0))
        spacing = int(min(160, max(24, round(self.virtual_grid_spacing))))
        thickness = min(3.0, max(0.5, float(self.virtual_grid_thickness)))
        core_alpha = int(45 + 185*brightness)
        lines = [("x", x) for x in range(0, int(width)+1, spacing)]
        lines.extend(("y", y) for y in range(0, int(height)+1, spacing))
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Layered translucent passes create a neon-like bloom without an expensive blur.
        if glow > 0.01:
            outer = QColor(base.red(), base.green(), base.blue(), int(6 + glow*58*brightness))
            middle = QColor(base.red(), base.green(), base.blue(), int(10 + glow*88*brightness))
            for axis, pos in lines:
                painter.setPen(QPen(outer, thickness + 7.0*glow, Qt.SolidLine, Qt.RoundCap))
                if axis == "x":
                    painter.drawLine(QPointF(pos, 0), QPointF(pos, height))
                else:
                    painter.drawLine(QPointF(0, pos), QPointF(width, pos))
                painter.setPen(QPen(middle, thickness + 3.0*glow, Qt.SolidLine, Qt.RoundCap))
                if axis == "x":
                    painter.drawLine(QPointF(pos, 0), QPointF(pos, height))
                else:
                    painter.drawLine(QPointF(0, pos), QPointF(width, pos))
        core = QColor(base.red(), base.green(), base.blue(), core_alpha)
        for axis, pos in lines:
            painter.setPen(QPen(core, thickness, Qt.SolidLine, Qt.RoundCap))
            if axis == "x":
                painter.drawLine(QPointF(pos, 0), QPointF(pos, height))
            else:
                painter.drawLine(QPointF(0, pos), QPointF(width, pos))
        painter.restore()

    def _maybe_refresh_garden_weather(self):
        """Refresh live weather asynchronously, no more often than the provider cadence."""
        if not getattr(self, "virtual_garden_mode", False):
            return
        now = time.monotonic()
        lock = getattr(self, "_garden_weather_lock", None)
        if lock is None:
            return
        with lock:
            if self._garden_weather_fetching:
                return
            if now - float(self._garden_weather_last_attempt) < WEATHER_REFRESH_SECONDS:
                return
            self._garden_weather_last_attempt = now
            self._garden_weather_fetching = True
            location = dict(self.garden_weather_location) if isinstance(self.garden_weather_location, dict) else None

        def fetch_worker():
            try:
                fresh = fetch_live_weather(location=location, timeout=5.0)
                with self._garden_weather_lock:
                    self.garden_weather = fresh
                    cached_location = fresh.get("location")
                    self.garden_weather_location = (dict(cached_location)
                                                    if isinstance(cached_location, dict) else location)
                    self._garden_background_cache_key = None
            except Exception:
                # Keep the last successful report; if there isn't one, the garden falls
                # back to a local-time sun/moon cycle and a clear, calm scene.
                pass
            finally:
                with self._garden_weather_lock:
                    self._garden_weather_fetching = False

        threading.Thread(target=fetch_worker, name="FreehandGardenWeather", daemon=True).start()

    def _garden_scene_state(self):
        """Snapshot weather and calculate current daylight, so paint never performs I/O."""
        with self._garden_weather_lock:
            weather = dict(self.garden_weather)
        now = datetime.now()
        timezone_name = str(weather.get("timezone") or "")
        if timezone_name:
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(timezone_name))
            except Exception:
                now = datetime.now()
        solar = calculate_solar_state(now, weather.get("sunrise"), weather.get("sunset"))
        try:
            cloud_cover = max(0.0, min(100.0, float(weather.get("cloud_cover", 0.0))))
        except (TypeError, ValueError):
            cloud_cover = 0.0
        try:
            wind_speed = max(0.0, min(120.0, float(weather.get("wind_speed", 4.0))))
        except (TypeError, ValueError):
            wind_speed = 4.0
        kind = weather_kind(weather.get("weather_code", 0))
        cloudiness = cloud_cover / 100.0
        if kind == "mostly-clear":
            cloudiness = max(cloudiness, 0.16)
        elif kind == "partly-cloudy":
            cloudiness = max(cloudiness, 0.42)
        elif kind == "overcast":
            cloudiness = max(cloudiness, 0.88)
        elif kind == "cloudy":
            cloudiness = max(cloudiness, 0.38)
        elif kind == "fog":
            cloudiness = max(cloudiness, 0.76)
        elif kind in ("rain", "snow", "storm"):
            cloudiness = max(cloudiness, 0.72)
        # Clouds and precipitation dim the sun; clear weather leaves the time-of-day
        # brightness curve intact.
        sun_brightness = float(solar["solar_factor"])
        if solar["is_day"]:
            sun_brightness = (0.18 + 0.82 * sun_brightness) * (1.0 - 0.68 * cloudiness)
            if kind == "storm":
                sun_brightness *= 0.38
            elif kind in ("rain", "snow", "fog"):
                sun_brightness *= 0.72
        return {
            **weather, **solar,
            "kind": kind,
            "cloud_cover": cloud_cover,
            "cloudiness": min(1.0, max(0.0, cloudiness)),
            "wind_speed": wind_speed,
            "sun_brightness": min(1.0, max(0.0, sun_brightness)),
            # Before dawn, the sun will enter from the left and the moon sits right;
            # after sunset, the moon appears on the opposite (left) side.
            "moon_progress": float(solar.get("night_progress", 0.0)),
        }

    @staticmethod
    def _garden_mix_colour(first, second, amount):
        amount = min(1.0, max(0.0, float(amount)))
        return QColor(
            round(first.red() * (1.0 - amount) + second.red() * amount),
            round(first.green() * (1.0 - amount) + second.green() * amount),
            round(first.blue() * (1.0 - amount) + second.blue() * amount),
            round(first.alpha() * (1.0 - amount) + second.alpha() * amount),
        )

    def _paint_virtual_garden_background(self, painter, width, height, scene):
        """Render static sky, celestial objects, clouds and hills into a cached image."""
        width, height = float(width), float(height)
        is_day = bool(scene["is_day"])
        solar = float(scene["solar_factor"])
        cloudiness = float(scene["cloudiness"])
        kind = scene["kind"]
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        if is_day:
            day_luma = 0.48 + 0.52 * solar
            sky_top = QColor(round(62*day_luma), round(163*day_luma), round(239*day_luma))
            sky_mid = QColor(round(117*day_luma), round(202*day_luma), round(250*day_luma))
            sky_bottom = QColor(round(224*day_luma), round(245*day_luma), round(255*day_luma))
            twilight = max(0.0, 1.0 - solar * 2.7)
            sky_mid = self._garden_mix_colour(sky_mid, QColor(246, 166, 121), twilight * 0.54)
            sky_bottom = self._garden_mix_colour(sky_bottom, QColor(255, 205, 158), twilight * 0.40)
            sky_top = self._garden_mix_colour(sky_top, QColor(101, 116, 143), cloudiness * 0.52)
            sky_mid = self._garden_mix_colour(sky_mid, QColor(161, 174, 185), cloudiness * 0.36)
        else:
            sky_top = self._garden_mix_colour(QColor(6, 13, 34), QColor(67, 77, 93), cloudiness * 0.62)
            sky_mid = self._garden_mix_colour(QColor(18, 32, 59), QColor(79, 91, 103), cloudiness * 0.56)
            sky_bottom = self._garden_mix_colour(QColor(48, 63, 83), QColor(116, 122, 123), cloudiness * 0.50)
        sky = QLinearGradient(0, 0, 0, height * 0.80)
        sky.setColorAt(0.0, sky_top)
        sky.setColorAt(0.48, sky_mid)
        sky.setColorAt(1.0, sky_bottom)
        painter.fillRect(QRectF(0, 0, width, height), QBrush(sky))

        if not is_day:
            # Deterministic star placement means the cached scene is stable between weather updates.
            for index in range(38):
                x = ((index * 131.7 + 19.0) % max(1.0, width))
                y = ((index * 37.9 + 11.0) % max(1.0, height * 0.50))
                alpha = 95 + ((index * 29) % 145)
                radius = 0.7 + (index % 3) * 0.32
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(221, 237, 255, alpha))
                painter.drawEllipse(QPointF(x, y), radius, radius)

            # A soft crescent moon on the other side of the scene from the day's sun path.
            moon_x = width * (0.16 + 0.68 * float(scene.get("moon_progress", 0.5)))
            moon = QPointF(moon_x, height * 0.19)
            moon_radius = max(17.0, height * 0.036)
            halo = QRadialGradient(moon, moon_radius * 3.3)
            halo.setColorAt(0.0, QColor(195, 220, 255, 60))
            halo.setColorAt(0.48, QColor(174, 207, 255, 25))
            halo.setColorAt(1.0, QColor(174, 207, 255, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(halo))
            painter.drawEllipse(moon, moon_radius * 3.3, moon_radius * 3.3)
            painter.setBrush(QColor(235, 241, 249, 245))
            painter.drawEllipse(moon, moon_radius, moon_radius)
            shadow_offset = moon_radius * 0.42
            cut = self._garden_mix_colour(sky_top, sky_mid, 0.28)
            painter.setBrush(cut)
            painter.drawEllipse(QPointF(moon.x() + shadow_offset, moon.y() - moon_radius * 0.10),
                                moon_radius * 0.90, moon_radius * 0.90)
        else:
            # The sun traverses the sky over daylight; its halo and colour follow both
            # solar elevation and real cloud cover/conditions.
            progress = float(scene["progress"])
            sun = QPointF(width * (0.12 + 0.76 * progress),
                          height * (0.27 - 0.15 * math.sin(math.pi * progress)))
            sun_radius = max(22.0, height * 0.050)
            brightness = float(scene["sun_brightness"])
            halo_radius = sun_radius * (2.1 + 1.6 * brightness)
            halo = QRadialGradient(sun, halo_radius)
            halo.setColorAt(0.0, QColor(255, 222, 137, int(108 * brightness)))
            halo.setColorAt(0.42, QColor(255, 218, 120, int(60 * brightness)))
            halo.setColorAt(1.0, QColor(255, 218, 120, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(halo))
            painter.drawEllipse(sun, halo_radius, halo_radius)
            sun_fill = QRadialGradient(
                QPointF(sun.x() - sun_radius * 0.23, sun.y() - sun_radius * 0.27),
                sun_radius * 1.5,
            )
            sun_fill.setColorAt(0.0, QColor(255, 255, 224, int(255 * brightness)))
            sun_fill.setColorAt(0.62, QColor(255, 236, 166, int(250 * brightness)))
            sun_fill.setColorAt(1.0, QColor(255, 180, 85, int(240 * brightness)))
            painter.setBrush(QBrush(sun_fill))
            painter.setPen(QPen(QColor(255, 241, 187, int(180*brightness)), max(1.0, height*0.001)))
            painter.drawEllipse(sun, sun_radius, sun_radius)

        # Real-time cloud cover changes both the light and the visible cloud layer.
        cloud_count = min(7, max(0, int(round(cloudiness * 7))))
        for index in range(cloud_count):
            cloud_x = ((index * 0.217 + 0.035) % 1.0) * width
            cloud_y = height * (0.095 + ((index * 7) % 7) * 0.023)
            cloud_w = height * (0.095 + ((index * 5) % 4) * 0.010)
            cloud_h = height * (0.027 + ((index * 3) % 3) * 0.006)
            if is_day:
                cloud_colour = QColor(245, 249, 251, int(35 + cloudiness * 175))
                cloud_shadow = QColor(93, 119, 142, int(10 + cloudiness * 60))
            else:
                cloud_colour = QColor(126, 145, 171, int(25 + cloudiness * 100))
                cloud_shadow = QColor(15, 28, 49, int(25 + cloudiness * 70))
            painter.setPen(Qt.NoPen)
            painter.setBrush(cloud_shadow)
            painter.drawEllipse(QRectF(cloud_x-cloud_w*0.34, cloud_y-cloud_h*0.02,
                                       cloud_w*0.68, cloud_h*1.20))
            painter.drawEllipse(QRectF(cloud_x-cloud_w*0.12, cloud_y-cloud_h*0.35,
                                       cloud_w*0.45, cloud_h*1.15))
            painter.drawEllipse(QRectF(cloud_x+cloud_w*0.10, cloud_y-cloud_h*0.10,
                                       cloud_w*0.48, cloud_h*1.0))
            painter.setBrush(cloud_colour)
            painter.drawEllipse(QRectF(cloud_x-cloud_w*0.35, cloud_y-cloud_h*0.12,
                                       cloud_w*0.66, cloud_h))
            painter.drawEllipse(QRectF(cloud_x-cloud_w*0.10, cloud_y-cloud_h*0.44,
                                       cloud_w*0.46, cloud_h*1.04))
            painter.drawEllipse(QRectF(cloud_x+cloud_w*0.12, cloud_y-cloud_h*0.15,
                                       cloud_w*0.45, cloud_h*0.98))

        # Rolling hills change to cooler, darker greens at night and in poor weather.
        if is_day:
            far_top, far_bottom = QColor(137, 199, 137), QColor(78, 158, 91)
            mid_top, mid_bottom = QColor(93, 183, 92), QColor(37, 126, 62)
            fore_top, fore_mid, fore_bottom = QColor(79, 179, 70), QColor(42, 143, 54), QColor(22, 93, 43)
            if kind in ("rain", "storm", "fog", "snow") or cloudiness > 0.65:
                far_top = self._garden_mix_colour(far_top, QColor(115, 143, 132), cloudiness * 0.55)
                far_bottom = self._garden_mix_colour(far_bottom, QColor(57, 104, 83), cloudiness * 0.48)
                mid_top = self._garden_mix_colour(mid_top, QColor(77, 122, 91), cloudiness * 0.50)
                mid_bottom = self._garden_mix_colour(mid_bottom, QColor(26, 82, 57), cloudiness * 0.42)
                fore_top = self._garden_mix_colour(fore_top, QColor(62, 122, 61), cloudiness * 0.35)
        else:
            far_top, far_bottom = QColor(35, 74, 78), QColor(21, 50, 53)
            mid_top, mid_bottom = QColor(29, 83, 58), QColor(15, 53, 40)
            fore_top, fore_mid, fore_bottom = QColor(28, 74, 42), QColor(17, 58, 34), QColor(7, 34, 24)
        distant = QPainterPath()
        distant.moveTo(0, height*0.59)
        distant.cubicTo(width*0.13, height*0.48, width*0.20, height*0.50, width*0.32, height*0.57)
        distant.cubicTo(width*0.44, height*0.65, width*0.52, height*0.49, width*0.67, height*0.55)
        distant.cubicTo(width*0.80, height*0.60, width*0.88, height*0.46, width, height*0.53)
        distant.lineTo(width, height); distant.lineTo(0, height); distant.closeSubpath()
        hill_grad = QLinearGradient(0, height*0.48, 0, height*0.91)
        hill_grad.setColorAt(0.0, far_top); hill_grad.setColorAt(1.0, far_bottom)
        painter.setPen(Qt.NoPen); painter.setBrush(QBrush(hill_grad)); painter.drawPath(distant)

        middle = QPainterPath()
        middle.moveTo(0, height*0.70)
        middle.cubicTo(width*0.16, height*0.59, width*0.27, height*0.66, width*0.40, height*0.70)
        middle.cubicTo(width*0.55, height*0.75, width*0.62, height*0.59, width*0.76, height*0.66)
        middle.cubicTo(width*0.87, height*0.70, width*0.94, height*0.61, width, height*0.65)
        middle.lineTo(width, height); middle.lineTo(0, height); middle.closeSubpath()
        middle_grad = QLinearGradient(0, height*0.59, 0, height)
        middle_grad.setColorAt(0.0, mid_top); middle_grad.setColorAt(1.0, mid_bottom)
        painter.setBrush(QBrush(middle_grad)); painter.drawPath(middle)

        foreground = QPainterPath()
        foreground.moveTo(0, height*0.79)
        foreground.cubicTo(width*0.20, height*0.71, width*0.31, height*0.77, width*0.47, height*0.81)
        foreground.cubicTo(width*0.64, height*0.85, width*0.82, height*0.70, width, height*0.77)
        foreground.lineTo(width, height); foreground.lineTo(0, height); foreground.closeSubpath()
        fore_grad = QLinearGradient(0, height*0.72, 0, height)
        fore_grad.setColorAt(0.0, fore_top); fore_grad.setColorAt(0.45, fore_mid); fore_grad.setColorAt(1.0, fore_bottom)
        painter.setBrush(QBrush(fore_grad)); painter.drawPath(foreground)
        painter.setPen(QPen(QColor(184, 229, 124, 48 if is_day else 18), max(1.0, height*0.0025),
                            Qt.SolidLine, Qt.RoundCap))
        highlight = QPainterPath(QPointF(0, height*0.79))
        highlight.cubicTo(width*0.20, height*0.71, width*0.31, height*0.77, width*0.47, height*0.81)
        highlight.cubicTo(width*0.64, height*0.85, width*0.82, height*0.70, width, height*0.77)
        painter.drawPath(highlight)

        # Storybook meadow details inspired by the supplied reference: tiny flower
        # clusters, pale grass flecks and tapered cypress-like trees on the ridges.
        painter.setPen(Qt.NoPen)
        for index in range(34):
            fx = ((index * 0.173 + 0.025) % 1.0) * width
            ridge_y = height * (0.61 + 0.11 * ((index * 7) % 13) / 13.0)
            tree_h = height * (0.025 + (index % 4) * 0.008)
            tree_w = tree_h * 0.22
            tree = QPainterPath()
            tree.moveTo(fx, ridge_y-tree_h)
            tree.cubicTo(fx-tree_w*0.55, ridge_y-tree_h*0.60, fx-tree_w*0.72, ridge_y-tree_h*0.18, fx, ridge_y)
            tree.cubicTo(fx+tree_w*0.72, ridge_y-tree_h*0.18, fx+tree_w*0.55, ridge_y-tree_h*0.60, fx, ridge_y-tree_h)
            painter.setBrush(QColor(39, 102, 44, 210 if is_day else 110))
            painter.drawPath(tree)
        for index in range(95):
            fx = ((index * 67.13 + 11.0) % max(1.0, width))
            fy = height * (0.83 + ((index * 19) % 16) / 100.0)
            size = max(1.0, height * (0.0012 + (index % 3) * 0.00045))
            if index % 4 == 0:
                petal = QColor(255, 246, 214, 230 if is_day else 100)
            elif index % 4 == 1:
                petal = QColor(255, 174, 207, 225 if is_day else 90)
            elif index % 4 == 2:
                petal = QColor(255, 255, 250, 235 if is_day else 110)
            else:
                petal = QColor(179, 208, 255, 210 if is_day else 80)
            painter.setBrush(petal)
            painter.drawEllipse(QPointF(fx, fy), size*1.5, size*1.5)
            painter.setBrush(QColor(255, 205, 70, 230 if is_day else 100))
            painter.drawEllipse(QPointF(fx, fy), size*0.48, size*0.48)
        for index in range(125):
            fx = (index * 83.7 + 5.0) % max(1.0, width)
            fy = height * (0.78 + ((index * 31) % 22) / 100.0)
            painter.setPen(QPen(QColor(226, 245, 167, 100 if is_day else 35), max(1.0, height*0.0011), Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(fx, fy), QPointF(fx+height*0.003, fy-height*0.008))

        if kind == "fog":
            fog = QLinearGradient(0, height*0.33, 0, height*0.86)
            fog.setColorAt(0.0, QColor(224, 235, 241, 0))
            fog.setColorAt(0.45, QColor(224, 235, 241, 78))
            fog.setColorAt(1.0, QColor(224, 235, 241, 12))
            painter.fillRect(QRectF(0, height*0.30, width, height*0.60), QBrush(fog))
        painter.restore()

    def _garden_grass_specs_for_size(self, width, height):
        """Precompute static grass blade geometry; only hand response moves per frame."""
        key = (int(width), int(height))
        if self._garden_grass_specs_size == key and self._garden_grass_specs is not None:
            return self._garden_grass_specs
        specs = []
        rows = (
            (0.700, 18.0, 30.0, 0.64, QColor(139, 218, 103), max(28.0, width/72.0)),
            (0.785, 27.0, 43.0, 0.82, QColor(97, 196, 72), max(22.0, width/88.0)),
            (0.905, 34.0, 57.0, 1.0, QColor(59, 159, 57), max(17.0, width/112.0)),
        )
        for row_index, (base_ratio, min_len, max_len, alpha_scale, base_color, step) in enumerate(rows):
            row_step = step * (1.12 if row_index == 0 else 1.0)
            blades = []
            x = -row_step
            while x < width + row_step:
                wave = math.sin(x*0.0081 + row_index*1.7) * height * 0.011
                base_y = min(height+4.0, height*base_ratio+wave)
                variation = (math.sin(x*0.047 + row_index*2.4)+1.0)*0.5
                blade_len = min_len + (max_len-min_len)*variation
                base_sway = math.sin(x*0.015+row_index*2.7)*4.0
                blades.append((x, base_y, blade_len, base_sway))
                x += row_step
            specs.append((row_index, alpha_scale, base_color, blades))
        self._garden_grass_specs_size = key
        self._garden_grass_specs = specs
        return specs

    def _paint_garden_grass(self, painter, width, height, scene):
        """Paint a compact, batched grass layer that bends away from nearby hand cursors."""
        cursor_points = []
        for side in ("left", "right"):
            cursor = getattr(self, "side_cursor_positions", {}).get(side)
            if cursor is None:
                continue
            try:
                cursor_x, cursor_y = float(cursor.x()), float(cursor.y())
            except (AttributeError, TypeError, ValueError):
                continue
            if cursor_y >= height * 0.53:
                cursor_points.append((cursor_x, cursor_y, side))
        wind = min(1.0, float(scene["wind_speed"])/28.0)
        phase = time.monotonic() * (0.35 + wind*1.4)
        influence_radius = max(110.0, height*0.20)
        radius_sq = influence_radius * influence_radius
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        for row_index, alpha_scale, base_color, blades in self._garden_grass_specs_for_size(width, height):
            path = QPainterPath()
            for x, base_y, blade_len, base_sway in blades:
                sway = base_sway + math.sin(x*0.008 + phase + row_index*1.1) * wind * 3.2
                push = 0.0
                for cursor_x, cursor_y, side in cursor_points:
                    dx = x - cursor_x
                    if abs(dx) >= influence_radius:
                        continue
                    dy = (base_y - blade_len*0.53) - cursor_y
                    if abs(dy) >= influence_radius:
                        continue
                    distance_sq = dx*dx + dy*dy
                    if distance_sq >= radius_sq:
                        continue
                    influence = 1.0 - distance_sq/radius_sq
                    direction = (-1.0 if side == "left" else 1.0) if abs(dx) < 1.0 else (1.0 if dx > 0 else -1.0)
                    push += direction * blade_len * 0.92 * influence
                bend = max(-blade_len*1.12, min(blade_len*1.12, sway+push))
                tip_x = x + bend
                tip_y = base_y - blade_len
                path.moveTo(QPointF(x, base_y))
                path.cubicTo(QPointF(x+bend*0.12, base_y-blade_len*0.42),
                             QPointF(x+bend*0.86, base_y-blade_len*0.78),
                             QPointF(tip_x, tip_y))
            colour = QColor(base_color)
            colour.setAlpha(int((210 if scene["is_day"] else 145)*alpha_scale))
            pen_width = (1.2 + 1.8*alpha_scale) * (1.0 if row_index == 2 else 0.76)
            painter.setPen(QPen(colour, pen_width, Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(path)
        painter.restore()

    def _paint_garden_weather_effects(self, painter, width, height, scene):
        """Low-cost animated precipitation driven by the latest current weather code."""
        kind = scene["kind"]
        if kind not in ("rain", "storm", "snow"):
            return
        phase = (time.monotonic() * (82.0 + float(scene["wind_speed"])*1.7)) % max(1.0, height+80.0)
        painter.save()
        if kind in ("rain", "storm"):
            rain_path = QPainterPath()
            count = 74 if kind == "storm" else 52
            slant = 3.0 + min(12.0, float(scene["wind_speed"])*0.22)
            for index in range(count):
                x = (index*97.31 + phase*0.62) % max(1.0, width)
                y = (index*71.27 + phase*1.35) % max(1.0, height)
                rain_path.moveTo(QPointF(x, y))
                rain_path.lineTo(QPointF(x-slant, y+9.0+(index%4)))
            painter.setPen(QPen(QColor(206, 231, 255, 108 if kind == "storm" else 78),
                                1.1, Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(rain_path)
        else:
            for index in range(34):
                x = (index*83.7 + math.sin(index*1.9)*19.0) % max(1.0, width)
                y = (index*59.1 + phase*0.60) % max(1.0, height)
                radius = 1.0 + (index%3)*0.65
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(241, 248, 255, 185))
                painter.drawEllipse(QPointF(x, y), radius, radius)
        painter.restore()

    def _paint_garden_weather_label(self, painter, width, height, scene):
        """Show unobtrusive current-weather status and required Open-Meteo attribution."""
        if scene.get("source") == "open-meteo":
            city = str(scene.get("city") or "Local area").strip()
            temperature = scene.get("temperature")
            try:
                temperature_text = f" · {float(temperature):.0f}°C" if math.isfinite(float(temperature)) else ""
            except (TypeError, ValueError):
                temperature_text = ""
            text = f"LIVE WEATHER · {city}{temperature_text} · Open-Meteo"
        else:
            text = "WEATHER OFFLINE · local day/night"
        painter.save()
        painter.setFont(self._ui_font(7, QFont.DemiBold))
        metrics = QFontMetricsF(painter.font())
        label_width = min(max(205.0, metrics.horizontalAdvance(text)+22.0), max(205.0, width-24.0))
        rect = QRectF(max(12.0, width-label_width-14.0), max(8.0, height-31.0), label_width, 21.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(8, 18, 32, 92 if scene["is_day"] else 142))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setPen(QColor(255, 255, 255, 210 if scene["is_day"] else 225))
        painter.drawText(rect.adjusted(9.0, 0.0, -6.0, 0.0), Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.desktop_mode == "virtual":
            if self._scene_transition is not None:
                event.accept()
                return
            if not self._gesture_ui_modal_open() and self._scene_arrow_hit(event.position()):
                self._start_scene_slide(time.monotonic())
                event.accept()
                return
        if (event.button() == Qt.LeftButton and self.desktop_mode == "virtual" and
                self.virtual_space_mode and not self._gesture_ui_modal_open()):
            point = event.position()
            if self._space_clear_hit(point):
                self._space_clear_constellations()
            else:
                self._space_start_constellation("mouse", point)
            event.accept()
            return
        if (event.button() == Qt.LeftButton and self.virtual_garden_mode and
                self.desktop_mode == "virtual" and not self._gesture_ui_modal_open()):
            point = event.position()
            if self._garden_sundial_hit(point):
                self._garden_toggle_sundial(time.monotonic())
                event.accept()
                return
            if self._garden_ball_hit(point) and self._garden_props["ball"].get("held") is None:
                ball = self._garden_props["ball"]
                ball["held"] = "mouse"
                ball["release_since"] = None
                ball["offset_x"] = ball["x"] - point.x()/max(1., float(self.width()))
                ball["offset_y"] = ball["y"] - point.y()/max(1., float(self.height()))
                ball["last_drag_at"] = time.monotonic()
                ball["vx"] = ball["vy"] = 0.0
                self.status = "Garden ball picked up"
                self._scene_tutorial_steps.add("ball")
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._garden_props["ball"].get("held") == "mouse":
            ball = self._garden_props["ball"]
            point = event.position()
            width, height = max(1., float(self.width())), max(1., float(self.height()))
            x = min(1.0-ball["r"], max(ball["r"], point.x()/width+ball.get("offset_x", 0.0)))
            y = min(1.0-ball["r"], max(ball["r"], point.y()/height+ball.get("offset_y", 0.0)))
            now = time.monotonic()
            dt = max(.008, now-float(ball.get("last_drag_at", now)))
            ball["vx"] = max(-1.8, min(1.8, (x-ball["x"])/dt))
            ball["vy"] = max(-1.8, min(1.8, (y-ball["y"])/dt))
            ball["x"], ball["y"] = x, y
            ball["last_drag_at"] = now
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.space_drawing_by_side.get("mouse") is not None:
            self._space_finish_constellation("mouse")
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._garden_props["ball"].get("held") == "mouse":
            self._garden_drop_ball("mouse")
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _garden_ball_hit(self, point):
        """Hit-test the painted ball in pixels (not mixed normalised x/y distances)."""
        if point is None or self.desktop_mode != "virtual" or not self.virtual_garden_mode:
            return False
        if self._gesture_ui_modal_open() or self.floating_clipboard_visible:
            return False
        ball = self._garden_props["ball"]
        width, height = max(1.0, float(self.width())), max(1.0, float(self.height()))
        radius = max(15.0, height*float(ball["r"]) + 11.0)
        return math.hypot(float(point.x()) - ball["x"]*width,
                          float(point.y()) - ball["y"]*height) <= radius

    def _garden_sundial_hit(self, point):
        if point is None or self.desktop_mode != "virtual" or not self.virtual_garden_mode:
            return False
        if self._gesture_ui_modal_open():
            return False
        sundial = self._garden_props["sundial"]
        width, height = max(1.0, float(self.width())), max(1.0, float(self.height()))
        x, y = float(point.x()), float(point.y())
        dx, dy = sundial["x"]*width, sundial["y"]*height
        radius = min(width, height)*0.043
        # Clicking the dial plate or its short pedestal activates the clock.
        on_plate = (abs(x-dx) <= radius*1.12 and abs(y-dy) <= radius*0.55+9.0)
        on_column = (abs(x-dx) <= radius*0.70 and dy <= y <= height*0.862)
        return bool(on_plate or on_column)

    @staticmethod
    def _garden_object_pinch(landmarks, held=False):
        """Local pinch recognition for scene props; do not change OS/Menu poses.

        Garden pickup should not require two *other* fingers to be extended, as
        the main Pinch Menu classifier does. It is deliberately only queried
        while handling a garden prop.
        """
        if landmarks is None or len(landmarks) < 21:
            return False
        palm = max(1e-5, distance3d(landmarks, 0, 9))
        index_gap = distance3d(landmarks, 4, 8)/palm
        if index_gap > (0.60 if held else 0.42):
            return False
        # Reject a thumb-to-middle-finger pinch, but tolerate a naturally
        # folded middle finger when index and thumb are very close together.
        middle_gap = distance3d(landmarks, 4, 12)/palm
        return bool(index_gap < 0.23 or index_gap + 0.065 < middle_gap)

    def _garden_toggle_sundial(self, now):
        self._garden_sundial_projection_active = not self._garden_sundial_projection_active
        self._scene_tutorial_steps.add("sundial")
        self.status = ("Sundial sky clock enabled" if self._garden_sundial_projection_active
                       else "Sundial sky clock disabled")
        self.update()

    def _garden_handle_sundial_frame(self, side, cursor, is_pinch, now):
        """Toggle on a fresh pinch over the dial; consume until it is released.

        This bypasses the ordinary Pinch Menu and native click dispatcher, so
        the dial can be toggled even if a fist or menu latch was left armed.
        """
        was_pinched = self._garden_sundial_hand_pinch[side]
        self._garden_sundial_hand_pinch[side] = bool(is_pinch)
        owned = self._garden_sundial_hand_owned[side]
        over = self._garden_sundial_hit(cursor)
        if owned:
            self._clear_menu_pinch_sequence(side)
            if not is_pinch:
                self._garden_sundial_hand_owned[side] = False
            return True
        if not over:
            return False
        self._clear_menu_pinch_sequence(side)
        self._reset_fist_gesture(side)
        if is_pinch and not was_pinched:
            self._garden_sundial_hand_owned[side] = True
            self._garden_toggle_sundial(now)
        return True

    def _garden_handle_ball_frame(self, side, cursor, is_pinch, now):
        """Capture ball pinch immediately: no menu countdown and no OS mouseDown."""
        if self.desktop_mode != "virtual" or not self.virtual_garden_mode:
            return False
        ball = self._garden_props["ball"]
        held = ball.get("held")
        if held is not None and held != side:
            # Still suppress Pinch Menu while hovering over an already-held ball.
            if self._garden_ball_hit(cursor):
                self._clear_menu_pinch_sequence(side)
                return True
            return False
        over_ball = self._garden_ball_hit(cursor)
        if held != side and not over_ball:
            return False

        # An interaction that begins over the ball must not arm the Pinch Menu,
        # even while just hovering or while its pinch is being released.
        self._clear_menu_pinch_sequence(side)
        self._reset_fist_gesture(side)
        if held != side:
            if not is_pinch:
                return True
            ball["held"] = side
            ball["release_since"] = None
            ball["offset_x"] = ball["x"] - float(cursor.x())/max(1.0, float(self.width()))
            ball["offset_y"] = ball["y"] - float(cursor.y())/max(1.0, float(self.height()))
            ball["last_drag_at"] = now
            ball["vx"] = ball["vy"] = 0.0
            self.gesture_owner = side
            self.status = "Garden ball picked up — move while pinching, release to drop"
            self._scene_tutorial_steps.add("ball")
        if is_pinch:
            ball["release_since"] = None
            width, height = max(1.0, float(self.width())), max(1.0, float(self.height()))
            x = min(1.0-ball["r"], max(ball["r"],
                    float(cursor.x())/width + ball.get("offset_x", 0.0)))
            y = min(1.0-ball["r"], max(ball["r"],
                    float(cursor.y())/height + ball.get("offset_y", 0.0)))
            dt = max(0.001, now-float(ball.get("last_drag_at", now)))
            if dt >= 0.008:
                target_vx = max(-1.8, min(1.8, (x-ball["x"])/dt))
                target_vy = max(-1.8, min(1.8, (y-ball["y"])/dt))
                ball["vx"] = 0.6*ball["vx"] + 0.4*target_vx
                ball["vy"] = 0.6*ball["vy"] + 0.4*target_vy
                ball["last_drag_at"] = now
            ball["x"], ball["y"] = x, y
        else:
            # Allow a tiny dropout, but release normally when the pinch opens.
            if ball.get("release_since") is None:
                ball["release_since"] = now
            if now-ball["release_since"] >= 0.08:
                self._garden_drop_ball(side)
        return True

    def _garden_drop_ball(self, side):
        ball = self._garden_props["ball"]
        if ball.get("held") != side:
            return
        ball["held"] = None
        ball["release_since"] = None
        ball.pop("offset_x", None)
        ball.pop("offset_y", None)
        ball.pop("last_drag_at", None)
        if self.gesture_owner == side:
            self.gesture_owner = None
        self.status = "Garden ball dropped"

    def _paint_garden_sky_clock(self, painter, width, height, scene):
        """Bat-signal-style illuminated digital clock projected from the sundial."""
        if not self._garden_sundial_projection_active:
            return
        dial = self._garden_props["sundial"]
        dx, dy = dial["x"]*width, dial["y"]*height
        sky_x, sky_y = max(width*0.15, min(width*0.84, dx+width*0.045)), height*0.265
        radius = min(height*0.100, width*0.115)
        now = datetime.now()
        tz_name = str(scene.get("timezone") or "")
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(tz_name))
            except Exception:
                pass
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Wide transparent cone radiates from the gnomon to the sky projection.
        beam = QPainterPath()
        beam.moveTo(dx, dy-height*0.040)
        beam.lineTo(sky_x-radius*0.80, sky_y)
        beam.lineTo(sky_x+radius*0.80, sky_y)
        beam.closeSubpath()
        gradient = QLinearGradient(dx, dy, sky_x, sky_y)
        gradient.setColorAt(0, QColor(255, 239, 156, 8))
        gradient.setColorAt(0.55, QColor(195, 236, 255, 34))
        gradient.setColorAt(1, QColor(185, 228, 255, 47))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawPath(beam)
        halo = QRadialGradient(QPointF(sky_x, sky_y), radius*1.35)
        halo.setColorAt(0.0, QColor(227, 246, 255, 118))
        halo.setColorAt(0.55, QColor(158, 215, 254, 93))
        halo.setColorAt(0.79, QColor(149, 201, 246, 32))
        halo.setColorAt(1.0, QColor(149, 201, 246, 0))
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(QPointF(sky_x, sky_y), radius*1.35, radius*1.35)
        painter.setBrush(QColor(12, 41, 68, 160))
        painter.setPen(QPen(QColor(216, 245, 255, 205), max(1.5, radius*0.035)))
        painter.drawEllipse(QPointF(sky_x, sky_y), radius, radius*0.76)
        clock_font = QFont("Consolas")
        clock_font.setWeight(QFont.Bold)
        clock_font.setPixelSize(max(13, int(radius*0.39)))
        painter.setFont(clock_font)
        painter.setPen(QColor(229, 250, 255, 250))
        painter.drawText(QRectF(sky_x-radius, sky_y-radius*0.43,
                                radius*2.0, radius*0.80), Qt.AlignCenter,
                         now.strftime("%H:%M:%S"))
        painter.restore()

    def _paint_garden_props(self, painter, width, height, scene):
        """Animate small birds and hand-pickable garden props with inexpensive 2D physics."""
        now = time.monotonic()
        # Repaints run at an adaptive rate; allow longer idle frame intervals
        # without slowing ball gravity, bounces or bird travel in Garden Mode.
        dt = min(0.12, max(0.0, now - getattr(self, "_garden_prop_last_tick", now)))
        self._garden_prop_last_tick = now
        props = self._garden_props
        # Object grabbing is handled in the gesture loop, not in paint. Otherwise
        # a native Windows drag can mistakenly move the ball and arm Pinch Menu.
        ball = props["ball"]
        if ball.get("held") in ("left", "right"):
            owner = ball["held"]
            last_seen = getattr(self, "hand_last_seen", {}).get(owner, 0.0) or 0.0
            if (not self.gesture_tracking_enabled.get(owner, True) or
                    now-float(last_seen) > 0.40):
                self._garden_drop_ball(owner)
        if ball["held"] is None:
            ball["vy"] += 0.78 * dt
            ball["x"] += ball["vx"] * dt
            ball["y"] += ball["vy"] * dt
            floor = 0.865
            if ball["y"] + ball["r"] > floor:
                ball["y"] = floor - ball["r"]
                if abs(ball["vy"]) > 0.045:
                    ball["vy"] = -ball["vy"] * 0.53
                else:
                    ball["vy"] = 0.0
                ball["vx"] *= 0.985
            if ball["x"] < ball["r"] or ball["x"] > 1.0-ball["r"]:
                ball["x"] = min(1.0-ball["r"], max(ball["r"], ball["x"]))
                ball["vx"] *= -0.68
            ball["vx"] *= max(0.0, 1.0 - dt*0.55)
        # Bird flock: staggered silhouettes fly across the sky, then respawn off-screen.
        if now - self._garden_bird_last_spawn > 2.8:
            self._garden_bird_last_spawn = now
            self._garden_bird_seed = (self._garden_bird_seed * 1.71 + 13.7) % 997.0
            self._garden_birds.append({
                "x": -0.06, "y": 0.20 + (self._garden_bird_seed % 100)/1000.0,
                "speed": 0.10 + (self._garden_bird_seed % 55)/400.0,
                "scale": 0.7 + (self._garden_bird_seed % 40)/100.0,
                "phase": self._garden_bird_seed,
            })
        self._garden_birds = [b for b in self._garden_birds if b["x"] < 1.08]
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        for bird in self._garden_birds:
            bird["x"] += bird["speed"] * dt
            bx, by = bird["x"]*width, bird["y"]*height
            span = height * 0.012 * bird["scale"]
            flap = math.sin(now*8.0 + bird["phase"]) * span * 0.7
            pen = QPen(QColor(43, 67, 65, 210 if scene["is_day"] else 170), max(1.2, height*0.0015), Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(bx-span, by+flap), QPointF(bx, by))
            painter.drawLine(QPointF(bx, by), QPointF(bx+span, by+flap))
        # Ball with a soft ground shadow, glossy red/orange surface.
        bx, by, br = ball["x"]*width, ball["y"]*height, ball["r"]*height
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(10, 44, 20, 65))
        painter.drawEllipse(QRectF(bx-br*1.15, height*0.856, br*2.3, br*0.35))
        ball_grad = QRadialGradient(QPointF(bx-br*0.34, by-br*0.38), br*1.6)
        ball_grad.setColorAt(0.0, QColor(255, 232, 155))
        ball_grad.setColorAt(0.35, QColor(255, 105, 56))
        ball_grad.setColorAt(1.0, QColor(177, 42, 29))
        painter.setBrush(QBrush(ball_grad))
        painter.setPen(QPen(QColor(255, 245, 201, 210), max(1.0, br*0.07)))
        painter.drawEllipse(QPointF(bx, by), br, br)
        # Sundial on its short, shaped stone pedestal (visible below the dial plate).
        dial = props["sundial"]
        dx, dy = dial["x"]*width, dial["y"]*height
        radius = min(width, height)*0.043
        ground_y = height*0.864
        column_top = dy + radius*0.06
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(7, 28, 19, 70))
        painter.drawEllipse(QRectF(dx-radius*1.12, ground_y-radius*0.16,
                                   radius*2.24, radius*0.40))
        column_gradient = QLinearGradient(dx-radius*0.71, 0, dx+radius*0.71, 0)
        column_gradient.setColorAt(0.0, QColor(85, 87, 84))
        column_gradient.setColorAt(0.32, QColor(198, 199, 188))
        column_gradient.setColorAt(0.75, QColor(153, 160, 152))
        column_gradient.setColorAt(1.0, QColor(87, 94, 91))
        painter.setBrush(QBrush(column_gradient))
        painter.setPen(QPen(QColor(79, 88, 82, 225), max(1.0, radius*0.048)))
        painter.drawRoundedRect(QRectF(dx-radius*0.67, column_top,
                                       radius*1.34, max(3.0, ground_y-column_top)),
                                radius*0.11, radius*0.11)
        painter.setBrush(QColor(174, 178, 167))
        painter.drawRoundedRect(QRectF(dx-radius*0.85, ground_y-radius*0.22,
                                       radius*1.70, radius*0.33),
                                radius*0.055, radius*0.055)
        painter.setBrush(QColor(196, 199, 190))
        painter.drawRoundedRect(QRectF(dx-radius*0.84, column_top-radius*0.20,
                                       radius*1.68, radius*0.39),
                                radius*0.07, radius*0.07)
        painter.setPen(QPen(QColor(86, 87, 61, 220), max(1.0, radius*0.08)))
        painter.setBrush(QColor(198, 183, 126))
        painter.drawEllipse(QPointF(dx, dy), radius, radius*0.34)
        painter.setBrush(QColor(227, 215, 164))
        painter.drawEllipse(QPointF(dx, dy-radius*0.06), radius*0.88, radius*0.29)
        painter.setPen(QPen(QColor(122, 104, 63, 180), max(1.0, radius*0.025)))
        for i in range(12):
            a = i*math.pi/6
            painter.drawLine(QPointF(dx+math.cos(a)*radius*0.67, dy-radius*0.06+math.sin(a)*radius*0.20),
                             QPointF(dx+math.cos(a)*radius*0.80, dy-radius*0.06+math.sin(a)*radius*0.24))
        progress = float(scene.get("progress", 0.5))
        angle = (progress-0.5)*math.pi*0.92
        gnomon_tip = QPointF(dx+math.cos(angle)*radius*0.18, dy-radius*0.06-math.sin(angle)*radius*1.05)
        painter.setPen(QPen(QColor(91, 75, 43), max(2.0, radius*0.10), Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(dx, dy-radius*0.06), gnomon_tip)
        shadow_angle = angle + math.pi
        painter.setPen(QPen(QColor(52, 62, 42, 145), max(2.0, radius*0.09), Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(dx, dy-radius*0.03),
                         QPointF(dx+math.cos(shadow_angle)*radius*0.83, dy-radius*0.03+math.sin(shadow_angle)*radius*0.25))
        painter.restore()

    def _paint_virtual_garden(self, painter, width, height):
        """Paint a cached, weather-aware Garden scene with hand-reactive foreground grass."""
        width, height = float(width), float(height)
        if width <= 0 or height <= 0:
            return
        self._maybe_refresh_garden_weather()
        scene = self._garden_scene_state()
        cache_key = (
            int(round(width)), int(round(height)), bool(scene["is_day"]),
            round(float(scene["progress"]), 2), round(float(scene.get("night_progress", 0.0)), 2),
            round(float(scene["solar_factor"]), 2), scene["kind"], int(float(scene["cloud_cover"])//8),
        )
        background = self._garden_background_cache
        if background is None or background.isNull() or self._garden_background_cache_key != cache_key:
            background = QImage(max(1, int(round(width))), max(1, int(round(height))),
                                QImage.Format.Format_ARGB32_Premultiplied)
            background.fill(QColor(7, 15, 35))
            background_painter = QPainter(background)
            try:
                self._paint_virtual_garden_background(background_painter, width, height, scene)
            finally:
                background_painter.end()
            self._garden_background_cache = background
            self._garden_background_cache_key = cache_key
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.drawImage(QPointF(0, 0), background)
        # Stars drawn in Space persist in the Garden sky when it is actually night.
        # The constellation layer is cached independently of Garden's weather cache.
        self._paint_garden_constellations(painter, width, height, scene)
        self._paint_garden_weather_effects(painter, width, height, scene)
        self._paint_garden_sky_clock(painter, width, height, scene)
        self._paint_garden_props(painter, width, height, scene)
        self._paint_garden_grass(painter, width, height, scene)
        self._paint_garden_weather_label(painter, width, height, scene)
        self._paint_scene_arrow(painter, width, height, "garden")
        self._paint_scene_tutorial_hint(painter, width, height, "garden", scene)
        painter.restore()

    def _scene_arrow_rect(self, scene=None, width=None, height=None):
        """Same arrow geometry for painting, mouse clicks and hand hit-testing."""
        scene = self.virtual_scene if scene is None else scene
        width = float(self.width() if width is None else width)
        height = float(self.height() if height is None else height)
        if scene not in ("garden", "space"):
            return QRectF()
        y = 22.0 if scene == "garden" else max(8.0, height-95.0)
        return QRectF(max(8.0, (width-206.0)/2.0), y, min(206.0, width-16.0), 55.0)

    def _scene_arrow_hit(self, point):
        return bool(point is not None and self.desktop_mode == "virtual" and
                    self.virtual_scene in ("garden", "space") and
                    self._scene_transition is None and
                    not self._gesture_ui_modal_open() and
                    self._scene_arrow_rect().contains(point))

    def _paint_scene_arrow(self, painter, width, height, scene):
        rect = self._scene_arrow_rect(scene, width, height)
        if rect.isEmpty():
            return
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(162, 214, 251, 175), 1.25))
            painter.setBrush(QColor(11, 25, 51, 195) if scene == "garden"
                             else QColor(18, 22, 58, 210))
            painter.drawRoundedRect(rect, 16.0, 16.0)
            x = rect.left() + 27.0
            y = rect.center().y()
            direction = -1.0 if scene == "garden" else 1.0
            painter.setPen(QPen(QColor(203, 244, 255, 245), 3.0,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(QPointF(x, y+direction*-11.0), QPointF(x, y+direction*11.0))
            painter.drawLine(QPointF(x, y+direction*11.0),
                             QPointF(x-9.0, y+direction*2.0))
            painter.drawLine(QPointF(x, y+direction*11.0),
                             QPointF(x+9.0, y+direction*2.0))
            painter.setPen(QColor(234, 245, 255))
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            painter.drawText(rect.adjusted(48.0, 0.0, -7.0, 0.0),
                             Qt.AlignVCenter | Qt.AlignLeft,
                             "EXPLORE STARS" if scene == "garden" else "BACK TO GARDEN")
        finally:
            painter.restore()

    def _paint_scene_tutorial_hint(self, painter, width, height, scene, weather=None):
        """Unintrusive play prompts, advanced by real interactions in each scene."""
        steps = self._scene_tutorial_steps
        if scene == "garden":
            if self.space_constellations and "constellation" in steps:
                hint = ("Look up! Your constellations shine in the night sky."
                        if weather is not None and not weather.get("is_day", True)
                        else "Your constellations will appear in this sky at night.")
            elif "ball" not in steps:
                hint = "TRY IT  ·  Pinch and carry the bouncing ball"
            elif "sundial" not in steps:
                hint = "TRY IT  ·  Tap the sundial to project the time"
            elif "space" not in steps:
                hint = "NEXT  ·  Select the up arrow to explore the stars"
            else:
                return
        elif scene == "space":
            hint = ("TRY IT  ·  Pinch a star, trace to others, then release"
                    if "constellation" not in steps
                    else "NEXT  ·  Use the down arrow to see your stars in Garden")
        else:
            return
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setFont(self._ui_font(9, QFont.DemiBold))
            available = max(170.0, width-40.0)
            text_width = min(available, QFontMetricsF(painter.font()).horizontalAdvance(hint)+30.0)
            y = 95.0 if scene == "garden" else 104.0
            bubble = QRectF(24.0, y, text_width, 33.0)
            painter.setPen(QPen(QColor(130, 200, 248, 70), 1.0))
            painter.setBrush(QColor(8, 21, 43, 175))
            painter.drawRoundedRect(bubble, 10.0, 10.0)
            painter.setPen(QColor(228, 247, 255, 238))
            painter.drawText(bubble.adjusted(12.0, 0.0, -6.0, 0.0),
                             Qt.AlignVCenter | Qt.AlignLeft, hint)
        finally:
            painter.restore()

    def _render_scene_slide_frame(self, scene):
        """Capture each scene once; the transition itself only moves these images."""
        width, height = max(1, self.width()), max(1, self.height())
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor(4, 11, 29))
        painter = QPainter(image)
        try:
            if scene == "garden":
                self._paint_virtual_garden(painter, width, height)
            else:
                self._paint_virtual_space(painter, width, height)
        finally:
            painter.end()
        return image

    def _start_scene_slide(self, now):
        """Slide vertically between the two playable virtual scenes, without blocking input."""
        if (self._scene_transition is not None or self.desktop_mode != "virtual" or
                self.virtual_scene not in ("garden", "space") or self._gesture_ui_modal_open()):
            return False
        source_scene = self.virtual_scene
        target_scene = "space" if source_scene == "garden" else "garden"
        source_image = self._render_scene_slide_frame(source_scene)
        self._set_virtual_scene(target_scene)
        self._scene_tutorial_steps.add("space" if target_scene == "space" else "returned")
        target_image = self._render_scene_slide_frame(target_scene)
        self._scene_transition = {
            "from": source_scene, "to": target_scene, "started": now,
            "duration": 0.90, "source": source_image, "target": target_image,
        }
        self.status = ("Ascending into Space View" if target_scene == "space"
                       else "Descending into the Garden")
        self.update()
        return True

    def _scene_arrow_handle_frame(self, side, cursor, landmarks, now, edge_tolerant=False):
        """Prioritise arrow pinch-clicks and silence menus while hovering arrows."""
        is_pinch = (not is_fist_or_grip_pose(landmarks, edge_tolerant=edge_tolerant)
                    and self._garden_object_pinch(landmarks,
                        held=self._scene_arrow_pinch_latched.get(side, False)))
        previous = self._scene_arrow_pinch_previous.get(side, False)
        self._scene_arrow_pinch_previous[side] = bool(is_pinch)
        if self._scene_transition is not None:
            self._clear_menu_pinch_sequence(side)
            self._reset_fist_gesture(side)
            if not is_pinch:
                self._scene_arrow_pinch_latched[side] = False
            return True
        if self._scene_arrow_pinch_latched.get(side, False):
            self._clear_menu_pinch_sequence(side)
            if not is_pinch:
                self._scene_arrow_pinch_latched[side] = False
            return True
        if not self._scene_arrow_hit(cursor):
            return False
        self._clear_menu_pinch_sequence(side)
        self._reset_fist_gesture(side)
        if is_pinch and not previous:
            if self._start_scene_slide(now):
                self._scene_arrow_pinch_latched[side] = True
        return True  # Hover itself should never start the Pinch Menu or fist scroll.

    def _paint_garden_constellations(self, painter, width, height, scene):
        """Draw saved Space constellations over the Garden sky, at night only."""
        if scene.get("is_day", True) or not self.space_constellations:
            return
        size = (max(1, int(width)), max(1, int(height*0.57)))
        key = (size, self._space_constellation_revision)
        if self._garden_constellation_cache is None or self._garden_constellation_cache_key != key:
            overlay = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
            overlay.fill(Qt.transparent)
            layer = QPainter(overlay)
            try:
                layer.setRenderHint(QPainter.Antialiasing, True)
                # The Space star identities and line topology are unchanged. Only
                # project their positions into the Garden's upper sky region.
                for constellation in self.space_constellations:
                    points = []
                    for star_id in constellation.get("stars", []):
                        if type(star_id) is int and 0 <= star_id < len(self._space_stars):
                            sx, sy = self._space_stars[star_id][:2]
                            points.append(QPointF(width*(0.08+0.84*sx),
                                                  height*(0.065+0.50*sy)))
                    if len(points) < 2:
                        continue
                    path = QPainterPath(points[0])
                    for point in points[1:]:
                        path.lineTo(point)
                    layer.setBrush(Qt.NoBrush)
                    layer.setPen(QPen(QColor(63, 177, 255, 42), 9.0,
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    layer.drawPath(path)
                    layer.setPen(QPen(QColor(151, 213, 255, 160), 3.0,
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    layer.drawPath(path)
                    layer.setPen(QPen(QColor(238, 252, 255, 210), 1.25,
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    layer.drawPath(path)
                    layer.setPen(Qt.NoPen)
                    for point in points:
                        layer.setBrush(QColor(126, 201, 255, 70))
                        layer.drawEllipse(point, 6.0, 6.0)
                        layer.setBrush(QColor(246, 251, 255, 245))
                        layer.drawEllipse(point, 2.3, 2.3)
            finally:
                layer.end()
            self._garden_constellation_cache = overlay
            self._garden_constellation_cache_key = key
        painter.save()
        try:
            painter.setOpacity(max(0.18, 1.0-0.80*float(scene.get("cloudiness", 0.0))))
            painter.drawImage(QPointF(0, 0), self._garden_constellation_cache)
        finally:
            painter.restore()

    @staticmethod
    def _make_space_stars():
        """Stable normalized star positions: ids survive resizing and app restarts."""
        rng = random.Random(812734)
        stars = []
        # 100 bright drawable anchors, visible even when the nebula is dim.
        for index in range(100):
            x = 0.045 + 0.91 * rng.random()
            y = 0.085 + 0.80 * rng.random()
            stars.append((x, y, 1.2 + 2.7*rng.random(), rng.random()*6.283, index))
        return stars

    def _set_virtual_scene(self, scene):
        if scene not in VIRTUAL_SCENES or scene == self.virtual_scene:
            return
        self._scene_transition = None
        self._finish_virtual_pen_stroke()
        for side in ("left", "right", "mouse"):
            self._space_finish_constellation(side)
        if self.virtual_garden_mode:
            for side in ("left", "right", "mouse"):
                if self._garden_props["ball"].get("held") == side:
                    self._garden_drop_ball(side)
        self.virtual_scene = scene
        self.virtual_garden_mode = scene == "garden"
        self.virtual_space_mode = scene == "space"
        if scene != "grid":
            self.virtual_pen_active = False
            self.virtual_pen_erase = False
            self.virtual_pen_popup_open = False
        self._glass_virtual_backdrop = None
        self._glass_virtual_backdrop_size = None
        self._schedule_virtual_desktop_save()
        self.status = f"Virtual scene: {VIRTUAL_SCENE_LABELS[scene].title()}"
        self.update()

    def _space_nearest_star(self, point, radius=None):
        if point is None:
            return None
        width, height = max(1., float(self.width())), max(1., float(self.height()))
        limit = (max(64., min(116., min(width, height)*0.098)) if radius is None else radius)
        best, best_distance = None, limit*limit
        for x, y, _, _, index in self._space_stars:
            distance = (point.x()-x*width)**2 + (point.y()-y*height)**2
            if distance < best_distance:
                best, best_distance = index, distance
        return best

    def _space_star_point(self, star_id, width, height):
        star = self._space_stars[int(star_id)]
        return QPointF(star[0]*width, star[1]*height)

    def _space_clear_rect(self, width=None, height=None):
        width = float(self.width() if width is None else width)
        height = float(self.height() if height is None else height)
        return QRectF(width-206.0, height-78.0, 182.0, 44.0)

    def _space_clear_hit(self, point):
        return bool(point is not None and self.virtual_space_mode and
                    self._space_clear_rect().contains(point) and
                    not self._gesture_ui_modal_open())

    def _space_clear_constellations(self):
        if self.space_constellations:
            self.space_constellations.clear()
            self._space_constellation_revision += 1
            self._garden_constellation_cache = None
            self._schedule_virtual_desktop_save()
        for side in self.space_drawing_by_side:
            self.space_drawing_by_side[side] = None
        self.status = "Space constellations cleared"
        self.update()

    def _space_start_constellation(self, side, point):
        star_id = self._space_nearest_star(point)
        if star_id is None:
            return False
        self.space_drawing_by_side[side] = {
            "stars": [star_id], "cursor": QPointF(point),
            "last_join_at": 0.,
        }
        self.status = "Tracing constellation — move across stars; release to finish"
        self.update()
        return True

    def _space_extend_constellation(self, side, point, now):
        stroke = self.space_drawing_by_side.get(side)
        if stroke is None or point is None:
            return False
        stroke["cursor"] = QPointF(point)
        candidate = self._space_nearest_star(point, radius=max(42., min(68., self.height()*.065)))
        if candidate is not None and candidate not in stroke["stars"]:
            latest = self._space_star_point(stroke["stars"][-1], self.width(), self.height())
            target = self._space_star_point(candidate, self.width(), self.height())
            if (math.hypot(latest.x()-target.x(), latest.y()-target.y()) >= 30. and
                    now - stroke["last_join_at"] >= 0.10):
                stroke["stars"].append(candidate)
                stroke["last_join_at"] = now
                if len(stroke["stars"]) >= 72:
                    self._space_finish_constellation(side)
                    return True
        self.update()
        return True

    def _space_finish_constellation(self, side):
        state = self.space_drawing_by_side.get(side)
        if state is None:
            return
        self.space_drawing_by_side[side] = None
        if len(state["stars"]) >= 2:
            self.space_constellations.append({"stars": list(state["stars"])})
            self.space_constellations = self.space_constellations[-120:]
            self._space_constellation_revision += 1
            self._garden_constellation_cache = None
            self._scene_tutorial_steps.add("constellation")
            self._schedule_virtual_desktop_save()
            self.status = f"Constellation saved · {len(state['stars'])} stars connected"
        self.update()

    def _space_handle_frame(self, side, cursor, is_pinch, now):
        """Own Space pinches before fist/Pinch Menu classification, never OS dragging."""
        if not self.virtual_space_mode or self.desktop_mode != "virtual":
            return False
        if not is_pinch:
            self.space_clear_press_latch[side] = False
            if self.space_drawing_by_side.get(side) is not None:
                self._space_finish_constellation(side)
                self._clear_menu_pinch_sequence(side)
                return True
            return False
        self._clear_menu_pinch_sequence(side)
        self._reset_fist_gesture(side)
        self.active_action[side] = None
        self.action_stop_wait_open[side] = True
        self.action_stop_open_since[side] = None
        if self._space_clear_hit(cursor):
            if not self.space_clear_press_latch[side]:
                self._space_clear_constellations()
                self.space_clear_press_latch[side] = True
            return True
        if self.space_clear_press_latch[side]:
            return True  # Must release after CLEAR before starting a new line.
        if self.space_drawing_by_side.get(side) is None:
            self._space_start_constellation(side, cursor)
        else:
            self._space_extend_constellation(side, cursor, now)
        return True

    def _paint_space_background(self, painter, width, height):
        """One-time cached nebula, ringed planet, dust field and stars (no network/assets)."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        base = QLinearGradient(0, 0, width, height)
        base.setColorAt(0, QColor(2, 4, 21))
        base.setColorAt(.45, QColor(11, 13, 48))
        base.setColorAt(1, QColor(1, 9, 26))
        painter.fillRect(QRectF(0, 0, width, height), QBrush(base))
        # Expansive clouds of gas with translucent radial colour glows.
        for x, y, radius, rgba in (
            (.18, .30, .61, (56, 42, 144, 94)),
            (.42, .62, .59, (20, 124, 160, 70)),
            (.65, .22, .48, (151, 52, 139, 50)),
            (.21, .78, .32, (22, 180, 182, 37)),
            (.89, .70, .44, (23, 59, 158, 50))):
            cx, cy = width*x, height*y
            gradient = QRadialGradient(QPointF(cx, cy), min(width, height)*radius)
            gradient.setColorAt(0, QColor(*rgba))
            gradient.setColorAt(.52, QColor(rgba[0], rgba[1], rgba[2], rgba[3]//3))
            gradient.setColorAt(1, QColor(rgba[0], rgba[1], rgba[2], 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            r = min(width, height)*radius
            painter.drawEllipse(QPointF(cx, cy), r, r)
        # Tilted luminous galactic arm: layered wispy ellipses.
        painter.save()
        painter.translate(width*.30, height*.45)
        painter.rotate(-24)
        for w, h, tint in ((.52, .13, (87, 83, 192, 15)),
                           (.47, .077, (119, 121, 210, 18)),
                           (.40, .039, (176, 187, 248, 15))):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(*tint))
            painter.drawEllipse(QRectF(-width*w*.5, -height*h*.5, width*w, height*h))
        painter.restore()
        rng = random.Random(93941)
        for i in range(520):
            x, y = rng.random()*width, rng.random()*height
            radius = .25 + rng.random()*.85
            alpha = rng.randint(46, 178)
            painter.setBrush(QColor(182+rng.randint(0, 70), 194+rng.randint(0, 50), 255, alpha))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), radius, radius)
        # Atmospheric halo behind the planet.
        planet_center = QPointF(width*.79, height*.47)
        planet_r = min(width, height)*.155
        aura = QRadialGradient(planet_center, planet_r*1.85)
        aura.setColorAt(0, QColor(71, 151, 226, 32))
        aura.setColorAt(.72, QColor(75, 88, 225, 36))
        aura.setColorAt(1, QColor(25, 80, 240, 0))
        painter.setBrush(QBrush(aura))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(planet_center, planet_r*1.85, planet_r*1.85)
        # Rings sit partly behind the planet and are redrawn as a foreground arc.
        painter.save()
        painter.translate(planet_center)
        painter.rotate(-25)
        for width_offset, alpha in ((1.75, 78), (1.92, 95), (2.07, 32)):
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(135, 187, 248, alpha), 2.0))
            painter.drawEllipse(QRectF(-planet_r*width_offset, -planet_r*.47,
                                       planet_r*width_offset*2, planet_r*.94))
        painter.restore()
        surface = QRadialGradient(QPointF(planet_center.x()-planet_r*.42,
                                         planet_center.y()-planet_r*.45), planet_r*1.85)
        surface.setColorAt(0, QColor(144, 207, 239))
        surface.setColorAt(.40, QColor(46, 117, 183))
        surface.setColorAt(.76, QColor(26, 57, 110))
        surface.setColorAt(1, QColor(7, 17, 50))
        painter.setPen(QPen(QColor(127, 191, 255, 150), 1.6))
        painter.setBrush(QBrush(surface))
        painter.drawEllipse(planet_center, planet_r, planet_r)
        # Discrete bright anchor stars with subtle cyan halo and diffraction crosses.
        for x, y, size, phase, _ in self._space_stars:
            point = QPointF(width*x, height*y)
            glow = QRadialGradient(point, size*5.0)
            glow.setColorAt(0, QColor(210, 248, 255, 180))
            glow.setColorAt(1, QColor(76, 184, 245, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(point, size*5.0, size*5.0)
            painter.setBrush(QColor(235, 248, 255, 238))
            painter.drawEllipse(point, max(.9, size*.55), max(.9, size*.55))
            if size > 2.5:
                painter.setPen(QPen(QColor(162, 219, 255, 90), .85))
                painter.drawLine(QPointF(point.x()-size*3, point.y()),
                                 QPointF(point.x()+size*3, point.y()))
                painter.drawLine(QPointF(point.x(), point.y()-size*3),
                                 QPointF(point.x(), point.y()+size*3))
        painter.restore()

    def _paint_space_constellation(self, painter, ids, width, height, active=False, cursor=None):
        points = [self._space_star_point(i, width, height) for i in ids if 0 <= i < len(self._space_stars)]
        if not points:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if len(points) >= 2:
            line = QPainterPath(points[0])
            for point in points[1:]:
                line.lineTo(point)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(55, 185, 255, 35), 12.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(line)
            painter.setPen(QPen(QColor(89, 217, 255, 105), 5.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(line)
            painter.setPen(QPen(QColor(213, 249, 255, 235), 1.9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(line)
        if active and cursor is not None:
            painter.setPen(QPen(QColor(174, 239, 255, 135), 1.6, Qt.DashLine))
            painter.drawLine(points[-1], cursor)
        for point in points:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(76, 199, 255, 90))
            painter.drawEllipse(point, 9., 9.)
            painter.setBrush(QColor(240, 251, 255))
            painter.drawEllipse(point, 3.5, 3.5)
        painter.restore()

    def _paint_virtual_space(self, painter, width, height):
        size = (max(1, int(width)), max(1, int(height)))
        if self._space_background_cache is None or self._space_background_size != size:
            canvas = QImage(size[0], size[1], QImage.Format.Format_RGB32)
            background_painter = QPainter(canvas)
            try:
                self._paint_space_background(background_painter, width, height)
            finally:
                background_painter.end()
            self._space_background_cache = canvas
            self._space_background_size = size
        painter.drawImage(QPointF(0, 0), self._space_background_cache)
        for drawing in self.space_constellations:
            self._paint_space_constellation(painter, drawing["stars"], width, height)
        for state in self.space_drawing_by_side.values():
            if state is not None:
                self._paint_space_constellation(painter, state["stars"], width, height,
                                                active=True, cursor=state["cursor"])
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QColor(207, 230, 255, 225))
        painter.setFont(self._ui_font(13, QFont.DemiBold))
        painter.drawText(QRectF(30, 26, min(620, width-50), 32), Qt.AlignLeft | Qt.AlignVCenter,
                         "SPACE VIEW   /   CONSTELLATIONS")
        painter.setPen(QColor(164, 201, 237, 205))
        painter.setFont(self._ui_font(9, QFont.Normal))
        painter.drawText(QRectF(30, 64, min(720, width-50), 32), Qt.AlignLeft | Qt.AlignVCenter,
                         "Pinch and trace between stars · release to save")
        button = self._space_clear_rect(width, height)
        painter.setBrush(QColor(11, 34, 64, 218))
        painter.setPen(QPen(QColor(104, 204, 240, 170), 1.3))
        painter.drawRoundedRect(button, 12, 12)
        painter.setPen(QColor(226, 247, 255))
        painter.setFont(self._ui_font(9, QFont.DemiBold))
        painter.drawText(button, Qt.AlignCenter, "CLEAR CONSTELLATIONS")
        self._paint_scene_arrow(painter, width, height, "space")
        self._paint_scene_tutorial_hint(painter, width, height, "space")
        painter.restore()

    def _save_virtual_desktop_data(self):
        self.virtual_desktop_data = {
            "background": {
                "grid_enabled": bool(self.virtual_grid_enabled),
                "garden_mode": bool(self.virtual_garden_mode),
                "scene": self.virtual_scene,
                "style": int(self.virtual_background_style),
                "brightness": float(self.virtual_grid_brightness),
                "glow": float(self.virtual_grid_glow),
                "spacing": float(self.virtual_grid_spacing),
                "thickness": float(self.virtual_grid_thickness),
            },
            "strokes": self.virtual_canvas_strokes,
            "constellations": self.space_constellations,
        }
        saved = save_virtual_desktop_data(self.virtual_desktop_data)
        if saved:
            self.virtual_desktop_data = normalise_virtual_desktop_data(self.virtual_desktop_data)

    def _schedule_virtual_desktop_save(self):
        timer = getattr(self, "virtual_desktop_save_timer", None)
        if timer is not None:
            timer.start(300)
        else:
            self._save_virtual_desktop_data()

    def _virtual_dock_geometry(self):
        """Dock and Pen app geometry, anchored above the bottom screen edge."""
        width, height = float(self.width()), float(self.height())
        dock = QRectF((width-292.0)*0.5, height-86.0, 292.0, 66.0)
        pen = QRectF(dock.left()+112.0, dock.top()+8.0, 48.0, 48.0)
        return {"dock": dock, "pen": pen}

    def _virtual_pen_popup_geometry(self):
        """Colour/eraser palette positioned directly above the Pen dock item."""
        pen = self._virtual_dock_geometry()["pen"]
        cell, gap, margin = 29.0, 7.0, 12.0
        # Palette items: seven colours, eraser, pointer and Clear Canvas.
        count = len(VIRTUAL_PEN_PALETTE)+3
        width = margin*2 + count*cell + (count-1)*gap
        height = 54.0
        left = min(max(8.0, pen.center().x()-width*0.5), max(8.0, self.width()-width-8.0))
        top = max(8.0, pen.top()-height-12.0)
        panel = QRectF(left, top, width, height)
        items = {}
        for index, (name, color) in enumerate(VIRTUAL_PEN_PALETTE):
            x = left+margin+index*(cell+gap)
            items[name.lower()] = QRectF(x, top+12.0, cell, cell)
        eraser_x = left+margin+len(VIRTUAL_PEN_PALETTE)*(cell+gap)
        items["eraser"] = QRectF(eraser_x, top+12.0, cell, cell)
        pointer_x = eraser_x+(cell+gap)
        items["pointer"] = QRectF(pointer_x, top+12.0, cell, cell)
        clear_x = pointer_x+(cell+gap)
        items["clear"] = QRectF(clear_x, top+12.0, cell, cell)
        return {"panel": panel, "items": items}

    def _virtual_pen_ui_contains(self, point):
        if point is None or self.virtual_scene != "grid":
            return False
        dock = self._virtual_dock_geometry()["dock"]
        if dock.contains(point):
            return True
        if self.virtual_pen_popup_open:
            return self._virtual_pen_popup_geometry()["panel"].contains(point)
        return False

    def _handle_virtual_dock_click(self, point):
        """Handle hand clicks on the Virtual apps dock and Pen palette."""
        if point is None or self.desktop_mode != "virtual" or self.virtual_scene != "grid":
            return False
        # The glass menu is modal, but this handler runs only after its own hit tests.
        if self.menu_open and self.hub_visible:
            return False
        if self.virtual_pen_popup_open:
            popup = self._virtual_pen_popup_geometry()
            if popup["panel"].contains(point):
                for name, rect in popup["items"].items():
                    if not rect.contains(point):
                        continue
                    if name == "eraser":
                        self.virtual_pen_active = True
                        self.virtual_pen_erase = True
                        self.status = "Virtual pen eraser selected — pinch and move to erase"
                    elif name == "pointer":
                        self.virtual_pen_active = False
                        self.virtual_pen_erase = False
                        self.status = "Virtual pen disabled — normal gestures restored"
                    elif name == "clear":
                        active_stroke = self.virtual_pen_active_stroke
                        if active_stroke is not None:
                            active_side = active_stroke.get("side")
                            self.active_push_sides.discard(active_side)
                            if self.gesture_owner == active_side:
                                self.gesture_owner = None
                        self.virtual_pen_active_stroke = None
                        self.virtual_canvas_strokes = []
                        self.virtual_canvas_layer = None
                        self.virtual_canvas_layer_size = None
                        self.virtual_canvas_layer_dirty = True
                        self._save_virtual_desktop_data()
                        self.status = "Virtual canvas cleared"
                    else:
                        color = next(color for label, color in VIRTUAL_PEN_PALETTE
                                     if label.lower() == name)
                        self.virtual_pen_active = True
                        self.virtual_pen_erase = False
                        self.virtual_pen_color = color
                        self.status = f"Virtual pen colour: {name}"
                    self.virtual_pen_popup_open = False
                    self.update()
                    return True
        geometry = self._virtual_dock_geometry()
        if geometry["pen"].contains(point):
            if self.virtual_garden_mode:
                self.virtual_pen_popup_open = False
                self.status = "Pen drawing is available in Grid mode; switch off Garden Mode in Settings → Virtual"
                self.update()
                return True
            self.virtual_pen_popup_open = not self.virtual_pen_popup_open
            self.status = "Choose a pen colour or eraser" if self.virtual_pen_popup_open else "Pen palette closed"
            self.update()
            return True
        if self.virtual_pen_popup_open:
            self.virtual_pen_popup_open = False
            self.update()
            if geometry["dock"].contains(point):
                return True
        return geometry["dock"].contains(point)

    def _virtual_normalized_point(self, point):
        return [min(1.0, max(0.0, float(point.x())/max(1.0, float(self.width())))),
                min(1.0, max(0.0, float(point.y())/max(1.0, float(self.height()))))]

    def _virtual_pixel_point(self, point):
        return QPointF(float(point[0])*self.width(), float(point[1])*self.height())

    def _draw_virtual_stroke(self, painter, stroke, points=None):
        values = stroke.get("points", []) if points is None else points
        if not values:
            return
        pixel_points = [self._virtual_pixel_point(point) for point in values]
        width = max(1.0, float(stroke.get("width", 0.0038))*min(self.width(), self.height()))
        color = QColor(str(stroke.get("color", "#43ddff")))
        if not color.isValid():
            color = QColor("#43ddff")
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if stroke.get("erase", False):
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        if len(pixel_points) == 1:
            painter.drawPoint(pixel_points[0])
        else:
            painter.drawPolyline(QPolygonF(pixel_points))
        painter.restore()

    def _ensure_virtual_canvas_layer(self):
        size = (max(1, self.width()), max(1, self.height()))
        if (self.virtual_canvas_layer is not None and not self.virtual_canvas_layer_dirty
                and self.virtual_canvas_layer_size == size):
            return self.virtual_canvas_layer
        layer = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
        layer.fill(Qt.transparent)
        canvas_painter = QPainter(layer)
        canvas_painter.setRenderHint(QPainter.Antialiasing, True)
        for stroke in self.virtual_canvas_strokes:
            self._draw_virtual_stroke(canvas_painter, stroke)
        active = self.virtual_pen_active_stroke
        if active is not None:
            self._draw_virtual_stroke(canvas_painter, active)
        canvas_painter.end()
        self.virtual_canvas_layer = layer
        self.virtual_canvas_layer_size = size
        self.virtual_canvas_layer_dirty = False
        return layer

    def _append_virtual_pen_point(self, point):
        active = self.virtual_pen_active_stroke
        if active is None or point is None:
            return
        normalized = self._virtual_normalized_point(point)
        points = active["points"]
        if points:
            previous = self._virtual_pixel_point(points[-1])
            if math.hypot(point.x()-previous.x(), point.y()-previous.y()) < 1.5:
                return
        layer = self._ensure_virtual_canvas_layer()
        before_count = len(points)
        points.append(normalized)
        # If the layer was freshly rebuilt, it already contains the stroke as it
        # existed before this point. Add only the newly reached segment.
        if before_count == 0:
            self._draw_virtual_stroke_segment(layer, active, None, normalized)
        else:
            self._draw_virtual_stroke_segment(layer, active, points[-2], normalized)
        self.update()

    def _draw_virtual_stroke_segment(self, layer, stroke, start, end):
        if layer is None or end is None:
            return
        painter = QPainter(layer)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width = max(1.0, float(stroke.get("width", 0.0038))*min(self.width(), self.height()))
        color = QColor(str(stroke.get("color", "#43ddff")))
        if not color.isValid():
            color = QColor("#43ddff")
        painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if stroke.get("erase", False):
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        end_point = self._virtual_pixel_point(end)
        if start is None:
            painter.drawPoint(end_point)
        else:
            painter.drawLine(self._virtual_pixel_point(start), end_point)
        painter.end()

    def _finish_virtual_pen_stroke(self):
        stroke = self.virtual_pen_active_stroke
        if stroke is None:
            return
        self.virtual_pen_active_stroke = None
        if len(stroke.get("points", [])) >= 1:
            stored = {key: stroke[key] for key in ("color", "erase", "width", "points") if key in stroke}
            self.virtual_canvas_strokes.append(stored)
            if len(self.virtual_canvas_strokes) > 2000:
                self.virtual_canvas_strokes = self.virtual_canvas_strokes[-2000:]
            self.virtual_canvas_layer_dirty = True
            self._save_virtual_desktop_data()
        self.gesture_owner = None
        self.active_push_sides.discard(stroke.get("side"))
        self.update()

    def _handle_virtual_pen_frame(self, side, cursor, is_pinch, now):
        """Consume one hand's pinch frames while the Virtual Pen is selected."""
        active = self.virtual_pen_active_stroke
        if active is not None and active.get("side") == side:
            if is_pinch and self.desktop_mode == "virtual":
                active["last_point_at"] = now
                self._append_virtual_pen_point(cursor)
            else:
                self._finish_virtual_pen_stroke()
            self._clear_menu_pinch_sequence(side)
            self.fist_started_at[side] = None
            self.fist_hold_armed[side] = False
            self.fist_action[side] = None
            self.active_action[side] = None
            if self.gesture_owner == side:
                self.gesture_owner = None
            self.action_stop_wait_open[side] = True
            self.action_stop_open_since[side] = None
            return True
        if (self.desktop_mode != "virtual" or self.virtual_scene != "grid" or
                not self.virtual_pen_active or (self.menu_open and self.hub_visible) or self.settings_open or
                self.floating_clipboard_visible or not is_pinch or
                self._virtual_pen_ui_contains(cursor)):
            return False
        self._clear_menu_pinch_sequence(side)
        stroke = {
            "side": side, "color": "#ffffff" if self.virtual_pen_erase else self.virtual_pen_color,
            "erase": bool(self.virtual_pen_erase),
            "width": 0.018 if self.virtual_pen_erase else 0.0038,
            "points": [], "last_point_at": now,
        }
        self.virtual_pen_active_stroke = stroke
        self.gesture_owner = side
        self.canvas_active_side = side
        self.active_push_sides.add(side)
        self.action_stop_wait_open[side] = False
        self._append_virtual_pen_point(cursor)
        return True

    def _paint_virtual_app_dock(self, painter):
        """Draw the bottom Virtual Apps dock and its Pen colour/eraser popup."""
        if (self.desktop_mode != "virtual" or self.virtual_scene != "grid" or
                (self.menu_open and self.hub_visible) or self.settings_open):
            return
        geometry = self._virtual_dock_geometry()
        dock, pen = geometry["dock"], geometry["pen"]
        painter.save()
        # Dock shadow + glass-like ribbon.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 105)))
        painter.drawRoundedRect(dock.translated(0, 4), 18, 18)
        dock_gradient = QLinearGradient(dock.topLeft(), dock.bottomRight())
        dock_gradient.setColorAt(0.0, QColor(25, 63, 82, 238))
        dock_gradient.setColorAt(1.0, QColor(4, 16, 30, 242))
        painter.setBrush(QBrush(dock_gradient))
        painter.setPen(QPen(QColor(103, 214, 236, 210), 1.4))
        painter.drawRoundedRect(dock, 18, 18)
        painter.setPen(QColor(180, 224, 235, 210))
        painter.setFont(self._ui_font(7, QFont.DemiBold))
        painter.drawText(QRectF(dock.left()+12, dock.top()+7, 86, 16), Qt.AlignLeft | Qt.AlignVCenter, "VIRTUAL APPS")
        painter.setPen(QPen(QColor(120, 205, 222, 100), 1.0))
        painter.drawLine(QPointF(dock.left()+101, dock.top()+11), QPointF(dock.left()+101, dock.bottom()-11))
        active_fill = (QColor(49, 139, 158, 235) if self.virtual_pen_active and not self.virtual_garden_mode
                       else QColor(42, 53, 61, 220) if self.virtual_garden_mode
                       else QColor(14, 42, 59, 235))
        painter.setBrush(QBrush(active_fill))
        painter.setPen(QPen(QColor(103, 226, 245, 235), 1.4))
        painter.drawRoundedRect(pen, 12, 12)
        if self.virtual_pen_active and not self.virtual_garden_mode:
            painter.setPen(QPen(QColor(115, 255, 245, 70), 5.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(pen.adjusted(-2, -2, 2, 2), 13, 13)
        # A minimal pen nib/stylus icon, drawn locally (no external image required).
        icon_center = QPointF(pen.center().x(), pen.center().y())
        painter.save()
        painter.translate(icon_center)
        painter.rotate(-42)
        painter.setPen(QPen(QColor(235, 251, 255), 3.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(QPointF(0, -12), QPointF(0, 8))
        painter.setPen(QPen(QColor(67, 221, 255), 3.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(QPointF(0, -12), QPointF(0, -2))
        painter.setPen(QPen(QColor(235, 251, 255), 1.6))
        painter.drawLine(QPointF(-3, 8), QPointF(3, 8))
        painter.restore()
        painter.setPen(QColor(235, 250, 255))
        painter.setFont(self._ui_font(9, QFont.DemiBold))
        painter.drawText(QRectF(pen.right()+9, pen.top()+3, 90, pen.height()-6), Qt.AlignLeft | Qt.AlignVCenter,
                         ("PEN" if not self.virtual_pen_active else
                          ("ERASER" if self.virtual_pen_erase else "PEN ACTIVE")))

        if self.virtual_pen_popup_open and not self.virtual_garden_mode:
            popup = self._virtual_pen_popup_geometry()
            panel = popup["panel"]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 95)))
            painter.drawRoundedRect(panel.translated(0, 3), 15, 15)
            gradient = QLinearGradient(panel.topLeft(), panel.bottomRight())
            gradient.setColorAt(0, QColor(22, 57, 76, 248))
            gradient.setColorAt(1, QColor(5, 17, 30, 250))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(101, 220, 242, 220), 1.3))
            painter.drawRoundedRect(panel, 15, 15)
            for name, rect in popup["items"].items():
                if name == "eraser":
                    selected = self.virtual_pen_erase and self.virtual_pen_active
                    painter.setBrush(QBrush(QColor(185, 83, 91, 235) if selected else QColor(46, 57, 70, 235)))
                    painter.setPen(QPen(QColor(255, 143, 148, 235), 1.1))
                    painter.drawRoundedRect(rect, 7, 7)
                    painter.setPen(QPen(QColor(255, 240, 240), 2.2, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(QPointF(rect.left()+7, rect.bottom()-8), QPointF(rect.right()-7, rect.top()+8))
                elif name == "clear":
                    painter.setBrush(QBrush(QColor(156, 43, 53, 240)))
                    painter.setPen(QPen(QColor(255, 125, 132, 240), 1.2))
                    painter.drawRoundedRect(rect, 7, 7)
                    painter.setPen(QPen(QColor(255, 244, 244), 2.2, Qt.SolidLine, Qt.RoundCap))
                    # Bin body, lid and handle.
                    painter.drawLine(QPointF(rect.left()+9, rect.top()+12), QPointF(rect.right()-9, rect.top()+12))
                    painter.drawLine(QPointF(rect.left()+12, rect.top()+14), QPointF(rect.left()+13, rect.bottom()-7))
                    painter.drawLine(QPointF(rect.right()-12, rect.top()+14), QPointF(rect.right()-13, rect.bottom()-7))
                    painter.drawLine(QPointF(rect.left()+13, rect.bottom()-7), QPointF(rect.right()-13, rect.bottom()-7))
                    painter.drawLine(QPointF(rect.center().x()-3, rect.top()+9), QPointF(rect.center().x()+3, rect.top()+9))
                elif name == "pointer":
                    selected = not self.virtual_pen_active
                    painter.setBrush(QBrush(QColor(45, 115, 139, 235) if selected else QColor(46, 57, 70, 235)))
                    painter.setPen(QPen(QColor(145, 224, 239, 235), 1.1))
                    painter.drawRoundedRect(rect, 7, 7)
                    painter.setPen(QPen(QColor(242, 252, 255), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.drawLine(QPointF(rect.left()+8, rect.top()+6), QPointF(rect.left()+8, rect.bottom()-7))
                    painter.drawLine(QPointF(rect.left()+8, rect.top()+6), QPointF(rect.right()-7, rect.top()+15))
                    painter.drawLine(QPointF(rect.left()+8, rect.bottom()-7), QPointF(rect.left()+17, rect.bottom()-12))
                else:
                    color = next(color for label, color in VIRTUAL_PEN_PALETTE if label.lower() == name)
                    painter.setBrush(QBrush(QColor(color)))
                    is_selected = self.virtual_pen_active and not self.virtual_pen_erase and self.virtual_pen_color.lower() == color.lower()
                    painter.setPen(QPen(QColor(235, 255, 255, 245) if is_selected else QColor(90, 145, 163, 190), 2 if is_selected else 1))
                    painter.drawEllipse(rect.adjusted(3, 3, -3, -3))
                    if is_selected:
                        painter.setPen(QPen(QColor(235, 255, 255, 230), 1.2))
                        painter.drawEllipse(rect.adjusted(0, 0, 0, 0))
        painter.restore()

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

    def _main_menu_exit_rect(self, cx=None, cy=None, scale=None):
        """Wide Quit Freehand button at the top of the main glass menu."""
        cx = self.hub_pos.x()*self.width() if cx is None else cx
        cy = self.hub_pos.y()*self.height() if cy is None else cy
        scale = self.menu_ui_scale() if scale is None else scale
        _panel_width, panel_height = self._menu_panel_dimensions()
        return QRectF(cx-175.0*scale, cy+(18.0-panel_height*0.5)*scale,
                      350.0*scale, 36.0*scale)

    def _tli_calibration_button_rects(self, cx=None, cy=None, scale=None):
        """Left/right TPI calibration buttons, taken from the shared Hands-page layout."""
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
        hand_fps_row_top = None
        if page == "interaction":
            stack("scroll_toggle")
            stack("swap_drag_scroll_toggle")
            stack("priority_toggle")
        elif page == "hands":
            stack("render_style")
            stack("skin_tint")
            stack("disabled_indicator_toggle")
            y += group
            stack("calibrate_left")
            stack("calibrate_right")
            y += group
            stack("hand_setup")
            # Hand display FPS belongs below the left-side buttons, not among
            # the recognition and appearance sliders on the right.
            hand_fps_row_top = y + 10.0
            y += 74.0
        elif page == "tli":
            stack("glow_toggle")
            y += 6.0
            half = (W-gap)*0.5
            buttons["swatch_left"] = (L, y, half, 70.0)
            buttons["swatch_right"] = (L+half+gap, y, half, 70.0)
            y += 70.0 + gap
        elif page == "sounds":
            for key, _label in SOUND_SETTING_ROWS:
                stack("sound:" + key, h=43.0, after=7.0)
            texts["help"] = (L, y, W, 42.0)
            y += 42.0 + gap
            right_h = max(right_h, 340.0)
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
        elif page == "virtual":
            stack("virtual_scene_switch")
            stack("virtual_grid_toggle")
            stack("virtual_background_style")
            texts["help"] = (L, y, W, 58.0)
            y += 58.0 + gap
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
        right_index = 0
        for row in self._settings_slider_rows(page):
            name = row[0]
            if page == "hands" and name == "hand_display_fps":
                # Reuse the same draw/hover/drag code as every other slider,
                # with its own left-column coordinates and button-safe spacing.
                row_top = hand_fps_row_top
                track_left, track_right = L+12.0, L+W-12.0
                label_x = L
            else:
                row_top = right_index*self.SETTINGS_SLIDER_PITCH
                right_index += 1
                track_left, track_right = R+12.0, R+W-12.0
                label_x = R
            if name == "scroll_notch_hue":
                track_right -= 30.0  # room for the colour swatch at the track end
            sliders.append({"name": name, "y": row_top+12.0, "left": track_left, "right": track_right,
                            "label": (label_x, row_top+22.0, W, 18.0)})
        if right_index:
            right_h = max(right_h, right_index*self.SETTINGS_SLIDER_PITCH - 12.0)

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

    def _virtual_scene_segments(self, rect, scale):
        """Matching three-way scene selector, with the same hit boxes as its painting."""
        area = rect.adjusted(rect.width()*self.RENDER_SWITCH_CAPTION, 3*scale, -3*scale, -3*scale)
        width = area.width()/len(VIRTUAL_SCENES)
        return [(style, QRectF(area.left()+i*width, area.top(), width, area.height()))
                for i, style in enumerate(VIRTUAL_SCENES)]

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
        widths = [82.0, 55.0, 48.0, 58.0, 76.0, 52.0, 48.0, 58.0, 58.0, 50.0]
        total = sum(widths)*scale + tab_gap*9.0
        left = cx-total*0.5
        _panel_width, panel_height = self._menu_panel_dimensions()
        y = cy-(panel_height*0.5-25.0)*scale
        # Keep UPDATE as the final tab at the far right of the strip.
        names = ("interaction", "hands", "tli", "notifications", "gestures", "voice", "d1", "sounds", "virtual", "update")
        rects = {}
        x = left
        for name, width in zip(names, widths):
            rects[name] = QRectF(x, y, width*scale, tab_height)
            x += (width*scale) + tab_gap
        return rects

    def _sound_dropdown_layout(self, cx, cy, scale):
        # Persistent list in the right column, rather than a popup over the rows.
        key = self.sound_dropdown_key or SOUND_SETTING_ROWS[0][0]
        layout = self._settings_layout("sounds")
        right = self.SETTINGS_RIGHT_X
        top = layout["top"] + 2.0
        item_h = 29.0 * scale
        options = ["None", *available_ui_sounds()]
        max_items = max(1, int(330.0 / 29.0))
        options = options[:max_items]
        left = cx + right*scale
        panel_top = cy + top*scale
        width = self.SETTINGS_COLUMN_W*scale
        panel = QRectF(left, panel_top, width, (len(options)+1)*item_h)
        items = [(value, QRectF(left, panel_top+(i+1)*item_h, width, item_h))
                 for i, value in enumerate(options)]
        return {"panel": panel, "items": items, "key": key,
                "header": QRectF(left, panel_top, width, item_h)}

    def _paint_sound_dropdown(self, painter, cx, cy, scale):
        layout = self._sound_dropdown_layout(cx, cy, scale)
        panel = layout["panel"]
        painter.setBrush(QBrush(QColor(5, 22, 35, 238)))
        painter.setPen(QPen(QColor(155, 245, 255, 210), 1.5*scale))
        painter.drawRoundedRect(panel, 10*scale, 10*scale)
        painter.setPen(QColor(235, 250, 255))
        painter.setFont(self._ui_font(max(7, int(9*scale)), QFont.DemiBold))
        heading = dict(SOUND_SETTING_ROWS).get(layout["key"], "SOUND")
        painter.drawText(layout["header"].adjusted(10*scale, 0, -6*scale, 0),
                         Qt.AlignVCenter | Qt.AlignLeft, heading + "  ·  CHOOSE SOUND")
        selected = self.ui_preferences.get(layout["key"], "None")
        for value, rect in layout["items"]:
            painter.setBrush(QBrush(QColor(40, 115, 82, 235) if value == selected else QColor(18, 55, 75, 220)))
            painter.setPen(QPen(QColor(110, 200, 220, 100), 0.7*scale))
            painter.drawRect(rect)
            painter.setPen(QColor(240, 252, 255))
            painter.setFont(self._ui_font(max(7, int(9*scale)), QFont.DemiBold))
            label = "None (silent)" if value == "None" else value
            self._draw_fitted_button_label(painter, rect, label, base_size=max(7, int(9*scale)), min_size=6, inset=9*scale)

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
            self.sound_dropdown_key = None
            if self.custom_gesture_key_capture is not None:
                self._cancel_custom_gesture_key_capture("Shortcut capture cancelled because the menu closed.")
            else:
                self._stop_native_custom_key_capture()
            # The search launcher is a companion to the glass menu. Never leave it
            # floating over the desktop after a gesture closes the glass menu.
            search_dialog = getattr(self, "search_dialog", None)
            if search_dialog is not None and search_dialog.isVisible():
                search_dialog.close()
        if reset_settings:
            self.settings_open = False
        if changed:
            play_ui_sound(self.ui_preferences.get("sound_glass_toggle", "None"))
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
                ("tli_glow_intensity", self.tli_glow_intensity, 0.0, 2.5, "TPI GLOW INTENSITY"),
                ("tli_colour_shift_degrees", self.tli_colour_shift_degrees, 0.0, 360.0, "TPI COLOUR / HUE"),
                ("tli_border_thickness", self.tli_border_thickness, 1.5, 8.0, "TPI BORDER THICKNESS"),
                ("tli_diameter", self.tli_diameter, 12.0, 96.0, "TPI DIAMETER"),
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
        if page == "virtual":
            return [
                ("virtual_grid_brightness", self.virtual_grid_brightness, 10.0, 100.0, "GRID BRIGHTNESS"),
                ("virtual_grid_glow", self.virtual_grid_glow, 0.0, 100.0, "GRID GLOW"),
                ("virtual_grid_spacing", self.virtual_grid_spacing, 24.0, 160.0, "GRID SPACING"),
                ("virtual_grid_thickness", self.virtual_grid_thickness, 0.5, 3.0, "GRID LINE THICKNESS"),
            ]
        if page == "hands":
            return [
                ("hand_display_fps", self.hand_display_fps, 5.0, 30.0, "HAND DISPLAY FPS"),
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

    def _clipboard_payload_from_mime(self, mime):
        """Snapshot ordinary system clipboard contents into a portable history entry."""
        if mime is None:
            return None
        image = None
        if mime.hasImage():
            try:
                value = mime.imageData()
                if isinstance(value, QImage):
                    image = value.copy()
                elif hasattr(value, "toImage"):
                    image = value.toImage().copy()
                elif value is not None:
                    image = QImage(value).copy()
                if image is not None and image.isNull():
                    image = None
            except Exception:
                image = None
        try:
            urls = [QUrl(url) for url in mime.urls()] if mime.hasUrls() else []
        except Exception:
            urls = []
        try:
            text = str(mime.text() or "") if mime.hasText() else ""
        except Exception:
            text = ""
        try:
            html = str(mime.html() or "") if mime.hasHtml() else ""
        except Exception:
            html = ""
        if image is not None:
            kind = "image"
        elif urls:
            kind = "files" if all(url.isLocalFile() for url in urls) else "links"
            # Explorer/Finder file-copy operations usually expose image files as URLs,
            # not as imageData. Load a lightweight preview while retaining the original
            # URLs as the actual drag payload so the copied file remains a file.
            if kind == "files":
                image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
                for url in urls[:8]:
                    if not url.isLocalFile():
                        continue
                    local_path = url.toLocalFile()
                    if Path(local_path).suffix.lower() not in image_suffixes:
                        continue
                    try:
                        candidate = QImage(local_path)
                        if not candidate.isNull():
                            image = candidate
                            break
                    except Exception:
                        pass
        elif text.strip() or html.strip():
            kind = "text"
        else:
            return None
        # Images are hashed from a tiny thumbnail, avoiding work on full-resolution
        # screenshots while still deduplicating repeated QClipboard notifications.
        parts = [kind, text[:200000], html[:200000], "\n".join(url.toString() for url in urls)]
        if image is not None:
            try:
                sample = image.scaled(12, 12, Qt.KeepAspectRatio, Qt.FastTransformation).convertToFormat(QImage.Format_RGBA8888)
                buffer = QBuffer()
                buffer.open(QIODevice.WriteOnly)
                sample.save(buffer, "PNG")
                parts.append(hashlib.sha1(bytes(buffer.data())).hexdigest())
                buffer.close()
                parts.append(f"{image.width()}x{image.height()}")
            except Exception:
                parts.append(f"{image.width()}x{image.height()}")
        signature = hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()
        return {"kind": kind, "text": text[:1000000], "html": html[:1000000],
                "image": image, "urls": urls, "signature": signature}

    def _capture_system_clipboard(self):
        """Record new OS clipboard content, including text, file URLs, and copied images."""
        clipboard = getattr(self, "_system_clipboard", None)
        if clipboard is None:
            try:
                clipboard = QApplication.clipboard()
            except Exception:
                return
        try:
            payload = self._clipboard_payload_from_mime(clipboard.mimeData())
        except Exception:
            return
        if payload is None:
            return
        now = time.monotonic()
        if now < getattr(self, "_clipboard_capture_suppression_until", 0.0):
            return
        signature = payload["signature"]
        # QClipboard can emit dataChanged more than once for a single native copy.
        if signature == self._clipboard_last_signature and now-self._clipboard_last_signature_at < 1.0:
            return
        if self.floating_clipboard_entries and self.floating_clipboard_entries[-1].get("signature") == signature:
            self._clipboard_last_signature = signature
            self._clipboard_last_signature_at = now
            return
        entry = {
            "id": uuid.uuid4().hex,
            "kind": payload["kind"],
            "text": payload["text"],
            "html": payload.get("html", ""),
            "image": payload["image"],
            "urls": payload["urls"],
            # A multi-file copy stays one history card and one transferable URL group.
            "group_count": len(payload["urls"]) if payload["kind"] == "files" else 1,
            "signature": signature,
            "created": datetime.now(),
            "image_path": None,
        }
        if entry["image"] is not None:
            original_image = entry["image"]
            entry["image_width"] = int(original_image.width())
            entry["image_height"] = int(original_image.height())
            # Keep original pixels on disk for clipboard/drop fidelity, but keep only a
            # modest preview in RAM so a run of copied 4K screenshots doesn't balloon
            # memory use. Image files copied from Explorer retain their original URL.
            if entry["kind"] == "image":
                path = self.floating_clipboard_temp_dir / f"clipboard-{entry['id']}.png"
                try:
                    if original_image.save(str(path), "PNG"):
                        entry["image_path"] = str(path)
                except Exception:
                    pass
            if (max(original_image.width(), original_image.height()) > 640 and
                    (entry["kind"] != "image" or entry["image_path"])):
                try:
                    entry["image"] = original_image.scaled(
                        640, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                except Exception:
                    pass
        self.floating_clipboard_entries.append(entry)
        evicted = self.floating_clipboard_entries[:-self.floating_clipboard_max_entries]
        del self.floating_clipboard_entries[:-self.floating_clipboard_max_entries]
        # Remove cached bitmap previews for history items that rolled out of the cap.
        for stale in evicted:
            stale_path = stale.get("image_path")
            if stale_path:
                try:
                    if Path(stale_path).resolve().parent == self.floating_clipboard_temp_dir.resolve():
                        Path(stale_path).unlink(missing_ok=True)
                except Exception:
                    pass
        self.floating_clipboard_scroll_offset = 0
        self._clipboard_last_signature = signature
        self._clipboard_last_signature_at = now
        self.update()
        overlay = getattr(self, "search_hand_overlay", None)
        if overlay is not None:
            overlay.update()

    def _delete_floating_clipboard_entry(self, entry_id):
        """Remove one clipboard-history item and its private cached bitmap, if any."""
        entry = next((item for item in self.floating_clipboard_entries
                      if item.get("id") == entry_id), None)
        if entry is None:
            return False
        self.floating_clipboard_entries = [
            item for item in self.floating_clipboard_entries if item.get("id") != entry_id]
        # Cancel an in-flight drag of the removed item so it cannot paste stale content.
        for side, state in self.floating_clipboard_drag_state.items():
            if state and state.get("type") == "entry" and state.get("entry_id") == entry_id:
                self.floating_clipboard_drag_state[side] = None
        self.floating_clipboard_drag_running = any(
            bool(state and state.get("running")) for state in self.floating_clipboard_drag_state.values())
        stale_path = entry.get("image_path")
        if stale_path:
            try:
                if Path(stale_path).resolve().parent == self.floating_clipboard_temp_dir.resolve():
                    Path(stale_path).unlink(missing_ok=True)
            except Exception:
                pass
        self.floating_clipboard_scroll_offset = min(
            int(self.floating_clipboard_scroll_offset), max(0, len(self.floating_clipboard_entries)-1))
        self.status = "Clipboard item deleted"
        self.update()
        overlay = getattr(self, "search_hand_overlay", None)
        if overlay is not None:
            overlay.update()
        return True

    def _set_floating_clipboard_visible(self, visible):
        self.floating_clipboard_visible = bool(visible)
        if self.floating_clipboard_visible:
            self.floating_clipboard_scroll_offset = 0
            self.status = f"Floating clipboard shown — {len(self.floating_clipboard_entries)} item(s); grab or pinch an item to drag it"
        else:
            for side, state in self.floating_clipboard_drag_state.items():
                if state and not state.get("running"):
                    self.floating_clipboard_drag_state[side] = None
            self.status = "Floating clipboard hidden"
        self.update()
        overlay = getattr(self, "search_hand_overlay", None)
        if overlay is not None:
            overlay.update()

    def _position_floating_clipboard_over_rect(self, anchor_rect):
        """Place the clipboard centred over a just-used Pinch Menu rectangle."""
        if anchor_rect is None or not self.floating_clipboard_visible:
            return
        layout = self._floating_clipboard_layout()
        if layout is None:
            return
        panel = layout["panel"]
        x = anchor_rect.center().x() - panel.width() * 0.5
        y = anchor_rect.center().y() - panel.height() * 0.5
        x = max(4.0, min(x, max(4.0, self.width() - panel.width() - 4.0)))
        y = max(4.0, min(y, max(4.0, self.height() - panel.height() - 4.0)))
        self.floating_clipboard_position = QPointF(x, y)
        self.update()
        overlay = getattr(self, "search_hand_overlay", None)
        if overlay is not None:
            overlay.update()

    def _floating_clipboard_layout(self):
        """Return the floating PowerPoint-style thumbnail strip in screen-local coords."""
        if not self.floating_clipboard_visible or self.width() <= 0 or self.height() <= 0:
            return None
        scale = min(1.0, max(0.72, min(self.width()/1440.0, self.height()/800.0)))
        panel_w = min(292.0*scale, max(222.0*scale, self.width()*0.205))
        pad = 10.0*scale
        header_h = 46.0*scale
        card_h = 88.0*scale
        footer_h = 34.0*scale
        newest_first = list(reversed(self.floating_clipboard_entries))
        start = max(0, min(int(self.floating_clipboard_scroll_offset), max(0, len(newest_first)-1)))
        visible = newest_first[start:start+5]
        content_h = len(visible)*card_h if visible else 75.0*scale
        panel_h = pad + header_h + content_h + footer_h + pad
        panel_h = min(panel_h, max(160.0*scale, self.height()*0.82))
        default_x = max(8.0, self.width()-panel_w-22.0*scale)
        default_y = max(8.0, min((self.height()-panel_h)*0.5, self.height()-panel_h-8.0))
        saved_position = getattr(self, "floating_clipboard_position", None)
        if saved_position is None:
            x, y = default_x, default_y
        else:
            x = max(4.0, min(float(saved_position.x()), max(4.0, self.width()-panel_w-4.0)))
            y = max(4.0, min(float(saved_position.y()), max(4.0, self.height()-panel_h-4.0)))
        panel = QRectF(x, y, panel_w, panel_h)
        header = QRectF(panel.left()+9*scale, panel.top()+6*scale, panel.width()-18*scale, 38*scale)
        close_rect = QRectF(panel.right()-30*scale, panel.top()+8*scale, 22*scale, 24*scale)
        cards = []
        delete_buttons = {}
        card_y = panel.top()+pad+header_h
        if visible:
            for entry in visible:
                card_rect = QRectF(panel.left()+pad, card_y, panel.width()-2*pad, card_h-5*scale)
                cards.append((entry, card_rect))
                # Red delete button fills the card's full height and sits inside its right edge.
                delete_buttons[entry.get("id")] = QRectF(
                    card_rect.right()-32*scale, card_rect.top(), 32*scale, card_rect.height())
                card_y += card_h
        older_rect = QRectF(panel.left()+pad, panel.bottom()-footer_h, (panel.width()-3*pad)/2, footer_h-2*scale)
        newer_rect = QRectF(older_rect.right()+pad, older_rect.top(), older_rect.width(), older_rect.height())
        return {"panel": panel, "header": header, "cards": cards, "delete_buttons": delete_buttons,
                "close": close_rect, "older": older_rect, "newer": newer_rect, "scale": scale,
                "empty": not bool(visible), "start": start, "count": len(newest_first)}

    def _floating_clipboard_contains(self, point):
        layout = self._floating_clipboard_layout()
        return bool(layout is not None and point is not None and layout["panel"].contains(point))

    def _floating_clipboard_hit_test(self, point):
        layout = self._floating_clipboard_layout()
        if layout is None or point is None:
            return None
        for name in ("close", "older", "newer"):
            if layout[name].contains(point):
                return {"type": "control", "name": name}
        for entry, rect in layout["cards"]:
            delete_rect = layout["delete_buttons"].get(entry.get("id"))
            if delete_rect is not None and delete_rect.contains(point):
                return {"type": "control", "name": "delete", "entry_id": entry.get("id")}
        # The title bar is a drag handle; the close button is tested above it.
        if layout["header"].contains(point):
            return {"type": "control", "name": "header"}
        for entry, rect in layout["cards"]:
            if rect.contains(point):
                return {"type": "entry", "entry": entry}
        return None

    def _paint_floating_clipboard(self, painter, now=None):
        layout = self._floating_clipboard_layout()
        if layout is None:
            return
        panel, scale = layout["panel"], layout["scale"]
        painter.save()
        painter.setPen(QPen(QColor(112, 232, 248, 210), 1.5*scale))
        painter.setBrush(QBrush(QColor(4, 17, 31, 229)))
        painter.drawRoundedRect(panel, 15*scale, 15*scale)
        header = QRectF(panel.left()+9*scale, panel.top()+6*scale, panel.width()-18*scale, 38*scale)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(22, 66, 83, 245)))
        painter.drawRoundedRect(header, 10*scale, 10*scale)
        self._paint_clipboard_icon(painter, QRectF(header.left()+5*scale, header.top()+1*scale,
                                                   30*scale, header.height()-2*scale), True)
        painter.setPen(QColor(230, 253, 255, 250))
        painter.setFont(self._ui_font(9, QFont.DemiBold))
        painter.drawText(QRectF(header.left()+37*scale, header.top(), header.width()-75*scale, header.height()),
                         Qt.AlignVCenter | Qt.AlignLeft,
                         f"FLOATING CLIPBOARD  ·  {layout['count']}")
        close_rect = layout["close"]
        close_hovered = any(point is not None and close_rect.contains(point)
                            for point in self.side_cursor_positions.values())
        close_gradient = QLinearGradient(close_rect.topLeft(), close_rect.bottomRight())
        if close_hovered:
            for expansion, alpha in ((7*scale, 45), (3.5*scale, 90)):
                painter.setPen(QPen(QColor(255, 90, 90, alpha), 3*scale))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(close_rect.adjusted(-expansion, -expansion, expansion, expansion),
                                        5*scale+expansion, 5*scale+expansion)
            close_gradient.setColorAt(0.0, QColor(255, 105, 100, 245))
            close_gradient.setColorAt(1.0, QColor(215, 30, 40, 250))
            painter.setPen(QPen(QColor(255, 215, 210, 250), 2.0*scale))
        else:
            close_gradient.setColorAt(0.0, QColor(200, 55, 60, 215))
            close_gradient.setColorAt(1.0, QColor(95, 12, 22, 235))
            painter.setPen(QPen(QColor(255, 150, 150, 210), 1.5*scale))
        painter.setBrush(QBrush(close_gradient))
        painter.drawRoundedRect(close_rect, 5*scale, 5*scale)
        painter.setPen(QColor(255, 247, 247, 250))
        painter.setFont(self._ui_font(13, QFont.Bold))
        painter.drawText(close_rect, Qt.AlignCenter, "×")
        # A subtle grip indicates that dragging the title bar moves the panel.
        grip_x = header.right()-37*scale
        painter.setPen(QPen(QColor(165, 223, 236, 205), 1.4*scale, Qt.SolidLine, Qt.RoundCap))
        for grip_y in (header.center().y()-5*scale, header.center().y(), header.center().y()+5*scale):
            painter.drawLine(QPointF(grip_x, grip_y), QPointF(grip_x+8*scale, grip_y))
        for index, (entry, rect) in enumerate(layout["cards"]):
            painter.setPen(QPen(QColor(91, 177, 198, 190), 1.0*scale))
            painter.setBrush(QBrush(QColor(11, 37, 53, 242)))
            painter.drawRoundedRect(rect, 8*scale, 8*scale)
            image_rect = QRectF(rect.left()+6*scale, rect.top()+6*scale, 62*scale, rect.height()-12*scale)
            image = entry.get("image")
            if image is not None and not image.isNull():
                painter.setPen(QPen(QColor(122, 220, 237, 200), 1.0*scale))
                painter.setBrush(QBrush(QColor(4, 15, 25, 240)))
                painter.drawRoundedRect(image_rect, 5*scale, 5*scale)
                thumb = image.scaled(max(1, int(image_rect.width()-4*scale)),
                                     max(1, int(image_rect.height()-4*scale)),
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawImage(QPointF(image_rect.center().x()-thumb.width()/2,
                                          image_rect.center().y()-thumb.height()/2), thumb)
            else:
                icon_rect = image_rect.adjusted(14*scale, 13*scale, -14*scale, -13*scale)
                painter.setPen(QPen(QColor(139, 236, 244, 225), 1.6*scale))
                painter.setBrush(QBrush(QColor(17, 67, 82, 220)))
                painter.drawRoundedRect(icon_rect, 4*scale, 4*scale)
                painter.setPen(QPen(QColor(139, 236, 244, 225), 1.3*scale))
                for row in range(3):
                    yy = icon_rect.top()+(row+1)*icon_rect.height()/4
                    painter.drawLine(QPointF(icon_rect.left()+4*scale, yy),
                                     QPointF(icon_rect.right()-4*scale, yy))
            delete_rect = layout["delete_buttons"].get(entry.get("id"))
            text_left = image_rect.right()+8*scale
            text_right = (delete_rect.left()-5*scale) if delete_rect is not None else rect.right()-5*scale
            text_rect = QRectF(text_left, rect.top()+7*scale, max(1.0, text_right-text_left), rect.height()-14*scale)
            if entry.get("kind") == "image":
                kind_label = "IMAGE"
                detail = f"{entry.get('image_width', entry['image'].width())} × {entry.get('image_height', entry['image'].height())} px" if image is not None else "Image"
            elif entry.get("kind") == "files":
                names = [Path(url.toLocalFile()).name for url in entry.get("urls", []) if url.isLocalFile()]
                file_count = len(entry.get("urls", []))
                kind_label = f"FILES · {file_count}" if file_count > 1 else "FILES"
                # Keep grouped copies as one card, but make every member discoverable
                # in the card's preview text rather than creating one entry per file.
                detail = " · ".join(names) or (f"{file_count} copied files" if file_count else "Copied file(s)")
            elif entry.get("kind") == "links":
                kind_label = "LINK / URL"
                detail = entry.get("text") or ", ".join(url.toString() for url in entry.get("urls", []))
            else:
                kind_label = "TEXT"
                detail = entry.get("text", "") or re.sub(r"<[^>]+>", " ", entry.get("html", ""))
            painter.setPen(QColor(132, 226, 239, 245))
            painter.setFont(self._ui_font(7, QFont.DemiBold))
            painter.drawText(QRectF(text_rect.left(), text_rect.top(), text_rect.width(), 15*scale),
                             Qt.AlignLeft | Qt.AlignVCenter, kind_label)
            painter.setPen(QColor(224, 242, 247, 245))
            painter.setFont(self._ui_font(7, QFont.Normal))
            metrics = QFontMetricsF(painter.font())
            detail_lines = str(detail or "").replace("\r", " ").splitlines() or [""]
            line_h = 14*scale
            for line_index in range(min(3, max(1, int((text_rect.height()-17*scale)/line_h)))):
                line = detail_lines[line_index] if line_index < len(detail_lines) else ""
                line = metrics.elidedText(line, Qt.ElideRight, int(text_rect.width()))
                y_line = text_rect.top()+16*scale+line_index*line_h
                painter.drawText(QRectF(text_rect.left(), y_line, text_rect.width(), line_h),
                                 Qt.AlignLeft | Qt.AlignVCenter, line)
            # Full-height red delete affordance for this history item.
            if delete_rect is not None:
                delete_hovered = any(point is not None and delete_rect.contains(point)
                                     for point in self.side_cursor_positions.values())
                delete_gradient = QLinearGradient(delete_rect.topLeft(), delete_rect.bottomRight())
                if delete_hovered:
                    painter.setPen(QPen(QColor(255, 100, 100, 100), 2.5*scale))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(delete_rect.adjusted(-2*scale, -2*scale, 2*scale, 2*scale),
                                            6*scale, 6*scale)
                    delete_gradient.setColorAt(0.0, QColor(255, 95, 95, 250))
                    delete_gradient.setColorAt(1.0, QColor(210, 24, 35, 255))
                    painter.setPen(QPen(QColor(255, 225, 225, 250), 1.5*scale))
                else:
                    delete_gradient.setColorAt(0.0, QColor(192, 48, 56, 225))
                    delete_gradient.setColorAt(1.0, QColor(92, 13, 25, 245))
                    painter.setPen(QPen(QColor(255, 165, 165, 225), 1.0*scale))
                painter.setBrush(QBrush(delete_gradient))
                painter.drawRoundedRect(delete_rect, 5*scale, 5*scale)
                painter.setPen(QColor(255, 247, 247, 250))
                painter.setFont(self._ui_font(14, QFont.Bold))
                painter.drawText(delete_rect, Qt.AlignCenter, "×")
        if layout["empty"]:
            empty_rect = QRectF(panel.left()+12*scale, panel.top()+56*scale,
                                panel.width()-24*scale, 67*scale)
            painter.setPen(QColor(178, 213, 225, 235))
            painter.setFont(self._ui_font(8, QFont.Normal))
            painter.drawText(empty_rect, Qt.AlignCenter | Qt.TextWordWrap,
                             "Copy text, images or files to collect them here.\nHold a pinch/grab and move to drag an item.")
        for name, rect, label in (("older", layout["older"], "↑ OLDER"),
                                  ("newer", layout["newer"], "↓ NEWER")):
            enabled = (layout["start"]+5 < layout["count"]) if name == "older" else layout["start"] > 0
            painter.setPen(QPen(QColor(88, 220, 240, 180 if enabled else 60), 1.0*scale))
            painter.setBrush(QBrush(QColor(18, 58, 75, 220 if enabled else 100)))
            painter.drawRoundedRect(rect, 6*scale, 6*scale)
            painter.setPen(QColor(216, 248, 252, 235 if enabled else 100))
            painter.setFont(self._ui_font(7, QFont.DemiBold))
            painter.drawText(rect, Qt.AlignCenter, label)

        # Draw a movable card beneath the live hand while an entry is being dragged.
        # The OS cursor alone is not enough feedback: without this preview it looked as
        # if the clipboard card stayed behind even though its drag state was following
        # the virtual hand. Keep the pixmap cached in the gesture state to avoid per-frame
        # image/text rendering work.
        for side, state in getattr(self, "floating_clipboard_drag_state", {}).items():
            if not state or state.get("type") != "entry" or not state.get("moved"):
                continue
            cursor = state.get("last_cursor")
            pixmap = state.get("preview_pixmap")
            if cursor is None or pixmap is None or pixmap.isNull():
                continue
            preview_w, preview_h = 220*scale, 77*scale
            px = min(max(4.0, cursor.x()+16*scale), max(4.0, self.width()-preview_w-4.0))
            py = min(max(4.0, cursor.y()+16*scale), max(4.0, self.height()-preview_h-4.0))
            painter.save()
            painter.setOpacity(0.94)
            painter.drawPixmap(QRectF(px, py, preview_w, preview_h), pixmap, QRectF(pixmap.rect()))
            painter.restore()
        painter.restore()

    def _clipboard_full_image(self, entry):
        """Return original-resolution image data, loading cached pixels only on demand."""
        image_path = entry.get("image_path") if entry else None
        if image_path:
            try:
                full = QImage(image_path)
                if not full.isNull():
                    return full
            except Exception:
                pass
        return entry.get("image") if entry else None

    def _clipboard_mime_for_entry(self, entry, for_drag=False):
        mime = QMimeData()
        image = self._clipboard_full_image(entry) if entry.get("kind") == "image" else entry.get("image")
        urls = list(entry.get("urls") or [])
        if entry.get("kind") == "image" and image is not None and not image.isNull():
            mime.setImageData(image)
            if for_drag and entry.get("image_path"):
                # A local file URL lets Explorer/Finder/file managers copy the bitmap to
                # folders, while imageData simultaneously supports chat and browser drops.
                urls = [QUrl.fromLocalFile(entry["image_path"])]
        if urls:
            mime.setUrls(urls)
        html = str(entry.get("html") or "")
        text = str(entry.get("text") or "")
        if html and entry.get("kind") != "image":
            mime.setHtml(html)
        if text and (entry.get("kind") != "image" or not for_drag):
            mime.setText(text)
        return mime

    def _set_system_clipboard_from_entry(self, entry):
        if entry is None:
            return False
        try:
            clipboard = QApplication.clipboard()
            # Selecting from our own history must not immediately create a duplicate
            # history entry via the clipboard's dataChanged signal.
            self._clipboard_capture_suppression_until = time.monotonic()+0.8
            if entry.get("kind") == "image" and entry.get("image") is not None:
                clipboard.setImage(self._clipboard_full_image(entry))
            else:
                clipboard.setMimeData(self._clipboard_mime_for_entry(entry, for_drag=False))
            return True
        except Exception as exc:
            self.status = f"Could not select clipboard item: {exc}"
            return False

    def _set_system_clipboard_for_drop(self, entry):
        """Publish drag formats for rejected-drop fallback (image data plus a file URL)."""
        if entry is None:
            return False
        try:
            mime = self._clipboard_mime_for_entry(entry, for_drag=True)
            self._clipboard_capture_suppression_until = time.monotonic()+0.8
            QApplication.clipboard().setMimeData(mime)
            return True
        except Exception as exc:
            self.status = f"Could not prepare clipboard drop: {exc}"
            return False

    def _clipboard_drag_pixmap(self, entry):
        pixmap = QPixmap(250, 88)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(116, 231, 246, 235), 1.5))
        painter.setBrush(QBrush(QColor(5, 26, 41, 238)))
        painter.drawRoundedRect(QRectF(1, 1, 248, 86), 10, 10)
        image = entry.get("image")
        if image is not None and not image.isNull():
            thumb = image.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawImage(QPointF(7+(70-thumb.width())/2, 9+(70-thumb.height())/2), thumb)
        if entry.get("kind") == "files":
            urls = list(entry.get("urls") or [])
            names = [Path(url.toLocalFile()).name for url in urls if url.isLocalFile()]
            count = len(urls)
            label = (f"{count} files · " + ", ".join(names[:3])) if count > 1 else (names[0] if names else "Copied file")
        elif entry.get("kind") == "image" and image is not None and not image.isNull():
            label = f"Image · {entry.get('image_width', image.width())} × {entry.get('image_height', image.height())}"
        else:
            label = str(entry.get("text") or "Copied item") or "Copied item"
        label = QFontMetricsF(QFont("Sans", 9)).elidedText(label.replace("\n", " "), Qt.ElideRight, 155)
        painter.setPen(QColor(234, 251, 255, 250))
        painter.setFont(QFont("Sans", 9, QFont.DemiBold))
        painter.drawText(QRectF(84, 8, 155, 72), Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap, label)
        painter.end()
        return pixmap

    def _run_floating_clipboard_native_drag(self, side, state_id):
        """Run an OS-native MIME drag so drops work with folders and accepting app targets."""
        state = self.floating_clipboard_drag_state.get(side)
        if not state or state.get("id") != state_id or not state.get("running"):
            return
        entry = next((item for item in self.floating_clipboard_entries
                      if item.get("id") == state.get("entry_id")), None)
        if entry is None:
            self.floating_clipboard_drag_state[side] = None
            return
        drag = None
        result = Qt.IgnoreAction
        target = None
        start_global = self._virtual_point_to_global(state.get("start", QPointF()))
        try:
            self._set_os_pointer(state.get("last_cursor", state.get("start")))
            if pyautogui is not None:
                pyautogui.mouseDown(button="left", _pause=False)
            drag = QDrag(self)
            drag.setMimeData(self._clipboard_mime_for_entry(entry, for_drag=True))
            drag.setPixmap(self._clipboard_drag_pixmap(entry))
            drag.setHotSpot(QPoint(16, 16))
            result = drag.exec(Qt.CopyAction | Qt.LinkAction, Qt.CopyAction)
            try:
                target = drag.target()
            except Exception:
                target = None
        except Exception as exc:
            self.status = f"Clipboard drag could not start: {exc}"
        finally:
            if pyautogui is not None:
                try:
                    pyautogui.mouseUp(button="left", _pause=False)
                except Exception:
                    pass
            drop_global = QCursor.pos()
            state_now = self.floating_clipboard_drag_state.get(side)
            if state_now and state_now.get("id") == state_id:
                self.floating_clipboard_drag_state[side] = None
            self.floating_clipboard_drag_running = False
            self.fist_mouse_down[side] = False
            self.pinch_drag_active[side] = False
            self.active_action[side] = None
            self._reset_fist_gesture(side)
            self._clear_menu_pinch_sequence(side)
            if self.gesture_owner == side:
                self.gesture_owner = None
            self.action_stop_wait_open[side] = True
            self.action_stop_open_since[side] = None
            # Some native fields (for example a basic web text input) decline MIME drops.
            # If a real target was under the drop point, fall back to normal clipboard paste.
            if result == Qt.IgnoreAction and drop_global is not None:
                moved_far = (start_global is None or math.hypot(drop_global.x()-start_global.x(),
                                                                 drop_global.y()-start_global.y()) >= 24)
                if moved_far and (target is not None or state.get("moved", False)):
                    self._paste_floating_clipboard_entry_at(entry, drop_global)
            elif result != Qt.IgnoreAction:
                self.status = "Clipboard item dropped"
            self.update()
            overlay = getattr(self, "search_hand_overlay", None)
            if overlay is not None:
                overlay.update()

    def _paste_floating_clipboard_entry_at(self, entry, global_pos):
        """Focus the drop target and paste without blocking the GUI event loop.

        Move the OS cursor to the Qt global point, then click its *current* position.
        Passing Qt global coordinates into PyAutoGUI directly can click the wrong place
        on Windows when display scaling or multiple monitors are involved.
        """
        if not self._set_system_clipboard_for_drop(entry) or pyautogui is None or global_pos is None:
            return False
        try:
            self._set_os_pointer_global(global_pos)
            pyautogui.click(_pause=False)
            keys = ("command", "v") if sys.platform == "darwin" else ("ctrl", "v")
            # Allow the receiving window to process the click/focus change. Do not sleep
            # on the GUI thread: clipboard drops happen while the camera/hand UI is live.
            QTimer.singleShot(140, lambda paste_keys=keys: self._finish_floating_clipboard_paste(paste_keys))
            self.status = "Pasting clipboard item at the drop point"
            return True
        except Exception as exc:
            self.status = f"Clipboard paste fallback failed: {exc}"
            return False

    def _finish_floating_clipboard_paste(self, keys):
        """Send the final paste shortcut from a non-blocking Qt timer callback."""
        try:
            if pyautogui is not None:
                pyautogui.hotkey(*keys, interval=0.015, _pause=False)
        except Exception as exc:
            self.status = f"Clipboard paste fallback failed: {exc}"

    def _set_system_clipboard_from_entry_id(self, entry_id):
        entry = next((item for item in self.floating_clipboard_entries if item.get("id") == entry_id), None)
        if entry is None:
            return False
        if self._set_system_clipboard_from_entry(entry):
            label = "image" if entry.get("kind") == "image" else ("text" if entry.get("kind") == "text" else "copied item")
            self.status = f"Selected clipboard {label}; move to a destination and paste"
            return True
        return False

    def _action_menu_item_names(self):
        return ("right_click", "clipboard_toggle", "glass_menu", "dictation_toggle", "stt_toggle")

    def _action_menu_labels(self):
        search_dictation = bool(self.search_dialog is not None and self.search_dialog.isVisible()
                                and self.search_dialog.dictation_enabled)
        dictation_on = self.focus_dictation_enabled or search_dictation
        return {
            # These two top-row controls are icon-only by design.
            "right_click": "",
            "clipboard_toggle": "",
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
        half_w = (item_w-gap)/2
        items = {
            # Right-click and Clipboard share the top row; their width exactly matches
            # the Dictate and STT buttons, preserving the existing compact menu rhythm.
            "right_click": QRectF(item_x, top, half_w, item_h),
            "clipboard_toggle": QRectF(item_x+half_w+gap, top, half_w, item_h),
            "glass_menu": QRectF(item_x, top+item_h+gap, item_w, item_h),
            "dictation_toggle": QRectF(item_x, top+2*(item_h+gap), half_w, item_h),
            "stt_toggle": QRectF(item_x+half_w+gap, top+2*(item_h+gap), half_w, item_h),
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
            own_pid = os.getpid()
            rect = wintypes.RECT()
            class_buffer = ctypes.create_unicode_buffer(256)
            found = {"hwnd": None, "matches": []}
            enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            get_window_pid = user32.GetWindowThreadProcessId
            get_window_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            get_window_pid.restype = wintypes.DWORD

            @enum_type
            def enum_window(hwnd, _lparam):
                try:
                    handle = int(hwnd)
                except (TypeError, ValueError):
                    handle = int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0)
                # Windows has a separate always-on-top SearchHandOverlay HWND. If the
                # hit-test picks that transparent, full-screen window, two-hand drag
                # moves the hand layer instead of the application, making the hands
                # appear duplicated/ghosted as the overlay snaps back to owner.geometry.
                # Exclude every window owned by this process, not just the main HWND.
                if handle == overlay_hwnd:
                    return True
                process_id = wintypes.DWORD(0)
                thread_id = get_window_pid(hwnd, ctypes.byref(process_id))
                if not thread_id or int(process_id.value) == own_pid:
                    return True
                if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
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

    def _focus_native_window_for_zoom(self, hwnd):
        """Focus the chosen browser once so keyboard zoom reaches that window."""
        if not hwnd:
            return
        if sys.platform.startswith("linux"):
            try:
                subprocess.run(["wmctrl", "-ia", hex(int(hwnd))], check=False, timeout=0.3,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        elif sys.platform.startswith("win"):
            try:
                ctypes.windll.user32.SetForegroundWindow(wintypes.HWND(int(hwnd)))
            except Exception:
                pass

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

    def _resize_native_window(self, hwnd, left, top, width, height):
        """Resize a native top-level application window without taking keyboard focus."""
        if not hwnd:
            return False
        left, top = int(round(left)), int(round(top))
        width, height = max(260, int(round(width))), max(160, int(round(height)))
        if sys.platform.startswith("linux"):
            try:
                subprocess.run(["wmctrl", "-ir", hex(int(hwnd)), "-e",
                                f"0,{left},{top},{width},{height}"],
                               check=True, timeout=0.35, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except Exception as exc:
                self.status = f"Linux window resize failed (X11/wmctrl required): {exc}"
                return False
        if not sys.platform.startswith("win"):
            return False
        try:
            user32 = ctypes.windll.user32
            flags = 0x0004 | 0x0010 | 0x4000  # NOZORDER | NOACTIVATE | ASYNCWINDOWPOS
            return bool(user32.SetWindowPos(wintypes.HWND(int(hwnd)), None,
                                            left, top, width, height, flags))
        except Exception as exc:
            self.status = f"Window resize failed: {exc}"
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
            "scroll_open_candidate_since": None,
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
            self.status = "No movable window under the starting TPI"
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
        """Best-effort check for a native horizontal scroll range under the TPI."""
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
        tracked TPI and inject a wheel event. Detached hands keep their native pointer
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
        # Preserve the menu's actual position before closing it. The clipboard has a
        # different aspect ratio, so its centre is aligned to the menu's old centre.
        clipboard_menu_anchor = None
        if action == "clipboard_toggle" and not self.floating_clipboard_visible:
            clipboard_menu_anchor, _ = self._action_menu_geometry(side)
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
            self.status = "Right-click sent at the Pinch Menu selection point (original TPI)"
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
        elif action == "clipboard_toggle":
            self._set_floating_clipboard_visible(not self.floating_clipboard_visible)
            if self.floating_clipboard_visible and clipboard_menu_anchor is not None:
                self._position_floating_clipboard_over_rect(clipboard_menu_anchor)
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
        if name == "hand_display_fps":
            value = float(round(value))
        attributes = {
            "hand_scale": ("hand_visual_scale", 0.005),
            "hand_display_fps": ("hand_display_fps", 0.5),
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
            "virtual_grid_brightness": ("virtual_grid_brightness", 0.5),
            "virtual_grid_glow": ("virtual_grid_glow", 0.5),
            "virtual_grid_spacing": ("virtual_grid_spacing", 1.0),
            "virtual_grid_thickness": ("virtual_grid_thickness", 0.05),
        }
        if name == "tts_speed":
            if abs(value-float(self.tts.settings["speed"])) > 0.015:
                self.tts.update_settings(speed=value, persist=False)
                self.tts_settings_save_timer.start(450)
        else:
            attr, threshold = attributes[name]
            old_value = float(getattr(self, attr))
            if abs(value-old_value) >= threshold:
                setattr(self, attr, int(value) if name == "hand_display_fps" else value)
                if name == "default_gesture_hold_seconds":
                    self.menu_pinch_hold_seconds = value
                    self.peace_toggle_hold_seconds = value
                    self.gesture_tracking_toggle_hold_seconds = value
                if name == "ui_font_scale":
                    self.ui_preferences["ui_font_scale"] = value
                    save_ui_preferences(self.ui_preferences)
                    self.update()
                elif name.startswith("virtual_grid_"):
                    self._schedule_virtual_desktop_save()
                    self._glass_virtual_backdrop = None
                    self._glass_virtual_backdrop_size = None
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
        if self._garden_sundial_hit(point):
            self._garden_toggle_sundial(now)
            return
        self.click_side = side
        play_ui_sound(self.ui_preferences.get("sound_pinch", "None"))
        if self._canvas_menu_panel_contains(point):
            play_ui_sound(self.ui_preferences.get("sound_glass_buttons", "None"))
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
                if self._main_menu_exit_rect(cx, cy, scale).contains(point):
                    self.status = "Exiting Freehand: Spatial Computing"
                    self.close()
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
            if self.settings_page == "sounds":
                popup = self._sound_dropdown_layout(cx, cy, scale)
                for value, rect in popup.get("items", []):
                    if rect.contains(point):
                        key = popup["key"]
                        self.sound_dropdown_key = key
                        self.ui_preferences[key] = value
                        save_ui_preferences(self.ui_preferences)
                        self.status = f"{key.replace('sound_', '').replace('_', ' ').title()} sound: {value}"
                        self.update()
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
                    self.status = "Left TPI colour changed"
                elif pressed("swatch_right"):
                    self._cycle_hand_colour("right")
                    self.status = "Right TPI colour changed"
                elif pressed("glow_toggle"):
                    self.tli_glow_enabled = not self.tli_glow_enabled
                    self.schedule_gesture_settings_save()
                    self.status = "TPI glow enabled" if self.tli_glow_enabled else "TPI glow disabled"
                return

            if self.settings_page == "gestures":
                self._click_custom_gestures_page(point, cx, cy, scale, now)
                return

            if self.settings_page == "virtual":
                if pressed("virtual_scene_switch"):
                    style = next((name for name, seg in self._virtual_scene_segments(
                        controls["virtual_scene_switch"], scale) if seg.contains(point)), None)
                    if style is None:  # Caption cycles through the scenes, like Hands / Render.
                        style = VIRTUAL_SCENES[(VIRTUAL_SCENES.index(self.virtual_scene)+1) % len(VIRTUAL_SCENES)]
                    self._set_virtual_scene(style)
                elif pressed("virtual_grid_toggle"):
                    self.virtual_grid_enabled = not self.virtual_grid_enabled
                    self._glass_virtual_backdrop = None
                    self._glass_virtual_backdrop_size = None
                    self._schedule_virtual_desktop_save()
                    self.status = "Virtual grid enabled" if self.virtual_grid_enabled else "Virtual grid hidden"
                    self.update()
                elif pressed("virtual_background_style"):
                    self.virtual_background_style = (self.virtual_background_style + 1) % len(VIRTUAL_BACKGROUND_STYLES)
                    self._glass_virtual_backdrop = None
                    self._glass_virtual_backdrop_size = None
                    self._schedule_virtual_desktop_save()
                    self.status = "Virtual background: " + self._virtual_background_style_data()["name"].title()
                    self.update()
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
                if pressed("disabled_indicator_toggle"):
                    self.show_gesture_disabled_indicator = not self.show_gesture_disabled_indicator
                    self.ui_preferences["show_gesture_disabled_indicator"] = self.show_gesture_disabled_indicator
                    save_ui_preferences(self.ui_preferences)
                    self.status = ("Disabled-gesture indicator shown" if self.show_gesture_disabled_indicator
                                   else "Disabled-gesture indicator hidden")
                    self.update()
                    return
                for calibration_side, calibration_rect in self._tli_calibration_button_rects(cx, cy, scale).items():
                    if calibration_rect.contains(point):
                        self._arm_tli_calibration(calibration_side)
                        return
                if pressed("hand_setup"):
                    self._start_hand_onboarding(first_run=False)
                    return
                return

            if self.settings_page == "sounds":
                for key, _label in SOUND_SETTING_ROWS:
                    if pressed("sound:" + key):
                        self.sound_dropdown_key = key
                        self.update()
                        break
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

        # Virtual desktop app controls remain interactive without forwarding clicks to OS apps.
        if self.desktop_mode == "virtual" and self._handle_virtual_dock_click(point):
            return

        # Detached hands can still operate Spatial Desktop's own glass controls, but
        # an outside-menu pinch must never move/click the native OS pointer.
        if self.mouse_detached_by_side.get(side, False):
            return
        if self.desktop_mode == "os" and pyautogui is not None:
            try:
                global_pos = self._virtual_point_to_global(point)
                self._set_os_pointer_global(global_pos)
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
        """Send a right-click at the captured TPI point (used by Right-click Mode)."""
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
                self._set_os_pointer_global(global_pos)
                pyautogui.click(button="right", _pause=False)
                self.right_click_context_active[side] = True
            except Exception as exc:
                self.status = f"OS right-click failed: {exc}"

    def _linux_x11_screen_context(self):
        """Return cached native X11 display/root metrics when running on XCB."""
        if not sys.platform.startswith("linux"):
            return None
        try:
            if QApplication.platformName().lower() not in ("xcb", "x11"):
                return None
        except Exception:
            return None
        cached = getattr(self, "_linux_x11_screen_context_cache", None)
        if cached is not None:
            return cached
        try:
            x11 = ctypes.CDLL("libX11.so.6")
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XOpenDisplay.restype = ctypes.c_void_p
            display = x11.XOpenDisplay(None)
            if not display:
                return None
            x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
            x11.XDefaultScreen.restype = ctypes.c_int
            x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            x11.XDefaultRootWindow.restype = ctypes.c_ulong
            x11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
            x11.XDisplayWidth.restype = ctypes.c_int
            x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
            x11.XDisplayHeight.restype = ctypes.c_int
            x11.XWarpPointer.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_int, ctypes.c_int,
            ]
            x11.XWarpPointer.restype = ctypes.c_int
            x11.XFlush.argtypes = [ctypes.c_void_p]
            x11.XFlush.restype = ctypes.c_int
            screen_number = int(x11.XDefaultScreen(display))
            x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x11.XRaiseWindow.restype = ctypes.c_int
            context = {
                "lib": x11,
                "display": display,
                "root": int(x11.XDefaultRootWindow(display)),
                "width": int(x11.XDisplayWidth(display, screen_number)),
                "height": int(x11.XDisplayHeight(display, screen_number)),
                "screen_number": screen_number,
                "created_at": time.monotonic(),
            }
            self._linux_x11_lib = x11
            self._linux_x11_display = display
            self._linux_x11_screen_context_cache = context
            return context
        except Exception:
            return None

    def _interaction_screen_geometry(self):
        """Use the X11 root-screen rectangle when Qt reports only the work area.

        Qtile can expose a QScreen height that stops above a bottom panel/strut. The X11
        root size is independent of EWMH work areas, so use it only when width matches
        closely and the missing height is small enough to be a reserved bar—not a
        high-DPI scale or multi-monitor geometry mismatch.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, max(1, self.width()), max(1, self.height()))
        qt_rect = screen.geometry()
        if not sys.platform.startswith("linux"):
            return qt_rect
        context = self._linux_x11_screen_context()
        if context is None:
            self._linux_native_pointer_warp_compatible = False
            return qt_rect
        try:
            context["width"] = int(context["lib"].XDisplayWidth(
                context["display"], int(context["screen_number"])))
            context["height"] = int(context["lib"].XDisplayHeight(
                context["display"], int(context["screen_number"])))
        except Exception:
            pass
        qt_bounds = (qt_rect.x(), qt_rect.y(), qt_rect.width(), qt_rect.height())
        resolved, native_compatible = resolve_fullscreen_bounds(
            qt_bounds, (context["width"], context["height"]))
        self._linux_native_pointer_warp_compatible = bool(native_compatible)
        return QRect(*resolved)

    def _virtual_point_to_global(self, point):
        """Convert a screen-local virtual-hand point into desktop-global coordinates."""
        if point is None:
            return None
        pixel = QPoint(round(point.x()), round(point.y()))
        if sys.platform.startswith("linux"):
            geometry = self._interaction_screen_geometry()
            return geometry.topLeft() + pixel
        return self.mapToGlobal(pixel)

    def _set_os_pointer_global(self, global_pos):
        """Warp through X11 root coordinates when Qtile's work area is too short.

        QCursor.setPos is the normal cross-platform route. On an XCB session whose Qt
        and X11 screen widths match, warping directly to the root avoids any Qt widget/
        work-area confinement and can reach the lower Qtile bar as well as the top bar.
        """
        if global_pos is None:
            return False
        if sys.platform.startswith("linux") and getattr(self, "_linux_native_pointer_warp_compatible", False):
            context = self._linux_x11_screen_context()
            if context is not None:
                try:
                    x = max(0, min(context["width"] - 1, int(global_pos.x())))
                    y = max(0, min(context["height"] - 1, int(global_pos.y())))
                    context["lib"].XWarpPointer(
                        context["display"], ctypes.c_ulong(0), ctypes.c_ulong(context["root"]),
                        0, 0, 0, 0, x, y)
                    context["lib"].XFlush(context["display"])
                    self.last_os_cursor_pos = QPoint(x, y)
                    return True
                except Exception:
                    pass
        QCursor.setPos(global_pos)
        self.last_os_cursor_pos = QPoint(global_pos)
        return True

    def _set_os_pointer(self, point):
        clipboard_interaction_active = any(
            state is not None for state in getattr(self, "floating_clipboard_drag_state", {}).values()
        )
        # The floating clipboard is a glass UI, but once a hand has picked an entry up,
        # it must be allowed to move the native pointer out of the glass panel and into
        # another application even if the main Glass Menu remains visible behind it.
        if point is None or (self._gesture_ui_modal_open() and not clipboard_interaction_active):
            return
        global_pos = self._virtual_point_to_global(point)
        if global_pos is None:
            return
        self._set_os_pointer_global(global_pos)

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
        """Stop every interaction for a disabled hand but keep its last TPI position frozen."""
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
        self._stop_timer_during_sound(f"fist:{side}")
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
                        "enable freehand", "disable freehand", "enable free hand", "disable free hand",
                        "enable freehand gestures", "disable freehand gestures", "enable free hand gestures", "disable free hand gestures",
                        "turn on freehand", "turn off freehand", "turn on free hand", "turn off free hand",
                        "turn freehand on", "turn freehand off", "turn free hand on", "turn free hand off",
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

    def set_freehand_gestures_enabled(self, enabled: bool, announce: bool = True):
        """Enable or disable both hands' gesture input while keeping camera tracking alive.

        This is the voice-command equivalent of the built-in thumbs-up toggle. Keeping
        inference alive is important: either hand can still make a thumbs-up to re-enable
        gesture input after it has been disabled.
        """
        enabled = bool(enabled)
        sides = ("left", "right")
        changed = any(self.gesture_tracking_enabled.get(side, True) != enabled for side in sides)
        if not changed:
            state = "on" if enabled else "off"
            self.status = f"Freehand gestures are already {'enabled' if enabled else 'disabled'}"
            if announce:
                self.tts.speak(f"Left hand gestures {state}. Right hand gestures {state}.")
            self.update()
            return

        now = time.monotonic()
        for side in sides:
            # Detect whether this command is being spoken during an already-held
            # thumbs-up. Latch only that live pose; don't force users to make and release
            # an extra thumbs-up if they start one immediately after the voice command.
            last_thumb = self.thumbs_up_last_seen.get(side)
            thumb_pose_recent = last_thumb is not None and now-last_thumb <= 0.18
            self.gesture_tracking_enabled[side] = enabled
            # Clear any in-flight drag, click, scroll or menu interaction before changing
            # state. Requiring an open hand after a change avoids accidentally reusing a
            # held pose as a fresh action. Latch an already-held thumbs-up until release so
            # the voice command cannot immediately toggle the same pose back.
            self._cancel_side_interactions_for_tracking_pause(side)
            self.action_stop_wait_open[side] = True
            self.action_stop_open_since[side] = None
            self.thumbs_up_toggle_latched[side] = bool(thumb_pose_recent)
            if not thumb_pose_recent:
                self.thumbs_up_hold_started[side] = None
                self.thumbs_up_last_seen[side] = None
            self.side_cursor_positions[side] = None
            self.smoothed_side_cursor_positions[side] = None
            self.cursor_updated_at[side] = 0.0
            self.tli_overrides.pop(side, None)

        if not enabled:
            if self.voice_selection_mode is not None:
                self._clear_voice_selection(release_mouse=True)
            self.active_slider = None
            self.active_slider_side = None
            self.tli_overrides = {}
            self.active_push_sides = set()
        self.gesture_owner = None
        self.pointer_side = None
        self.canvas_active_side = None
        self._refresh_hand_tracking_notice(now, show_enabled=enabled)
        if not enabled:
            self.hand_tracking_notice_text = "Freehand gestures disabled"
            self.hand_tracking_notice_until = now + 5.0
        self.status = "Freehand gestures enabled" if enabled else "Freehand gestures disabled"
        if announce:
            state = "on" if enabled else "off"
            self.tts.speak(f"Left hand gestures {state}. Right hand gestures {state}.")
        self.update()

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
        """Return the active TPI's global screen position, falling back to the OS cursor."""
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
                # Restore only a minimised target. SW_RESTORE also unmaximises a
                # maximised Explorer window, which unexpectedly shrinks it for every
                # voice navigation/selection command.
                if user32.IsIconic(wintypes.HWND(target)):
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
            self.status = "Resume hand tracking before starting a TPI text selection."
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
            QTimer.singleShot(120, self._capture_system_clipboard)
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
            QTimer.singleShot(180, self._capture_system_clipboard)
            self.status = "Copy sent to the selected text or item. Say 'paste here' at the destination."
            self.tts.speak("Copied. Move to the destination and say paste here.")
            return True
        return False

    def _paste_at_tli(self):
        point = self._current_tli_global_position()
        if self.desktop_mode != "os":
            self.status = "Switch to OS Desktop mode to paste at the TPI position."
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
            self.status = "Pasted clipboard contents at the TPI position."
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

    def _run_windows_explorer_navigation(self, command):
        """Use Shell automation to select/open items in the remembered Explorer view."""
        self._remember_external_foreground_window(force=True)
        hwnd = int(getattr(self, "last_external_foreground_hwnd", 0) or 0)
        if not hwnd or not self._focus_external_target(close_search=True):
            return False, "Open File Explorer to the folder you want to control, then repeat the command."
        hwnd = int(getattr(self, "last_external_foreground_hwnd", 0) or hwnd)
        payload = dict(command)
        payload["hwnd"] = hwnd
        payload_data = base64.b64encode(json.dumps(payload, ensure_ascii=True).encode("utf-8")).decode("ascii")
        script = r"""$ErrorActionPreference = 'Stop'
try {
  $payload = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('__PAYLOAD__')) | ConvertFrom-Json
  $shell = New-Object -ComObject Shell.Application
  $view = $null
  foreach ($candidate in $shell.Windows()) {
    try {
      if ([Int64]$candidate.HWND -eq [Int64]$payload.hwnd -and $candidate.Document -and $candidate.Document.Folder) {
        $view = $candidate
        break
      }
    } catch { }
  }
  if ($null -eq $view) { throw 'The active external window is not a File Explorer folder. Open the target folder in File Explorer and try again.' }
  $document = $view.Document
  $items = @($document.Folder.Items())

  function Test-ItemKind($item, $kind) {
    if (-not $kind -or $kind -eq 'all') { return $true }
    $isFolder = [bool]$item.IsFolder
    if ($kind -eq 'folder') { return $isFolder }
    if ($kind -eq 'file') { return (-not $isFolder) }
    return $true
  }
  function Normalize-SpokenItemName($value) {
    # People naturally omit punctuation while speaking names like llama.cpp,
    # my-folder, or my_folder. Strip separators only for the fallback match.
    return [regex]::Replace(([string]$value).ToLowerInvariant(), '[^\p{L}\p{N}]', '')
  }
  function Find-SpokenItem($name, $kind) {
    $foundItems = @($items | Where-Object { ([string]$_.Name).Equals([string]$name, [System.StringComparison]::OrdinalIgnoreCase) -and (Test-ItemKind $_ $kind) })
    if ($foundItems.Count -eq 0 -and -not ([string]$name).Contains('.')) {
      $foundItems = @($items | Where-Object {
        ([System.IO.Path]::GetFileNameWithoutExtension([string]$_.Name)).Equals([string]$name, [System.StringComparison]::OrdinalIgnoreCase) -and (Test-ItemKind $_ $kind)
      })
    }
    if ($foundItems.Count -eq 0) {
      $spokenKey = Normalize-SpokenItemName $name
      if ($spokenKey.Length -gt 0) {
        $foundItems = @($items | Where-Object {
          (Normalize-SpokenItemName $_.Name) -eq $spokenKey -and (Test-ItemKind $_ $kind)
        })
      }
    }
    if ($foundItems.Count -eq 0) { throw ("No matching item named '{0}' was found in the current folder." -f $name) }
    if ($foundItems.Count -gt 1) { throw ("More than one item matches '{0}'. Say its full filename or specify file/folder." -f $name) }
    return $foundItems[0]
  }

  switch ([string]$payload.action) {
    'select_all_kind' {
      $selected = @($items | Where-Object { Test-ItemKind $_ ([string]$payload.kind) })
      if ($selected.Count -eq 0) { throw ("There are no {0} to select in this folder." -f $payload.kind) }
      $first = $true
      foreach ($item in $selected) {
        $flags = 1
        if ($first) { $flags = 29 }
        $document.SelectItem($item, $flags)
        $first = $false
      }
      [Console]::Out.WriteLine(('Selected {0} {1}.' -f $selected.Count, $payload.kind))
    }
    'select_items' {
      $selected = @()
      foreach ($name in $payload.names) { $selected += Find-SpokenItem ([string]$name) ([string]$payload.kind) }
      $first = $true
      foreach ($item in $selected) {
        $flags = 1
        if ($first) { $flags = 29 }
        $document.SelectItem($item, $flags)
        $first = $false
      }
      [Console]::Out.WriteLine(('Selected {0} item(s).' -f $selected.Count))
    }
    'open_item' {
      $item = Find-SpokenItem ([string]$payload.name) ([string]$payload.kind)
      if ([bool]$item.IsFolder) {
        try { $view.Navigate2([string]$item.Path) } catch { $view.Navigate([string]$item.Path) }
      } else {
        Start-Process -FilePath ([string]$item.Path)
      }
      [Console]::Out.WriteLine(('Opened {0}.' -f $item.Name))
    }
    'navigate_item' {
      $item = Find-SpokenItem ([string]$payload.name) 'folder'
      if (-not [bool]$item.IsFolder) { throw ("'{0}' is not a folder." -f $item.Name) }
      try { $view.Navigate2([string]$item.Path) } catch { $view.Navigate([string]$item.Path) }
      [Console]::Out.WriteLine(('Navigating to {0}.' -f $item.Name))
    }
    'create_item' {
      $directory = [string]$document.Folder.Self.Path
      if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw 'The active location is not a file-system folder.' }
      $kind = [string]$payload.kind
      if ($kind -eq 'folder') {
        $base = 'New folder'
        $target = Join-Path $directory $base
        $index = 2
        while (Test-Path -LiteralPath $target) { $target = Join-Path $directory (('{0} ({1})' -f $base, $index)); $index++ }
        New-Item -ItemType Directory -Path $target -ErrorAction Stop | Out-Null
        [Console]::Out.WriteLine(('Created folder {0}.' -f [System.IO.Path]::GetFileName($target)))
      } elseif ($kind -eq 'text') {
        $base = 'New Text Document'
        $target = Join-Path $directory ($base + '.txt')
        $index = 2
        while (Test-Path -LiteralPath $target) { $target = Join-Path $directory (('{0} ({1}).txt' -f $base, $index)); $index++ }
        [System.IO.File]::WriteAllText($target, '')
        [Console]::Out.WriteLine(('Created text file {0}.' -f [System.IO.Path]::GetFileName($target)))
        Start-Process -FilePath $target
      } elseif ($kind -eq 'powerpoint') {
        $base = 'Presentation'
        $target = Join-Path $directory ($base + '.pptx')
        $index = 2
        while (Test-Path -LiteralPath $target) { $target = Join-Path $directory (('{0} ({1}).pptx' -f $base, $index)); $index++ }
        $powerPoint = $null
        $presentation = $null
        try {
          $powerPoint = New-Object -ComObject PowerPoint.Application
          $presentation = $powerPoint.Presentations.Add()
          $presentation.SaveAs($target, 24)
          $presentation.Close()
          $powerPoint.Quit()
          [Console]::Out.WriteLine(('Created PowerPoint presentation {0}.' -f [System.IO.Path]::GetFileName($target)))
          Start-Process -FilePath $target
        } catch {
          try { if ($presentation) { $presentation.Close() } } catch { }
          try { if ($powerPoint) { $powerPoint.Quit() } } catch { }
          throw 'PowerPoint could not create a presentation. Check that Microsoft PowerPoint is installed.'
        }
      } else { throw 'Unsupported new-item type.' }
    }
    default { throw 'Unsupported file-navigation command.' }
  }
} catch {
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 1
}
""".replace("__PAYLOAD__", payload_data)
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                capture_output=True, text=True, timeout=7.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return False, "PowerShell was not found; File Explorer voice navigation is unavailable."
        except subprocess.TimeoutExpired:
            return False, "File Explorer did not respond to the voice command in time."
        if result.returncode == 0:
            message = (result.stdout or "").strip().splitlines()
            return True, message[-1] if message else "File Explorer command completed."
        # PowerShell diagnostics can include source excerpts and invocation text.
        # Never send raw stderr to TTS: report a short, user-facing failure instead.
        return False, "I couldn't complete that file command. Check the item name and make sure File Explorer is open to the correct folder."

    def execute_navigation_voice_command(self, command):
        """Execute a parsed, explicit file/folder navigation command."""
        action = command.get("action")
        if action == "create_item" and not sys.platform.startswith("win"):
            if command.get("kind") == "folder" and self._voice_hotkey("ctrl", "shift", "n"):
                self.status = "Creating a new folder in the active file manager."
                self.tts.speak(self.status)
                return True
            self.status = "Creating text files and PowerPoint presentations by voice is currently supported through Windows File Explorer."
            self.tts.speak(self.status)
            return False
        if action == "navigate_drive":
            if not sys.platform.startswith("win"):
                self.status = "Drive-letter navigation is available on Windows only."
                self.tts.speak(self.status)
                return False
            drive = str(command.get("drive", "C")).upper()
            drive_path = f"{drive}:\\"
            if not re.fullmatch(r"[A-Z]", drive) or not os.path.isdir(drive_path):
                self.status = f"The {drive}: drive is not available."
                self.tts.speak(self.status)
                return False
            try:
                subprocess.Popen(["explorer.exe", drive_path])
                self.status = f"Opening the {drive}: drive."
                self.tts.speak(self.status)
                return True
            except Exception as exc:
                self.status = f"Could not open the {drive}: drive: {exc}"
                return False

        if sys.platform.startswith("win"):
            success, message = self._run_windows_explorer_navigation(command)
        elif sys.platform == "darwin":
            success, message = self._run_macos_finder_navigation(command)
        else:
            success, message = self._run_linux_file_manager_navigation(command)
        self.status = message
        if message:
            self.tts.speak(message)
        return success

    def _run_macos_finder_navigation(self, command):
        """Use Finder AppleScript for item selection and opening."""
        action = command.get("action")
        try:
            if action == "select_all_kind":
                kind = command.get("kind")
                if kind == "folder":
                    expression = "every folder of front window"
                elif kind == "file":
                    expression = "every file of front window"
                else:
                    expression = "every item of front window"
                script = f'tell application "Finder" to set selection to {expression}'
                subprocess.run(["osascript", "-e", script], check=True, timeout=4,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                return True, "Selected items in Finder."
            if action == "select_items":
                names = command.get("names", [])
            else:
                names = [command.get("name", "")]
            def apple_string(value):
                return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
            if action in {"select_items", "open_item", "navigate_item"}:
                if action == "select_items":
                    exprs = [f'(every item of front window whose name is {apple_string(n)})' for n in names]
                    script = 'tell application "Finder"\n set picked to {}\n' + ''.join(f' set picked to picked & {expr}\n' for expr in exprs) + ' set selection to picked\nend tell'
                else:
                    name = str(names[0])
                    expr = f'(first item of front window whose name is {apple_string(name)})'
                    script = f'tell application "Finder" to open {expr}'
                subprocess.run(["osascript", "-e", script], check=True, timeout=4,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                return True, "Finder command completed."
        except Exception:
            return False, "Finder could not find or select that item."
        return False, "That navigation command is not supported in Finder."

    def _run_linux_file_manager_navigation(self, command):
        """Use conservative X11 type-ahead actions on desktop file managers."""
        if not shutil.which("xdotool") or not self.last_external_window_id:
            return False, "Named file selection on Linux requires an active X11 file manager and xdotool."
        try:
            subprocess.run(["xdotool", "windowactivate", "--sync", str(self.last_external_window_id)],
                           check=True, timeout=0.75, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            action = command.get("action")
            if action == "select_all_kind":
                if command.get("kind") == "all":
                    subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+a"], check=True, timeout=1)
                    return True, "Selected all items in the active file manager."
                return False, "Filtering selection to files or folders needs file-manager-specific accessibility support on Linux."
            names = command.get("names", []) if action == "select_items" else [command.get("name", "")]
            if len(names) != 1:
                return False, "Selecting multiple named items is supported directly in Windows File Explorer; this Linux file manager needs an accessibility adapter."
            name = str(names[0])
            subprocess.run(["xdotool", "key", "--clearmodifiers", "Escape"], timeout=0.5)
            subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "15", name], check=True, timeout=2)
            if action in {"open_item", "navigate_item"}:
                subprocess.run(["xdotool", "key", "Return"], check=True, timeout=1)
                return True, f"Opening {name}."
            if action == "select_items":
                return True, f"Requested selection of {name} in the active file manager."
        except Exception:
            return False, "The active Linux file manager did not accept the voice command."
        return False, "That file-manager command is not supported on Linux."

    def execute_pc_voice_command(self, action):
        """Run a matched deterministic desktop command; no shell text is interpreted."""
        self._clear_heard_popup()
        if action == "show_clipboard":
            self._set_floating_clipboard_visible(True)
            return True
        if action == "hide_clipboard":
            self._set_floating_clipboard_visible(False)
            return True
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
        if action in ("create_new_text_file", "create_new_powerpoint"):
            kind = "text" if action == "create_new_text_file" else "powerpoint"
            self.execute_navigation_voice_command({"action": "create_item", "kind": kind})
            return True
        if action == "new_folder":
            if sys.platform.startswith("win"):
                self.execute_navigation_voice_command({"action": "create_item", "kind": "folder"})
            elif not self._voice_hotkey("ctrl", "shift", "n"):
                self.status = "Folder creation shortcut was not accepted by the active file manager."
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
        if action == "open_my_computer":
            try:
                if sys.platform.startswith("win"):
                    subprocess.Popen(["explorer.exe", "shell:MyComputerFolder"])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "/"])
                else:
                    subprocess.Popen(["xdg-open", "/"])
                self.status = "Opening this computer's drives and locations."
                self.tts.speak("Opening my computer.")
            except Exception as exc:
                self.status = f"Could not open the computer view: {exc}"
            return True
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
            "new_tab": ("ctrl", "t"), "close_tab": ("ctrl", "w"),
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
        # Global Freehand gesture commands run before dictation and keep camera inference
        # alive, matching the thumbs-up gesture toggle rather than pausing the camera.
        enable_freehand_phrases = {
            "enable freehand", "enable free hand", "enable freehand gestures", "enable free hand gestures",
            "turn on freehand", "turn on free hand", "turn freehand on", "turn free hand on",
            "turn on freehand gestures", "enable gestures",
        }
        disable_freehand_phrases = {
            "disable freehand", "disable free hand", "disable freehand gestures", "disable free hand gestures",
            "turn off freehand", "turn off free hand", "turn freehand off", "turn free hand off",
            "turn off freehand gestures", "disable gestures",
        }
        if low in enable_freehand_phrases:
            self._clear_heard_popup()
            self.set_freehand_gestures_enabled(True)
            return
        if low in disable_freehand_phrases:
            self._clear_heard_popup()
            self.set_freehand_gestures_enabled(False)
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
        navigation_command = parse_navigation_command(text)
        if navigation_command:
            self.execute_navigation_voice_command(navigation_command)
            return
        # A phrase that clearly asks to select/open a file, folder, or drive must
        # never be sent to the generative assistant. That fallback can speak model
        # output verbatim (including code) when speech recognition mangles a name.
        if is_navigation_intent(text):
            self._clear_heard_popup()
            self.status = "I couldn't identify that file or folder. Try ‘select the llama.cpp folder’ or ‘open the Agents folder’."
            return
        # d1 interprets less exact natural phrasing; its output is restricted to
        # supported action IDs and falls through on uncertainty.
        try:
            pc_action = classify_voice_command(text, aliases_for_help())
        except Exception:
            pc_action = None
        if pc_action in {"new_folder", "create_new_text_file", "create_new_powerpoint"} and not has_explicit_create_prefix(text):
            # Do not let a fuzzy classifier turn a selected item named "new folder 2"
            # (or a truncated STT phrase "new folder") into a file-creation action.
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
            self.status = "Voice commands: enable/disable Freehand, pause/resume tracking, search bar, copy this/copy text, paste here/paste this, show running apps, start text selection, copy image, media playback/volume, window management, file navigation, close window, open folders, stop listening"
            self.tts.speak("You can say enable Freehand, disable Freehand, pause tracking, resume tracking, begin dictation, end dictation, show menu, show context, search bar, start text selection, copy this, copy text, copy image, paste here, paste this, show running apps, switch window, play pause, next track, previous track, volume up, volume down, mute, unmute, minimize window, maximize window, show desktop, open downloads, create new folder, create new text file, create new powerpoint, select all, undo, redo, close window, or open user folder.")
        else:
            # If speech recognition drops the verb and only hears an unprefixed
            # creation phrase, never infer a destructive/creating action from it.
            unprefixed_create = re.fullmatch(r"new (?:folder(?: \d+)?|text file|power ?point(?: presentation)?)", low)
            if unprefixed_create:
                self.status = "To create an item, start the command with 'create', for example 'create new folder'."
                return
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
            self.tpi_calibration_draft_points[tracked_side] = []
        self.tli_calibration_active_side = side
        self.tli_calibration_pending[side] = True
        self.tli_calibration_wait_release[side] = True
        self.tli_calibration_samples[side] = []
        self.tpi_calibration_draft_points[side] = []
        self.smoothed_side_cursor_positions[side] = None
        self.cursor_updated_at[side] = 0.0
        self.status = (f"{side.title()} TPI calibration — release the selecting pinch, then "
                       "move the TPI to point 1 of 9 and pinch-hold")

        window = TliCalibrationWindow(side)
        window.set_full_screen_geometry(self._interaction_screen_geometry())
        window.cancel_requested.connect(self._cancel_tli_calibration)
        self.tli_calibration_window = window
        window.set_target_index(0)
        window.set_phase("Release the selecting pinch, then pinch-hold on target 1/9")
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
            self.tpi_calibration_draft_points[side] = []
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
            self.status = "TPI calibration cancelled"
        self.update()

    def _calibration_pinch_inside_window(self, landmarks):
        """Test the exact scaled/calibrated TPI point against the outlined target area."""
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
        """Draw the currently transformed hand and TPI above the target window itself."""
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
        """Collect nine independent screen targets for one hand's TPI correction mesh."""
        active_side = self.tli_calibration_active_side
        if active_side is None:
            return False
        # Calibration is modal: even the other hand cannot move the OS pointer or click.
        if side != active_side:
            return True
        if not self.tli_calibration_pending.get(side, False):
            return True
        window = self.tli_calibration_window
        saved_points = self.tpi_calibration_draft_points[side]
        target_index = len(saved_points)
        if self.tli_calibration_wait_release.get(side, False):
            if not is_pinch:
                self.tli_calibration_wait_release[side] = False
                self.tli_calibration_samples[side] = []
                if window is not None:
                    window.set_phase(
                        f"Point {target_index+1}/9 · {TPI_CALIBRATION_TARGET_NAMES[target_index]} — move TPI to target and pinch-hold")
            elif window is not None:
                phase = ("Release the selecting pinch" if target_index == 0 else
                         f"Point {target_index}/9 saved — release, then move to the next target")
                window.set_phase(phase, 0)
            return True
        if not is_pinch:
            # Each target needs one continuous, deliberate hold; don't pool separate pulses.
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase(f"Point {target_index+1}/9 · pinch and hold on the target", 0)
            return True
        if not self._calibration_pinch_inside_window(landmarks):
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase(f"Point {target_index+1}/9 · move the TPI into the target ring", 0)
            return True
        # Store raw (pre-grid-correction) TLI pixels. The active hand bypasses its old
        # screen grid inside _scale_hand_tli_point(), making recalibration deterministic.
        raw_anchor = self.interactive_hand_anchor(landmarks, side)
        raw_screen = self._scale_hand_tli_point(
            raw_anchor,
            [(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in landmarks],
            side,
        )
        if raw_screen is None or not all(math.isfinite(v) for v in (raw_screen.x(), raw_screen.y())):
            self.tli_calibration_samples[side] = []
            if window is not None:
                window.set_phase("Unstable TPI position — hold steady", 0)
            return True
        samples = self.tli_calibration_samples[side]
        samples.append({"x": float(raw_screen.x()), "y": float(raw_screen.y())})
        if window is not None:
            window.set_phase(
                f"Point {target_index+1}/9 · sampling {len(samples)}/{self.tli_calibration_sample_count}",
                len(samples),
            )
        if len(samples) < self.tli_calibration_sample_count:
            return True

        observed = {
            "x": sum(float(row["x"]) for row in samples) / len(samples),
            "y": sum(float(row["y"]) for row in samples) / len(samples),
        }
        saved_points.append(observed)
        self.tli_calibration_samples[side] = []
        if len(saved_points) < TPI_CALIBRATION_TARGET_COUNT:
            next_index = len(saved_points)
            self.tli_calibration_wait_release[side] = True
            if window is not None:
                window.set_target_index(next_index)
                window.set_phase(
                    f"Point {len(saved_points)}/9 saved — release pinch, then move to {TPI_CALIBRATION_TARGET_NAMES[next_index]}",
                    self.tli_calibration_sample_count,
                )
            self.status = (f"{side.title()} TPI calibration: point {len(saved_points)}/9 saved; "
                           f"next {TPI_CALIBRATION_TARGET_NAMES[next_index].lower()}")
            return True

        screen = QApplication.primaryScreen()
        geometry = screen.geometry() if screen is not None else None
        width = geometry.width() if geometry is not None else self.width()
        height = geometry.height() if geometry is not None else self.height()
        calibration = make_tpi_screen_calibration(saved_points, width, height)
        if calibration is not None:
            prior = self.tpi_screen_calibrations.get(side)
            self.tpi_screen_calibrations[side] = calibration
            persisted = save_tpi_screen_calibrations(
                TPI_SCREEN_CALIBRATION_FILE, self.tpi_screen_calibrations)
            self.tli_calibration_persisted[side] = persisted or prior is not None
            if not persisted and prior is not None:
                # Preserve the last valid mesh if writing the new one failed.
                self.tpi_screen_calibrations[side] = prior
            status = (f"{side.title()} TPI nine-point calibration saved" if persisted else
                      f"{side.title()} TPI calibrated for this session; saving failed")
            if not persisted and prior is not None:
                status = f"{side.title()} TPI calibration save failed; previous calibration restored"
            self.status = status
            self.tli_calibration_pending[side] = False
            self.tli_calibration_wait_release[side] = False
            self.tpi_calibration_draft_points[side] = []
            self.tli_calibration_active_side = None
            self._cancel_tli_calibration(announce=False)
            self.status = status
        else:
            # Do not overwrite the last good calibration with a collapsed/invalid mesh.
            self.tpi_calibration_draft_points[side] = []
            self.tli_calibration_samples[side] = []
            self.tli_calibration_wait_release[side] = True
            if window is not None:
                window.set_target_index(0)
                window.set_phase("Calibration mesh was unstable; start again at point 1/9", 0)
            self.status = f"{side.title()} TPI calibration failed validation; please repeat"
        return True

    def raw_hand_anchor(self, landmarks, side=None):
        """Return the calibrated palm-relative TPI or the live fingertip midpoint fallback."""
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
        geometry = self._interaction_screen_geometry()
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
        screen_geometry = self._interaction_screen_geometry()
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
        """Map the calibrated palm-relative TPI before the common hand-scale transform."""
        return self.interactive_visual_point(*self.raw_hand_anchor(landmarks, side))

    # ------------------------------------------------------------------
    # Custom gestures (Settings -> Gestures)
    # ------------------------------------------------------------------
    CUSTOM_GESTURE_COUNTDOWN = 2.0
    CUSTOM_GESTURE_SAMPLES = 15
    CUSTOM_GESTURE_TIMEOUT = 7.0

    def _custom_gesture_layout(self, cx, cy, scale):
        """Single source of truth for Gestures-page geometry (drawing and hit-testing)."""
        # The editor controls above the list stay in the left settings column.
        # Gesture rows alone span both columns, using the full width beneath the
        # sliders so action labels have room and the list fills the panel.
        top = self._settings_layout("gestures")["top"]
        L, W = self.SETTINGS_LEFT_X, self.SETTINGS_COLUMN_W
        list_width = self.SETTINGS_RIGHT_X + self.SETTINGS_COLUMN_W - L
        # Leave clear visual gutters between action, shortcut capture, and delete.
        delete_x = L + list_width - 32.0
        capture_x = delete_x - 48.0 - 8.0
        action_x = L + 150.0
        action_width = capture_x - action_x - 8.0

        def r(x, y, w, h):
            return QRectF(cx+x*scale, cy+(top+y)*scale, w*scale, h*scale)
        rows = []
        for index in range(CUSTOM_GESTURE_MAX):
            y = 156 + index*40
            rows.append({
                "row": r(L, y, list_width, 38),
                "side": r(L+2, y+3, 40, 32),
                "timer": r(L+46, y+3, 42, 32),
                "label": r(L+94, y+3, 48, 32),
                "action": r(action_x, y+3, action_width, 32),
                "capture": r(capture_x, y+3, 48, 32),
                "delete": r(delete_x, y+3, 32, 32),
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
                self._start_custom_gesture_key_capture(index)
                return
            if pyautogui is None:
                self.custom_gesture_message = "Keyboard shortcut unavailable: pyautogui is not installed."
                return
            # Saved META is portable across Windows (Win), Linux (Super/Win), and
            # macOS (Command). Legacy WIN/SUPER/COMMAND names are translated too.
            keys = pyautogui_key_combo(combo)
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
        """Press a mouse button at the gesture's TPI (midway between TPIs for overlap).

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
        """Recognise recorded poses per hand and the two-TPI overlap; fire after a hold."""
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
                    self._maybe_play_timer_during_sound(
                        f"custom:{side}", trigger.since, now)
                    rings[side] = (self.custom_gestures[candidate].get("label", ""), progress,
                                   max(0.0, hold*(1.0-progress)))
            if fired is not None:
                self._run_custom_gesture_action(fired, side)

        # Built-in: overlapping both TPIs (open hands, nothing else in progress) and
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
            self.status = "TPI overlap → glass menu " + ("shown" if self.menu_open and self.hub_visible else "hidden")
        self.custom_gesture_live = live
        self.custom_gesture_rings = rings

    def _click_custom_gestures_page(self, point, cx, cy, scale, now):
        layout = self._custom_gesture_layout(cx, cy, scale)
        if self.custom_gesture_recording is not None:
            # Any press during recording cancels it, so a stray pinch can't save a bad pose.
            self.custom_gesture_recording = None
            self.custom_gesture_message = "Recording cancelled."
            return
        if self.custom_gesture_key_capture is not None:
            active_index = self.custom_gesture_key_capture
            active_key_rect = (layout["rows"][active_index]["capture"]
                               if 0 <= active_index < len(layout["rows"]) else None)
            if active_key_rect is None or not active_key_rect.contains(point):
                self._cancel_custom_gesture_key_capture("Shortcut capture cancelled. Click KEY to try again.")
                return
        if layout["record_left"].contains(point):
            self._start_custom_gesture_recording("left", now); return
        if layout["record_right"].contains(point):
            self._start_custom_gesture_recording("right", now); return
        for index, row in enumerate(layout["rows"][:len(self.custom_gestures)]):
            if not row["row"].contains(point):
                continue
            gesture = self.custom_gestures[index]
            if row["capture"].contains(point):
                # KEY is deliberately disabled for actions that don't invoke a key
                # combo. Users must choose CUSTOM KEY from the action selector first.
                if gesture.get("action") == "custom_key":
                    self._start_custom_gesture_key_capture(index)
                return
            if row["delete"].contains(point):
                if self.custom_gesture_key_capture is not None:
                    self._cancel_custom_gesture_key_capture()
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
                # The action selector always opens the action list. Shortcut capture is
                # assigned explicitly using the adjacent KEY button.
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
                    self.custom_gesture_key_capture = None
                    self.custom_gesture_message = (
                        "CUSTOM KEY selected. Click KEY on this row to capture a shortcut; "
                        "Windows Win+Tab is supported. Esc cancels capture.")
                else:
                    if self.custom_gesture_key_capture is not None:
                        self._cancel_custom_gesture_key_capture()
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
        close_hovered = any(point is not None and layout["close"].contains(point)
                            for point in self.side_cursor_positions.values())
        if close_hovered:
            for expansion, alpha, width in ((6*scale, 45, 3*scale), (2.5*scale, 90, 2*scale)):
                painter.setPen(QPen(QColor(255, 75, 85, alpha), width))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(layout["close"].adjusted(-expansion, -expansion, expansion, expansion),
                                        8*scale+expansion, 8*scale+expansion)
        painter.setBrush(QBrush(QColor(210, 42, 52, 250) if close_hovered else QColor(120, 30, 38, 230)))
        painter.setPen(QPen(QColor(255, 225, 225, 255) if close_hovered else QColor(255, 150, 150, 200),
                            1.8*scale if close_hovered else 1.0*scale))
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
        hover_points = [point for point in self.side_cursor_positions.values() if point is not None]

        def draw_control_glow(rect, rgb=(100, 255, 255)):
            """Consistent soft highlight for the smaller custom-gesture controls."""
            if not any(rect.contains(point) for point in hover_points):
                return False
            for expansion, alpha, width in ((6*scale, 35, 3*scale), (2.5*scale, 78, 2*scale)):
                painter.setPen(QPen(QColor(rgb[0], rgb[1], rgb[2], alpha), width))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect.adjusted(-expansion, -expansion, expansion, expansion),
                                        8*scale+expansion, 8*scale+expansion)
            return True

        painter.setPen(QColor(195, 235, 242))
        self._draw_fitted_button_label(painter, layout["intro"],
                                       "Use ▾ to choose an action; click KEY to capture a shortcut.\n"
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
                                       "BUILT-IN: OVERLAP BOTH TPIs AND HOLD → GLASS MENU",
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
            side_hovered = draw_control_glow(row["side"])
            painter.setBrush(QBrush(QColor(38, 135, 148, 245) if side_hovered else QColor(18, 80, 93, 235)))
            painter.setPen(QPen(QColor(205, 255, 255, 250) if side_hovered else QColor(125, 235, 245, 160),
                                1.8*scale if side_hovered else 1.0*scale))
            painter.drawRoundedRect(row["side"], 7*scale, 7*scale)
            painter.setPen(QColor(235, 255, 255))
            painter.setFont(font(7))
            painter.drawText(row["side"], Qt.AlignCenter, side_text)

            # Per-gesture circle timer toggle: ON = hold with a filling ring at the TLI,
            # OFF = fire after a brief stability check.
            timer_on = gesture.get("timer", True)
            timer_hovered = draw_control_glow(row["timer"])
            if timer_hovered:
                painter.setBrush(QBrush(QColor(68, 190, 112, 250) if timer_on else QColor(90, 133, 150, 235)))
                painter.setPen(QPen(QColor(220, 255, 225, 250), 1.8*scale))
            else:
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
                action_label = display_key_combo(gesture.get("key_combo")) or "SET KEY COMBO…"
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

            capturing = self.custom_gesture_key_capture == index
            capture_enabled = action == "custom_key"
            if capturing:
                # Active capture gets a persistent amber glow even when not hovered.
                for expansion, alpha, width in ((6*scale, 45, 3*scale), (2.5*scale, 95, 2*scale)):
                    painter.setPen(QPen(QColor(255, 210, 95, alpha), width))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(row["capture"].adjusted(-expansion, -expansion, expansion, expansion),
                                            8*scale+expansion, 8*scale+expansion)
            elif capture_enabled:
                draw_control_glow(row["capture"])
            # A KEY button is genuinely inert until CUSTOM KEY is chosen in the
            # action selector: no hover glow, no bright border, and muted label.
            if capturing:
                capture_fill, capture_edge, capture_text = (
                    QColor(145, 92, 24, 245), QColor(255, 224, 145, 245), QColor(255, 248, 222))
            elif capture_enabled:
                capture_fill, capture_edge, capture_text = (
                    QColor(24, 89, 92, 235), QColor(175, 250, 255, 190), QColor(225, 250, 255))
            else:
                capture_fill, capture_edge, capture_text = (
                    QColor(17, 35, 43, 155), QColor(86, 116, 125, 100), QColor(105, 137, 146))
            painter.setBrush(QBrush(capture_fill))
            painter.setPen(QPen(capture_edge, 1.0*scale))
            painter.drawRoundedRect(row["capture"], 7*scale, 7*scale)
            painter.setPen(capture_text)
            self._draw_fitted_button_label(painter, row["capture"], "PRESS" if capturing else "KEY",
                                           base_size=8, min_size=6, inset=2*scale)

            delete_hovered = draw_control_glow(row["delete"], (255, 75, 85))
            painter.setBrush(QBrush(QColor(210, 42, 52, 250) if delete_hovered else QColor(120, 30, 38, 230)))
            painter.setPen(QPen(QColor(255, 225, 225, 255) if delete_hovered else QColor(255, 150, 150, 200),
                                1.8*scale if delete_hovered else 1.0*scale))
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
        if name == "hand_display_fps":
            return f"{int(round(value))} FPS"
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
        if name in ("virtual_grid_brightness", "virtual_grid_glow"):
            return f"{value:.0f}%"
        if name == "virtual_grid_spacing":
            return f"{value:.0f}px"
        if name == "virtual_grid_thickness":
            return f"{value:.2f}px"
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
        # These landmarks are already in primary-screen coordinates. Use the same
        # physical screen dimensions for the contact correction; QWidget.height()
        # can reflect Qtile's reserved work area and truncate the lower screen band.
        contact_geometry = self._interaction_screen_geometry()
        contact_width = contact_geometry.width() if contact_geometry is not None else self.width()
        contact_height = contact_geometry.height() if contact_geometry is not None else self.height()
        px_x = max(1.0, contact_width-1)/0.60
        px_y = max(1.0, contact_height-1)/0.68
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
        """Return one consistent fixed-scale transform for landmarks and the TPI.

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
        geometry = self._interaction_screen_geometry()
        width = geometry.width() if geometry is not None else self.width()
        height = geometry.height() if geometry is not None else self.height()
        clamped = QPointF(min(max(0.0, x), max(1.0, width-1.0)),
                          min(max(0.0, y), max(1.0, height-1.0)))
        # While recalibrating this hand, deliberately expose its uncorrected TLI so the
        # new mesh is sampled against raw positions, not its previous correction grid.
        if (side in ("left", "right") and
                getattr(self, "tli_calibration_active_side", None) != side):
            calibration = getattr(self, "tpi_screen_calibrations", {}).get(side)
            corrected = map_tpi_screen_point(clamped, calibration, width, height)
            if corrected is not None:
                return QPointF(float(corrected[0]), float(corrected[1]))
        return clamped

    def _visual_tli_position(self, side, cursor, points=None):
        """Return the already-transformed TPI; only slider reticles override its position."""
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

    def _paint_peace_sign_timer(self, painter, side, now):
        """Show the SAME progress ring for disabling and re-enabling a hand.

        Disabled hands intentionally have no TLI/cursor. The separate display-only
        anchor captured from live hand landmarks remains available until the pose
        ends, so the countdown never depends on enabled cursor tracking.
        """
        started = self.peace_hold_started.get(side)
        if started is None or self.peace_toggle_latched.get(side, False):
            return
        points = self.hand_positions.get(side, [])
        if len(points) < 21:
            return  # No visible hand: never leave a phantom countdown on screen.
        center = self.peace_timer_anchor_by_side.get(side)
        if center is None:
            # Defensive fallback for the first frame or restored state; disabled
            # tracking must not prevent the countdown from being drawn.
            center = self._scale_hand_tli_point(
                self.interactive_hand_anchor(points, side), points, side)
        if center is None:
            return
        center = QPointF(center)
        elapsed = max(0.0, now-started)
        duration = max(0.01, self.peace_toggle_hold_seconds)
        progress = min(1.0, elapsed/duration)
        remaining = max(0.0, duration-elapsed)
        ring = QRectF(center.x()-32, center.y()-32, 64, 64)
        painter.save()
        try:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 220, 110, 100), 3))
            painter.drawEllipse(ring)
            painter.setPen(QPen(QColor(255, 220, 110, 248), 4))
            painter.drawArc(ring, 90*16, -int(progress*360*16))
            painter.setPen(QColor(255, 242, 205, 245))
            painter.setFont(self._ui_font(8, QFont.DemiBold))
            painter.drawText(QRectF(center.x()-22, center.y()+23, 44, 16),
                             Qt.AlignCenter, f"{remaining:.1f}s")
        finally:
            painter.restore()

    def _paint_gesture_disabled_indicator(self, painter, points, side):
        """Draw a red prohibition badge over a hand whose gesture input is disabled."""
        if (not getattr(self, "show_gesture_disabled_indicator", True) or
                side not in ("left", "right") or
                getattr(self, "gesture_tracking_enabled", {}).get(side, True) or
                not points or len(points) < 18):
            return

        source_points = points
        display = getattr(self, "hand_display_positions", {}).get(side)
        if (getattr(self, "hand_render_style", "flat") != "skeleton" and
                display is not None and display[0] is source_points):
            points = display[1]

        origin, visual_scale = self._hand_visual_transform(source_points, side)

        def mapped(index):
            q = self.interactive_visual_point(points[index][0], points[index][1])
            x, y = scale_hand_point((q.x(), q.y()),
                                    (origin.x(), origin.y()), visual_scale)
            return QPointF(x, y)

        # Centre the mark on the palm rather than on the TLI, which is deliberately
        # hidden while disabled. Use the same transformed landmarks as the hand mesh.
        palm_indices = (0, 5, 9, 13, 17)
        palm = [mapped(i) for i in palm_indices]
        center = QPointF(sum(pt.x() for pt in palm)/len(palm),
                         sum(pt.y() for pt in palm)/len(palm))
        palm_width = math.hypot(mapped(5).x()-mapped(17).x(),
                                mapped(5).y()-mapped(17).y())
        diameter = max(40.0, min(76.0, palm_width*1.55))
        radius = diameter*0.43

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        # Dark red edge keeps the symbol legible over every skin tone and background;
        # the brighter inner stroke gives it the same bold appearance as the reference.
        painter.setPen(QPen(QColor(95, 0, 8, 235), diameter*0.13,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawEllipse(center, radius, radius)
        painter.drawLine(QPointF(center.x()-radius*0.70, center.y()-radius*0.70),
                         QPointF(center.x()+radius*0.70, center.y()+radius*0.70))
        painter.setPen(QPen(QColor(255, 20, 34, 245), diameter*0.085,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawEllipse(center, radius, radius)
        painter.drawLine(QPointF(center.x()-radius*0.70, center.y()-radius*0.70),
                         QPointF(center.x()+radius*0.70, center.y()+radius*0.70))
        painter.restore()

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
        if display is not None and display[0] is points:
            if style == "skeleton":
                points = self.hand_stable_positions.get(side) or points
            else:
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
        the selected Hand Display FPS, repositioning the cached image between
        pose renders. Gesture input stays independent of this visual limit.
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
        render_interval = 1.0 / max(5, min(30, self.hand_display_fps))
        cache = self._hand_mesh_frame_cache.get(model_side)
        render_now = (cache is None or now - cache["rendered_at"] >= render_interval or
                      (cache is not None and colour != cache["colour"]))

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
        # its vertex pose. Finger articulation refreshes on the chosen FPS
        # cadence; translation stays visually responsive between mesh renders.
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
        """Paint a bright, large TPI; its centre is exactly the interaction point."""
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
        if visible:
            if timer_completed:
                play_ui_sound(self.ui_preferences.get("sound_gesture_timer", "None"))
            else:
                play_ui_sound(self.ui_preferences.get("sound_pinch_menu", "None"))
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

    def _paint_right_click_icon(self, painter, rect, highlighted=False):
        """Draw a clear mouse pictogram with its right button visibly pressed."""
        painter.save()
        cx, cy = rect.center().x(), rect.center().y()
        height = min(34.0, rect.height()*0.64)
        width = height*0.68
        mouse = QRectF(cx-width/2, cy-height/2, width, height)
        accent = QColor(150, 246, 255, 255) if highlighted else QColor(105, 220, 245, 245)
        painter.setPen(QPen(QColor(222, 251, 255, 245), 1.8))
        painter.setBrush(QBrush(QColor(14, 48, 65, 235)))
        painter.drawRoundedRect(mouse, width*0.48, width*0.48)
        # The highlighted top-right half is the pressed secondary button.
        pressed = QRectF(cx+0.7, mouse.top()+1.7, width/2-1.7, height*0.34)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(pressed, 3.0, 3.0)
        painter.setPen(QPen(QColor(220, 250, 255, 230), 1.4))
        painter.drawLine(QPointF(cx, mouse.top()+2), QPointF(cx, mouse.bottom()-4))
        painter.drawLine(QPointF(mouse.left()+2, mouse.top()+height*0.36),
                         QPointF(mouse.right()-2, mouse.top()+height*0.36))
        # Short press marks reinforce that the right button is being activated.
        painter.setPen(QPen(accent, 1.5, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(mouse.right()+3, cy-10), QPointF(mouse.right()+7, cy-13))
        painter.drawLine(QPointF(mouse.right()+4, cy-4), QPointF(mouse.right()+9, cy-5))
        painter.restore()

    def _paint_clipboard_icon(self, painter, rect, highlighted=False):
        """Draw a compact clipboard pictogram for the floating-history toggle."""
        painter.save()
        cx, cy = rect.center().x(), rect.center().y()
        height = min(34.0, rect.height()*0.64)
        width = height*0.72
        body = QRectF(cx-width/2, cy-height/2+2, width, height-2)
        accent = QColor(135, 249, 207, 255) if highlighted else QColor(112, 228, 242, 245)
        painter.setPen(QPen(QColor(224, 253, 255, 245), 1.8))
        painter.setBrush(QBrush(QColor(13, 48, 64, 235)))
        painter.drawRoundedRect(body, 4.0, 4.0)
        clip = QRectF(cx-width*0.24, body.top()-3, width*0.48, 7)
        painter.setPen(QPen(accent, 1.6))
        painter.setBrush(QBrush(QColor(20, 73, 88, 255)))
        painter.drawRoundedRect(clip, 2.5, 2.5)
        painter.setPen(QPen(accent, 1.8, Qt.SolidLine, Qt.RoundCap))
        for row in range(3):
            yy = body.top()+height*(0.37+row*0.19)
            painter.drawLine(QPointF(body.left()+5, yy), QPointF(body.right()-5, yy))
        painter.restore()

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
            if name == "right_click":
                self._paint_right_click_icon(painter, rect, hovered)
            elif name == "clipboard_toggle":
                self._paint_clipboard_icon(painter, rect, hovered or self.floating_clipboard_visible)
            else:
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
        """Smooth the already hand-scaled screen point used by the TPI and OS pointer."""
        previous = self.smoothed_side_cursor_positions.get(side)
        last_time = self.cursor_updated_at.get(side, 0.0)
        # Use the same full-screen coordinate system as interactive_visual_point().
        # Under Qtile the widget may temporarily report the work-area height (ending
        # above the bottom bar), which would clamp the cursor and hand-follow offset.
        screen_geometry = self._interaction_screen_geometry()
        h = max(1.0, float(screen_geometry.height() if screen_geometry is not None else self.height()))
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
                    if not self.gesture_tracking_enabled.get(side, True)]
        if disabled:
            label = " & ".join(disabled)
            self.hand_tracking_notice_text = f"{label} cursor + gestures detached"
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
            # A true tracking-toggle thumbs-up must point nearly vertically upward,
            # not diagonally out to the side. Use aspect-corrected camera pixels.
            thumb_vertical = thumb_axis_is_vertical_up(landmarks, max_angle_degrees=25.0)
            return thumb_up and fingers_folded and thumb_extended and thumb_vertical
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

    def _stop_timer_during_sound(self, key):
        """End one gesture's audio at the same moment its hold/countdown ends."""
        stop_event = self._timer_during_sound_stop_events.pop(key, None)
        if stop_event is not None:
            stop_event.set()
        self._timer_during_sound_played_at.pop(key, None)

    def _stop_inactive_timer_during_sounds(self):
        """End sounds whose timers vanished during this hand-tracking frame."""
        for key in list(self._timer_during_sound_stop_events):
            if key not in self._timer_during_sound_seen_keys:
                self._stop_timer_during_sound(key)
        self._timer_during_sound_seen_keys.clear()

    def _maybe_play_timer_during_sound(self, key, started_at, now):
        """Play after 200 ms; keep it cancellable while this timer is visible."""
        if started_at is None:
            self._stop_timer_during_sound(key)
            return
        self._timer_during_sound_seen_keys.add(key)
        if self._timer_during_sound_played_at.get(key) == started_at:
            return
        # A new hold must never inherit a sound from the previous hold.
        if key in self._timer_during_sound_stop_events:
            self._stop_timer_during_sound(key)
        if now - float(started_at) < 0.2:
            return
        self._timer_during_sound_played_at[key] = started_at
        stop_event = threading.Event()
        self._timer_during_sound_stop_events[key] = stop_event
        play_ui_sound(self.ui_preferences.get("sound_gesture_timer_during", "None"),
                      stop_event=stop_event)

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
        # Timer-during sound is delayed until the pinch hold survives 200 ms.
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
        self._stop_timer_during_sound(f"pinch_menu:{side}")
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
            self.status = "3D context menu opened at the TPI — hover an option and pinch or grip to select"
            self.update()
            return
        self._activate_canvas_right_click(point, "right", now)
        self.status = "Modifier right-click sent at the starting TPI"

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
            self.status = "Move the TPI onto a desktop window before using grip-drag"
            return False

        hwnd = self._find_native_window_at(origin)
        rect = self._native_window_rect(hwnd) if hwnd else None
        if not hwnd or rect is None:
            self.modifier_click_armed["right"] = False
            self.status = "No movable window under the right-hand TPI"
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
        if (getattr(self, "tli_calibration_window", None) is not None and
                self.tli_calibration_window.isVisible()):
            target = self.tli_calibration_window
        elif (getattr(self, "hand_calibration_window", None) is not None and
              self.hand_calibration_window.isVisible()):
            target = self.hand_calibration_window
        elif (search_overlay is not None and search_overlay.isVisible() and
              (search_dialog is not None and search_dialog.isVisible() or
               getattr(search_overlay, "_global_hand_layer_active", False))):
            # The independent hand-visual window is raised without activation while
            # ordinary apps or Windows 11 Task View switch the active desktop. The
            # Windows shell may still composite Task View in a higher z-order band.
            target = search_overlay
        elif search_dialog is not None and search_dialog.isVisible():
            # If hand overlay creation failed, don't raise the full-screen owner over Search.
            return
        else:
            target = self

        if not target.isVisible():
            return
        # Pin the main tracking overlay to every Windows virtual desktop. This is
        # best-effort: the shell COM API is undocumented and may be unavailable.
        if (not getattr(self, "_windows_virtual_desktop_pinned", False) and
                now-float(getattr(self, "_windows_virtual_desktop_pin_last_attempt", 0.0)) >= 2.0):
            self._windows_virtual_desktop_pin_last_attempt = now
            try:
                pinned = pin_window_on_all_virtual_desktops(int(self.winId()))
                if pinned:
                    self._windows_virtual_desktop_pinned = True
                elif not getattr(self, "_windows_virtual_desktop_pin_warning_reported", False):
                    print("Spatial Desktop: Windows could not pin the hand overlay to all virtual desktops; continuing normally.")
                    self._windows_virtual_desktop_pin_warning_reported = True
            except Exception as exc:
                if not getattr(self, "_windows_virtual_desktop_pin_warning_reported", False):
                    print("Spatial Desktop: virtual-desktop pinning unavailable:", exc)
                    self._windows_virtual_desktop_pin_warning_reported = True
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

        When Qt reports only the Qtile work area, _interaction_screen_geometry() uses
        matching native X11 root dimensions to recover the missing bottom strip. The
        fullscreen paint surface, hand coordinates and pointer warping then share that
        same corrected screen rectangle.
        """
        if not sys.platform.startswith("linux"):
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        screen_rect = self._interaction_screen_geometry()
        try:
            reported_origin = self.mapToGlobal(QPoint(0, 0))
        except Exception:
            reported_origin = screen_rect.topLeft()
        origin_matches = reported_origin == screen_rect.topLeft()
        size_matches = self.size() == screen_rect.size()
        geometry_matches = self.geometry() == screen_rect
        if not (origin_matches and size_matches and geometry_matches):
            # Qtile/XLibre can reapply the work-area bounds after startup. Restore the
            # QWidget geometry first, then also resize the unmanaged X11 window itself
            # to the full physical screen. This ensures its paint surface extends over
            # a bottom bar even if the WM leaves the native window at work-area height.
            self.setGeometry(screen_rect)
            try:
                context = self._linux_x11_screen_context()
                if context is not None and getattr(self, "_linux_native_pointer_warp_compatible", False):
                    x11 = context["lib"]
                    x11.XMoveResizeWindow.argtypes = [
                        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                        ctypes.c_uint, ctypes.c_uint,
                    ]
                    x11.XMoveResizeWindow.restype = ctypes.c_int
                    x11.XMoveResizeWindow(
                        context["display"], ctypes.c_ulong(int(self.winId())),
                        ctypes.c_int(screen_rect.x()), ctypes.c_int(screen_rect.y()),
                        ctypes.c_uint(screen_rect.width()), ctypes.c_uint(screen_rect.height()),
                    )
                    x11.XFlush(context["display"])
            except Exception:
                # QWidget geometry remains the portable fallback on unsupported sessions.
                pass
            self.update()

    # Repaint interval used while nothing on screen is moving (see _scene_is_idle).
    IDLE_REPAINT_INTERVAL = 0.33

    def _scene_is_idle(self, now):
        """True when no hand is in view and nothing is animating.

        Garden Mode is animated only when its virtual scene is actually visible;
        selecting Garden while in OS mode must not force nonstop repaints.
        """
        if (self.desktop_mode == "virtual" and
                (getattr(self, "virtual_garden_mode", False) or
                 getattr(self, "_scene_transition", None) is not None)):
            return False
        try:
            if getattr(self, "virtual_pen_active_stroke", None) is not None:
                stroke = self.virtual_pen_active_stroke
                if now-float(stroke.get("last_point_at", now)) > 1.2:
                    self._finish_virtual_pen_stroke()
                else:
                    return False
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
        """Keep gestures responsive while limiting full-screen painting/inference work."""
        garden_visible = (self.desktop_mode == "virtual" and
                          getattr(self, "virtual_garden_mode", False))
        if self.menu_open and self.hub_visible:
            # Pause expensive background animation when reading a glass settings page.
            desired_ms = 50 if hand_present else 100
        elif getattr(self, "_scene_transition", None) is not None:
            desired_ms = 33  # A short, smooth slide; static images keep this inexpensive.
        elif garden_visible:
            # The animated garden used to force 60 FPS even with no hands.
            # 30 FPS during interaction, 15 FPS while idle preserves gentle motion.
            desired_ms = 33 if hand_present else 66
        else:
            desired_ms = 50 if hand_present else 100
        timer = getattr(self, "timer", None)
        if timer is not None and timer.interval() != desired_ms:
            timer.setInterval(desired_ms)

    def _reset_middle_pinch_zoom(self):
        """Clear a pending or active two-hand pinch zoom session."""
        state = getattr(self, "middle_pinch_zoom", None)
        if state is None:
            return
        state.update({
            "started_at": None, "last_both_seen_at": None, "active": False,
            "mode": None, "hwnd": None, "rect": None, "corners": None,
            "start_points": None, "start_distance": None, "last_distance": None,
            "zoom_remainder": 0.0,
            "last_resize_rect": None, "last_resize_at": 0.0,
        })

    def _begin_middle_pinch_zoom(self, cursor_points, now):
        """Choose window resize from opposite corner zones; otherwise zoom content."""
        state = self.middle_pinch_zoom
        points = {side: self._virtual_point_to_global(cursor_points[side])
                  for side in ("left", "right")}
        if any(point is None for point in points.values()):
            return False
        point_tuples = {side: (float(point.x()), float(point.y())) for side, point in points.items()}
        midpoint = QPointF((cursor_points["left"].x()+cursor_points["right"].x())*0.5,
                           (cursor_points["left"].y()+cursor_points["right"].y())*0.5)
        initial_distance = math.hypot(
            point_tuples["right"][0]-point_tuples["left"][0],
            point_tuples["right"][1]-point_tuples["left"][1])
        left_hwnd = self._find_native_window_at(cursor_points["left"], prefer_smallest=True)
        right_hwnd = self._find_native_window_at(cursor_points["right"], prefer_smallest=True)
        centre_hwnd = self._find_native_window_at(midpoint, prefer_smallest=True)

        # Find the smallest native app window containing BOTH pinch starts. This lets
        # mode selection use where the pinches begin in that window, not whether the
        # app happens to be a browser, file manager, image viewer, or presentation app.
        candidates = []
        for candidate in (left_hwnd, right_hwnd, centre_hwnd):
            if candidate and all(int(candidate) != int(existing) for existing in candidates):
                candidates.append(candidate)
        shared_windows = []
        for candidate in candidates:
            rect = self._native_window_rect(candidate)
            if rect is None:
                continue
            left, top, right, bottom = rect
            if all(left <= x < right and top <= y < bottom
                   for x, y in point_tuples.values()):
                shared_windows.append(((right-left)*(bottom-top), candidate, rect))
        if shared_windows:
            _area, target, target_rect = min(shared_windows, key=lambda row: row[0])
            action, corners = classify_middle_pinch_window_action(
                target_rect, point_tuples["left"], point_tuples["right"],
                corner_zone_ratio=0.25)
        else:
            # If the hands straddle application windows, never resize either by
            # accident. Content zoom is the safe fallback for the active target.
            target = centre_hwnd or left_hwnd or right_hwnd
            target_rect = self._native_window_rect(target) if target else None
            action, corners = "zoom", None

        state["active"] = True
        state["start_points"] = point_tuples
        state["start_distance"] = max(1.0, initial_distance)
        state["last_distance"] = initial_distance
        state["zoom_remainder"] = 0.0
        state["last_resize_at"] = 0.0
        if action == "resize" and target and target_rect and corners:
            # On tiled Linux WMs, float/activate the app before resizing. Re-test the
            # same 25%-inset corner zones after the WM has finished preparing it.
            self._prepare_native_window_drag(target)
            target_rect = self._native_window_rect(target) or target_rect
            action, corners = classify_middle_pinch_window_action(
                target_rect, point_tuples["left"], point_tuples["right"],
                corner_zone_ratio=0.25)
            if action == "resize" and corners:
                state["mode"] = "resize"
                state["hwnd"] = target
                state["rect"] = target_rect
                state["corners"] = corners
                state["last_resize_rect"] = target_rect
                self.status = "Window resize active — move hands from opposite corner zones"
            else:
                # Window geometry changed during preparation; do not resize the wrong
                # edges. Continue in generic content zoom mode instead.
                state["mode"] = "zoom"
                state["hwnd"] = target
                state["rect"] = target_rect
                state["corners"] = None
                state["last_resize_rect"] = None
                self._focus_native_window_for_zoom(target)
                self.status = "Content zoom active — move hands apart to zoom in, together to zoom out"
        else:
            state["mode"] = "zoom"
            state["hwnd"] = target
            state["rect"] = target_rect
            state["corners"] = None
            state["last_resize_rect"] = None
            # Generic Ctrl/Command+wheel handles browser pages, Explorer icon sizing,
            # presentation canvases, and image viewers without relying on app names.
            self._focus_native_window_for_zoom(target)
            self.status = "Content zoom active — move hands apart to zoom in, together to zoom out"
        self.update()
        return True

    def _apply_middle_pinch_zoom_step(self, direction):
        """Dispatch a generic Ctrl/Command+wheel step at the live hand cursor.

        Unlike moving the cursor to the midpoint, this operates at the current OS
        pointer location. Ctrl+wheel is supported by browsers, Explorer, many image
        viewers, and presentation canvases. No application-name heuristics are used.
        """
        if pyautogui is None:
            self.status = "Content zoom unavailable: pyautogui is not installed"
            return False
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        modifier_down = False
        try:
            pyautogui.keyDown(modifier, _pause=False)
            modifier_down = True
            pyautogui.scroll(1 if direction > 0 else -1, _pause=False)
            return True
        except Exception as exc:
            self.status = f"Content zoom input failed: {exc}"
            return False
        finally:
            if modifier_down:
                try:
                    pyautogui.keyUp(modifier, _pause=False)
                except Exception:
                    pass

    def _update_middle_pinch_zoom(self, cursor_points, now):
        """Apply distance changes to the chosen window bounds or its content zoom."""
        state = self.middle_pinch_zoom
        points = {side: self._virtual_point_to_global(cursor_points[side])
                  for side in ("left", "right")}
        if any(point is None for point in points.values()):
            return
        current = {side: (float(point.x()), float(point.y())) for side, point in points.items()}
        if state.get("mode") == "resize":
            rect = resize_rect_from_corner_deltas(
                state["rect"], state["start_points"], current, state["corners"])
            if rect is None:
                return
            previous = state.get("last_resize_rect")
            moved_enough = previous is None or any(abs(a-b) >= 2.0 for a, b in zip(rect, previous))
            # wmctrl resizing spawns a process on X11; cap it at ~15 Hz to keep the
            # new built-in gesture from adding a large CPU cost during a resize.
            if (moved_enough and now-float(state.get("last_resize_at", 0.0)) >= 0.065):
                left, top, right, bottom = rect
                resized = self._resize_native_window(state.get("hwnd"), left, top,
                                                      right-left, bottom-top)
                state["last_resize_at"] = now
                if resized:
                    state["last_resize_rect"] = rect
            return
        distance = math.hypot(current["right"][0]-current["left"][0],
                              current["right"][1]-current["left"][1])
        if state.get("mode") == "resize_scale":
            rect = resize_rect_from_distance_scale(
                state.get("rect"), state.get("start_distance"), distance)
            if rect is None:
                return
            previous = state.get("last_resize_rect")
            moved_enough = previous is None or any(abs(a-b) >= 2.0 for a, b in zip(rect, previous))
            # Window-manager operations are intentionally throttled to avoid extra CPU
            # use while still providing smooth enough resizing by hand.
            if moved_enough and now-float(state.get("last_resize_at", 0.0)) >= 0.065:
                left, top, right, bottom = rect
                resized = self._resize_native_window(state.get("hwnd"), left, top,
                                                      right-left, bottom-top)
                state["last_resize_at"] = now
                if resized:
                    state["last_resize_rect"] = rect
            state["last_distance"] = distance
            return
        previous_distance = state.get("last_distance")
        state["last_distance"] = distance
        if previous_distance is None:
            return
        # Ignore small tremor and emit zoom keyboard steps only after a meaningful
        # change in hand separation. No OS cursor repositioning is required.
        state["zoom_remainder"] = float(state.get("zoom_remainder", 0.0)) + (distance-previous_distance)
        remainder = state["zoom_remainder"]
        if abs(remainder) < 34.0:
            return
        emitted = 0
        while abs(remainder) >= 34.0 and emitted < 3:
            direction = 1 if remainder > 0 else -1
            if not self._apply_middle_pinch_zoom_step(direction):
                state["zoom_remainder"] = 0.0
                return
            remainder -= 34.0*direction
            emitted += 1
        state["zoom_remainder"] = remainder

    def _process_middle_pinch_zoom(self, cursor_points, poses, now):
        """Consume the two-hand thumb-middle pinch and its shared Pinch Menu buffer."""
        state = self.middle_pinch_zoom
        # A closed grip has the middle fingertip naturally folded near the thumb.
        # Never allow the raw middle-pinch classifier to steal an ordinary fist/grab.
        middle = {
            side: bool(
                poses.get(side, {}).get("middle_pinch")
                and not poses.get(side, {}).get("fist", False)
                and side in cursor_points
            )
            for side in ("left", "right")
        }
        any_middle = middle["left"] or middle["right"]
        both_middle = middle["left"] and middle["right"]
        if not any_middle and state.get("started_at") is None and not state.get("active"):
            return False
        # A middle-finger pinch is reserved for this built-in action. If the glass UI owns
        # gestures, consume it without touching underlying application windows.
        search_dialog = getattr(self, "search_dialog", None)
        modal = (self._gesture_ui_modal_open() or self.tli_calibration_active_side is not None or
                 getattr(self, "hand_calibration_active", False) or self.voice_selection_mode is not None or
                 self.custom_gesture_recording is not None or
                 any(self.fist_action_menu_open.values()) or
                 (search_dialog is not None and search_dialog.isVisible()))
        if modal or self.desktop_mode != "os":
            was_active = bool(state.get("active"))
            self._reset_middle_pinch_zoom()
            if any_middle and not modal and self.status != "Pinch zoom and native window resize require OS desktop mode":
                self.status = "Pinch zoom and native window resize require OS desktop mode"
            elif was_active and not modal:
                self.status = "Pinch zoom and native window resize require OS desktop mode"
            return any_middle
        # The zoom/resize gesture is explicitly two-handed. A single middle pinch
        # with no session armed must fall through to normal gesture processing (not
        # consume the frame), preserving one-hand clicks and fist-grab window drag.
        if state.get("started_at") is None and not state.get("active") and not both_middle:
            return False

        if both_middle:
            state["last_both_seen_at"] = now
            if state.get("started_at") is None:
                state["started_at"] = now
                # Do not let this two-hand hold continue an in-progress click/drag/menu.
                for side in ("left", "right"):
                    self._end_pinch_drag(side, status=False)
                    if self.fist_mouse_down.get(side, False):
                        self._release_fist_mouse(side)
                    self.active_action[side] = None
                    self._reset_fist_gesture(side)
                    self._clear_menu_pinch_sequence(side)
                    self.active_push_sides.discard(side)
                if self.gesture_owner in ("left", "right"):
                    self.gesture_owner = None
            if not state.get("active") and now-float(state["started_at"]) >= self.pinch_buffer_seconds:
                if not self._begin_middle_pinch_zoom(cursor_points, now):
                    self._reset_middle_pinch_zoom()
            if state.get("active"):
                self._update_middle_pinch_zoom(cursor_points, now)
            self._request_scene_repaint(now)
            return True
        # Tolerate a brief one-frame landmark glitch, but release if either pinch ends.
        if state.get("started_at") is not None:
            last_seen = state.get("last_both_seen_at") or state.get("started_at")
            if now-float(last_seen) <= 0.16:
                self._request_scene_repaint(now)
                return True
            was_active = bool(state.get("active"))
            self._reset_middle_pinch_zoom()
            if was_active:
                self.status = "Pinch zoom/resize released"
            self._request_scene_repaint(now)
        return any_middle

    def _paint_middle_pinch_zoom_timer(self, painter, now=None):
        """Draw the shared buffer countdown on each hand while pinch zoom is arming."""
        state = getattr(self, "middle_pinch_zoom", {})
        started = state.get("started_at") if isinstance(state, dict) else None
        if started is None:
            return
        now = time.monotonic() if now is None else float(now)
        if state.get("active"):
            points = [self.side_cursor_positions.get(side) for side in ("left", "right")]
            if all(point is not None for point in points):
                centre = QPointF((points[0].x()+points[1].x())*0.5,
                                 (points[0].y()+points[1].y())*0.5)
                label = "RESIZE WINDOW" if state.get("mode") == "resize" else "PINCH ZOOM"
                badge = QRectF(centre.x()-68, centre.y()-16, 136, 32)
                painter.setPen(QPen(QColor(90, 235, 255, 210), 1.5))
                painter.setBrush(QColor(12, 24, 36, 185))
                painter.drawRoundedRect(badge, 8, 8)
                painter.setPen(QColor(220, 252, 255, 245))
                painter.setFont(self._ui_font(8, QFont.DemiBold))
                painter.drawText(badge, Qt.AlignCenter, label)
            return
        elapsed = max(0.0, now-float(started))
        remaining = max(0.0, self.pinch_buffer_seconds-elapsed)
        progress = min(1.0, elapsed/max(0.01, self.pinch_buffer_seconds))
        for side in ("left", "right"):
            cursor = self.side_cursor_positions.get(side)
            if cursor is None:
                continue
            ring = QRectF(cursor.x()-39, cursor.y()-39, 78, 78)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(95, 235, 255, 80), 3))
            painter.drawEllipse(ring)
            painter.setPen(QPen(QColor(95, 235, 255, 245), 4))
            painter.drawArc(ring, 90*16, -int(progress*360*16))
            painter.setPen(QColor(225, 252, 255, 245))
            painter.setFont(self._ui_font(8, QFont.DemiBold))
            painter.drawText(QRectF(cursor.x()-24, cursor.y()+25, 48, 16), Qt.AlignCenter,
                             f"{remaining:.1f}s")

    def update_scene(self):
        """Mutually-exclusive quick clicks, direct fist manipulation, and Pinch Menu gestures."""
        # Track which hand toggled this frame so only that hand's gesture processing pauses.
        gesture_tracking_just_toggled = set()
        self._timer_during_sound_seen_keys.clear()
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
            self._stop_inactive_timer_during_sounds()
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
            self._stop_inactive_timer_during_sounds()
            return

        menu_state = bool(self.menu_open and self.hub_visible)
        if self._applied_desktop_mode != self.desktop_mode or getattr(self, "_applied_menu_state", None) != menu_state:
            self.apply_desktop_mode()

        self.hand_positions = {"left": [], "right": []}
        self.hand_stable_positions = {"left": None, "right": None}
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
                window.set_phase("Hand not detected — show the hand and pinch on the highlighted target", 0)
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
            # Stabilise the drawn shape, not the camera data used for gestures,
            # cursor tracking or the existing two-hand contact calibration.
            stable_visual = self._hand_visual_motion_filter.update(side, self.hand_positions[side], now)
            self.hand_stable_positions[side] = stable_visual
            self.hand_display_positions[side] = (
                self.hand_positions[side], self._hand_tucker.update(side, stable_visual, now))
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

            edge_geometry = self._interaction_screen_geometry()
            edge_width = edge_geometry.width() if edge_geometry is not None else self.width()
            edge_height = edge_geometry.height() if edge_geometry is not None else self.height()
            edge_tolerant = (cursor.x() <= edge_width*0.045 or
                             cursor.x() >= edge_width*0.955 or
                             cursor.y() <= edge_height*0.055 or
                             cursor.y() >= edge_height*0.945)
            # Peace sign is the combined per-hand cursor + gesture tracking toggle.
            # It is recognized even while tracking is disabled so it can re-enable it.
            peace = False if calibration_modal_frame else self._is_peace_sign(landmarks)
            if peace:
                # This display-only anchor is deliberately separate from
                # side_cursor_positions, which becomes None when tracking is disabled.
                # It never drives pointer movement or any interaction.
                self.peace_timer_anchor_by_side[side] = QPointF(cursor)
                if self.peace_hold_started[side] is None:
                    self.peace_hold_started[side] = now
                # The same selected "gesture timer during" sound accompanies both
                # disabling and re-enabling tracking. Play it once per held pose.
                if not self.peace_toggle_latched[side]:
                    self._maybe_play_timer_during_sound(
                        f"peace_toggle:{side}", self.peace_hold_started[side], now)
                if (now - self.peace_hold_started[side] >= self.peace_toggle_hold_seconds
                        and not self.peace_toggle_latched[side]):
                    enabled = not self.gesture_tracking_enabled.get(side, True)
                    self.gesture_tracking_enabled[side] = enabled
                    self.mouse_detached_by_side[side] = not enabled
                    self.peace_toggle_latched[side] = True
                    self._stop_timer_during_sound(f"peace_toggle:{side}")
                    gesture_tracking_just_toggled.add(side)
                    state = "enabled" if enabled else "disabled"
                    self.status = f"{side.title()} hand cursor and gestures {state}"
                    if enabled and all(self.gesture_tracking_enabled.get(other, True)
                                       for other in ("left", "right") if other != side):
                        self.hand_tracking_notice_text = f"{side.title()} cursor + gestures enabled"
                        self.hand_tracking_notice_until = now + 5.0
                    else:
                        self._refresh_hand_tracking_notice(now)
                    self.tts.speak(f"{side} {state}.")
                    self.menu_pinch_started[side] = None
                    self.fist_started_at[side] = None
                    self.action_stop_wait_open[side] = True
                    self.action_stop_open_since[side] = None
                    if not enabled:
                        self._end_pinch_drag(side, status=False)
                        self.active_action[side] = None
                        self.fist_action_menu_open[side] = False
                        self.fist_hovered_action[side] = None
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                    else:
                        self.side_cursor_positions[side] = None
                        self.smoothed_side_cursor_positions[side] = None
                        self.cursor_updated_at[side] = 0.0
                        self.tli_overrides.pop(side, None)
            else:
                self._stop_timer_during_sound(f"peace_toggle:{side}")
                self.peace_hold_started[side] = None
                self.peace_toggle_latched[side] = False
                self.peace_timer_anchor_by_side[side] = None

            # Disabled hands retain visible landmarks and peace-sign recognition,
            # but must never drive the native pointer or other gestures.
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

            # Explore Garden <-> Space before Garden props, fist scrolling, pinch
            # menus or constellation drawing. Hovering the arrow consumes gestures.
            if (self.desktop_mode == "virtual" and not calibration_modal_frame and not peace and
                    self.virtual_scene in ("garden", "space") and not self._gesture_ui_modal_open()):
                if self._scene_arrow_handle_frame(side, cursor, landmarks, now, edge_tolerant):
                    continue
            # Interact with Garden props BEFORE fist/Pinch Menu classification,
            # gesture-owner arbitration and OS mouse dragging. The interaction
            # retains its own pinch state and cannot arm the Pinch Menu timer.
            if self.virtual_garden_mode and self.desktop_mode == "virtual" and not calibration_modal_frame:
                active_ball = self._garden_props["ball"].get("held") == side
                active_dial = self._garden_sundial_hand_owned[side]
                object_pinch = self._garden_object_pinch(
                    landmarks, held=bool(active_ball or active_dial))
                if self._garden_handle_ball_frame(side, cursor, object_pinch, now):
                    if active_ball or self._garden_props["ball"].get("held") == side:
                        self.active_push_sides.add(side)
                    continue
                if self._garden_handle_sundial_frame(side, cursor, object_pinch, now):
                    continue
            if (self.virtual_space_mode and self.desktop_mode == "virtual" and
                    not calibration_modal_frame and not peace and not self._gesture_ui_modal_open()):
                # A closed fist must not masquerade as a thumb/index drawing pinch.
                draw_pinch = (not is_fist_or_grip_pose(landmarks, edge_tolerant=edge_tolerant)
                              and self._garden_object_pinch(
                                  landmarks, held=self.space_drawing_by_side.get(side) is not None))
                if self._space_handle_frame(side, cursor, draw_pinch, now):
                    continue
            if (self.virtual_space_mode and self._gesture_ui_modal_open() and
                    self.space_drawing_by_side.get(side) is not None):
                self._space_finish_constellation(side)
            raw_fist = is_fist_or_grip_pose(landmarks, edge_tolerant=edge_tolerant)
            raw_pinch = is_thumb_index_menu_pinch(landmarks, edge_tolerant=edge_tolerant)
            raw_middle_pinch = is_thumb_middle_zoom_pinch(landmarks, edge_tolerant=edge_tolerant)
            raw_open = is_open_hand_pose(landmarks, edge_tolerant=edge_tolerant)
            if not self.gesture_tracking_enabled.get(side, True) or side in gesture_tracking_just_toggled:
                raw_fist = raw_pinch = raw_middle_pinch = raw_open = False
            if peace:
                raw_fist = raw_pinch = raw_middle_pinch = raw_open = False
            calibration_consumes_gesture = self._process_tli_calibration_frame(
                side, landmarks, raw_pinch)
            if calibration_modal_frame or calibration_consumes_gesture:
                # Calibration pinches must never also click, drag, or start built-in timers.
                raw_fist = raw_pinch = raw_middle_pinch = raw_open = False
            # Treat the open glass menu/settings as a gesture-modal surface. A hand
            # outside the visible panel may still be tracked/rendered, but its pose
            # cannot start a pinch timer, drag, scroll, or click through to an app.
            search_surface_active = (self.search_dialog is not None and self.search_dialog.isVisible())
            clipboard_drag_in_progress = self.floating_clipboard_drag_state.get(side) is not None
            if (self.menu_open and self.hub_visible and not search_surface_active and
                    not self._canvas_menu_panel_contains(cursor) and
                    not self._floating_clipboard_contains(cursor) and
                    not clipboard_drag_in_progress):
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
                "middle_pinch": raw_middle_pinch,
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
            self._stop_inactive_timer_during_sounds()
            return

        if self._process_middle_pinch_zoom(cursor_points, poses, now):
            # While this two-hand pose is arming or active, do not let it leak into
            # custom-pose recognition, pinch-menu timers, clicks or ordinary pointer moves.
            self.pointer_side = None
            self.active_push_sides.clear()
            self.depth += (self.target_depth-self.depth)*0.16
            self._request_scene_repaint(now)
            self._stop_inactive_timer_during_sounds()
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
            self._stop_inactive_timer_during_sounds()
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
            # A lost hand must not leave the scene-arrow pinch latched, otherwise
            # its next appearance could skip the first intentional pinch-click.
            self._scene_arrow_pinch_previous[side] = False
            self._scene_arrow_pinch_latched[side] = False
            clipboard_was_active = self.floating_clipboard_drag_state.get(side) is not None
            if clipboard_was_active:
                clipboard_state = self.floating_clipboard_drag_state.get(side) or {}
                # If tracking drops during a non-modal text/image drag, treat it as a
                # release at the last OS cursor point rather than silently discarding it.
                # The preview follows the hand and the destination receives clipboard paste
                # without entering Qt's blocking native drag loop.
                if (clipboard_state.get("running") and
                        clipboard_state.get("mode") in ("text_paste", "image_paste", "files_paste") and
                        clipboard_state.get("moved")):
                    entry_id = clipboard_state.get("entry_id")
                    entry_obj = next((item for item in self.floating_clipboard_entries
                                      if item.get("id") == entry_id), None)
                    drop_global = QCursor.pos()
                    if entry_obj is not None:
                        QTimer.singleShot(0, lambda item=entry_obj, pos=drop_global:
                                          self._paste_floating_clipboard_entry_at(item, pos))
                elif clipboard_state.get("running") and pyautogui is not None:
                    try:
                        pyautogui.mouseUp(button="left", _pause=False)
                    except Exception:
                        pass
                self.floating_clipboard_drag_state[side] = None
            was_active = (self.fist_started_at[side] is not None or
                          self.menu_pinch_started[side] is not None or
                          self.pinch_drag_active[side] or
                          self.active_action[side] is not None or
                          self.fist_action_menu_open[side] or clipboard_was_active)
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

            # Floating clipboard cards take gesture ownership before ordinary pinch/fist
            # clicks. Moving a held gesture out of a card starts a real OS MIME drag; a
            # short hold/release simply loads that item into the system clipboard.
            clipboard_state = self.floating_clipboard_drag_state.get(side)
            if clipboard_state is not None:
                clipboard_state["last_cursor"] = QPointF(cursor)
                held_for_clipboard = bool(is_fist or is_pinch)
                if held_for_clipboard:
                    clipboard_state["last_held_at"] = now
                    clipboard_state["release_candidate_since"] = None
                elif clipboard_state.get("release_candidate_since") is None:
                    clipboard_state["release_candidate_since"] = now
                movement = math.hypot(cursor.x()-clipboard_state["start"].x(),
                                      cursor.y()-clipboard_state["start"].y())
                clipboard_state["moved"] = movement >= 18.0
                if (clipboard_state.get("type") == "control" and
                        clipboard_state.get("control") == "header"):
                    if held_for_clipboard:
                        if movement >= 6.0:
                            panel_layout = self._floating_clipboard_layout()
                            panel_start = clipboard_state.get("panel_start")
                            if panel_layout is not None and panel_start is not None:
                                panel_rect = panel_layout["panel"]
                                new_x = panel_start.x()+cursor.x()-clipboard_state["start"].x()
                                new_y = panel_start.y()+cursor.y()-clipboard_state["start"].y()
                                new_x = max(4.0, min(new_x, max(4.0, self.width()-panel_rect.width()-4.0)))
                                new_y = max(4.0, min(new_y, max(4.0, self.height()-panel_rect.height()-4.0)))
                                self.floating_clipboard_position = QPointF(new_x, new_y)
                                self.floating_clipboard_drag_state[side]["moved"] = True
                                self.update()
                                overlay = getattr(self, "search_hand_overlay", None)
                                if overlay is not None:
                                    overlay.update()
                        continue
                    # Releasing on the header ends a possible panel move; there is no click action.
                    if (clipboard_state.get("release_candidate_since") is not None and
                            now-clipboard_state["release_candidate_since"] >= 0.14):
                        self.floating_clipboard_drag_state[side] = None
                        self._clear_menu_pinch_sequence(side)
                        self._reset_fist_gesture(side)
                        if self.gesture_owner == side:
                            self.gesture_owner = None
                        self.action_stop_wait_open[side] = True
                        self.action_stop_open_since[side] = None
                    continue
                if clipboard_state.get("running"):
                    # Plain text and copied bitmap items use a non-modal cursor-follow
                    # + click/paste path. QDrag.exec enters a platform-native nested drag
                    # loop, which can starve camera/gesture updates on some Windows/Qt
                    # combinations and can prevent the preview from following the hand.
                    if clipboard_state.get("mode") in ("text_paste", "image_paste", "files_paste"):
                        self._set_os_pointer(cursor)
                        release_since = clipboard_state.get("release_candidate_since")
                        should_finish = (not held_for_clipboard and (
                            (release_since is not None and now-release_since >= 0.14) or
                            now-clipboard_state.get("last_held_at", now) > 0.65))
                        if should_finish:
                            entry_id = clipboard_state.get("entry_id")
                            entry_obj = next((item for item in self.floating_clipboard_entries
                                              if item.get("id") == entry_id), None)
                            self.floating_clipboard_drag_state[side] = None
                            self.floating_clipboard_drag_running = False
                            self.fist_mouse_down[side] = False
                            self.pinch_drag_active[side] = False
                            self.active_action[side] = None
                            self._reset_fist_gesture(side)
                            self._clear_menu_pinch_sequence(side)
                            if self.gesture_owner == side:
                                self.gesture_owner = None
                            self.action_stop_wait_open[side] = True
                            self.action_stop_open_since[side] = None
                            # _set_os_pointer(cursor) has already put the native pointer
                            # at this hand position. Use Qt's authoritative current global
                            # point for the drop instead of recomputing from the overlay's
                            # potentially work-area-offset origin.
                            drop_global = QCursor.pos()
                            if entry_obj is not None and clipboard_state.get("moved"):
                                QTimer.singleShot(0, lambda item=entry_obj, pos=drop_global:
                                                   self._paste_floating_clipboard_entry_at(item, pos))
                            elif entry_obj is not None:
                                self._set_system_clipboard_from_entry_id(entry_id)
                            self.update()
                            overlay = getattr(self, "search_hand_overlay", None)
                            if overlay is not None:
                                overlay.update()
                        continue
                    # Native file/image/link drags release their OS mouse grab when the
                    # hand opens. This path deliberately does not handle plain text.
                    if ((is_open and not held_for_clipboard) or
                            (not held_for_clipboard and now-clipboard_state.get("last_held_at", now) > 1.0)):
                        if not clipboard_state.get("release_sent") and pyautogui is not None:
                            try:
                                pyautogui.mouseUp(button="left", _pause=False)
                            except Exception:
                                pass
                            clipboard_state["release_sent"] = True
                    continue
                if (held_for_clipboard and clipboard_state.get("type") == "entry" and
                        clipboard_state.get("can_drag") and movement >= 22.0 and
                        now-clipboard_state.get("started_at", now) >= 0.08):
                    entry_id = clipboard_state.get("entry_id")
                    entry_obj = next((item for item in self.floating_clipboard_entries
                                      if item.get("id") == entry_id), None)
                    if entry_obj is not None and entry_obj.get("kind") in ("text", "image", "files"):
                        # Use the non-modal cursor-follow/paste path for text, image and
                        # grouped file entries. QDrag.exec() can crash or stall while
                        # transferring a multi-URL Windows Explorer selection; publishing
                        # the entire URL list to the system clipboard and pasting at the
                        # drop point preserves the group as one card and avoids that path.
                        clipboard_state["running"] = True
                        clipboard_state["mode"] = {
                            "text": "text_paste", "image": "image_paste", "files": "files_paste"
                        }[entry_obj.get("kind")]
                        self.floating_clipboard_drag_running = True
                        if entry_obj.get("kind") == "text":
                            self.status = "Dragging clipboard text — release over a text field to paste"
                        elif entry_obj.get("kind") == "image":
                            self.status = "Dragging clipboard image — release over an image target or folder to paste"
                        else:
                            self.status = (f"Dragging {len(entry_obj.get('urls') or [])} copied file(s) "
                                           "as one group — release over a folder or compatible target to paste")
                        continue
                    clipboard_state["running"] = True
                    clipboard_state["mode"] = "native"
                    self.floating_clipboard_drag_running = True
                    drag_id = clipboard_state["id"]
                    QTimer.singleShot(0, lambda captured_side=side, captured_id=drag_id:
                                       self._run_floating_clipboard_native_drag(captured_side, captured_id))
                    continue
                if is_open and not held_for_clipboard:
                    if clipboard_state.get("type") == "entry":
                        entry_id = clipboard_state.get("entry_id")
                        entry_obj = next((item for item in self.floating_clipboard_entries
                                          if item.get("id") == entry_id), None)
                        if entry_obj is not None:
                            self._set_system_clipboard_from_entry_id(entry_id)
                            if clipboard_state.get("moved") and not clipboard_state.get("can_drag"):
                                self.status = "Clipboard item selected; attach this hand to drag it into desktop apps"
                    else:
                        control = clipboard_state.get("control")
                        if control == "close":
                            self._set_floating_clipboard_visible(False)
                        elif control == "delete":
                            self._delete_floating_clipboard_entry(clipboard_state.get("entry_id"))
                        elif control == "older":
                            self.floating_clipboard_scroll_offset = min(
                                max(0, len(self.floating_clipboard_entries)-1),
                                self.floating_clipboard_scroll_offset+5)
                        elif control == "newer":
                            self.floating_clipboard_scroll_offset = max(0, self.floating_clipboard_scroll_offset-5)
                        self.update()
                    self.floating_clipboard_drag_state[side] = None
                    self._clear_menu_pinch_sequence(side)
                    self._reset_fist_gesture(side)
                    if self.gesture_owner == side:
                        self.gesture_owner = None
                    self.action_stop_wait_open[side] = True
                    self.action_stop_open_since[side] = None
                    continue
                if not held_for_clipboard and now-clipboard_state.get("last_held_at", now) > 0.65:
                    self.floating_clipboard_drag_state[side] = None
                    if self.gesture_owner == side:
                        self.gesture_owner = None
                else:
                    continue
            elif (self.floating_clipboard_visible and (is_fist or is_pinch) and
                  self.active_action.get(side) is None and not self.pinch_drag_active.get(side, False) and
                  not self.fist_action_menu_open.get(side, False) and
                  not self.action_stop_wait_open.get(side, False) and
                  self.fist_started_at.get(side) is None and self.menu_pinch_started.get(side) is None and
                  not self.left_click_modifier_active and self.gesture_owner in (None, side)):
                hit = self._floating_clipboard_hit_test(cursor)
                if hit is not None:
                    can_drag = bool(self.desktop_mode == "os" and pyautogui is not None and
                                    not self.mouse_detached_by_side.get(side, False))
                    selected_entry = hit.get("entry")
                    preview_pixmap = self._clipboard_drag_pixmap(selected_entry) if selected_entry is not None else None
                    panel_layout = self._floating_clipboard_layout() if (
                        hit.get("type") == "control" and hit.get("name") == "header") else None
                    panel_start = QPointF(panel_layout["panel"].topLeft()) if panel_layout is not None else None
                    state = {"id": uuid.uuid4().hex, "type": hit["type"],
                             "entry_id": (selected_entry.get("id") if selected_entry is not None
                                          else hit.get("entry_id")),
                             "control": hit.get("name"), "start": QPointF(cursor),
                             "panel_start": panel_start,
                             "last_cursor": QPointF(cursor), "started_at": now,
                             "last_held_at": now, "release_candidate_since": None,
                             "preview_pixmap": preview_pixmap, "running": False, "moved": False,
                             "can_drag": can_drag, "release_sent": False}
                    self.floating_clipboard_drag_state[side] = state
                    self.gesture_owner = side
                    self.canvas_active_side = side
                    self.active_push_sides.add(side)
                    continue

            # When the Virtual Pen is selected, thumb/index pinch is drawing input on
            # the virtual canvas, not a native click or Pinch Menu hold. Dock/palette
            # regions are excluded so they remain clickable with a short pinch.
            if self.virtual_pen_active_stroke is not None and self.virtual_pen_active_stroke.get("side") == side:
                if self._handle_virtual_pen_frame(side, cursor, is_pinch, now):
                    continue
            elif (self.desktop_mode == "virtual" and self.virtual_pen_active and is_pinch
                  and not (self.menu_open and self.hub_visible) and not self.settings_open
                  and not self.floating_clipboard_visible and not self._virtual_pen_ui_contains(cursor)):
                if self._handle_virtual_pen_frame(side, cursor, is_pinch, now):
                    continue

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
                mode = active.get("mode")
                one_hand_fist_scroll = (mode == "scroll_window" and
                                        not active.get("two_hand_clutch", False))
                if one_hand_fist_scroll:
                    # A brief false-open classification makes the scroll anchor feel as
                    # though it has been released. Keep scrolling through these frames;
                    # only a sustained, clearly open hand with genuinely extended fingers
                    # may release the right-fist scroll.
                    clear_open_pose = (is_open and not is_fist and not is_pinch and
                                       not bool(pose.get("fingers_closed", False)))
                    if clear_open_pose:
                        candidate_since = active.get("scroll_open_candidate_since")
                        if candidate_since is None:
                            candidate_since = now
                            active["scroll_open_candidate_since"] = candidate_since
                        open_duration = max(0.0, now - candidate_since)
                        if not active_fist_scroll_release_confirmed(
                                is_open=is_open, is_fist=is_fist, is_pinch=is_pinch,
                                fingers_closed=bool(pose.get("fingers_closed", False)),
                                open_duration=open_duration, release_seconds=0.48):
                            is_fist, is_pinch, is_open = True, False, False
                        else:
                            # Reuse the candidate start as the release clock so the
                            # ordinary cooldown does not add a second delay.
                            active["open_since"] = candidate_since
                    else:
                        active["scroll_open_candidate_since"] = None
                # While any drag/scroll is active, keep its held-grip interpretation for
                # non-open frames so pinch/ambiguous classifications cannot steal it.
                if mode in ("drag_window", "scroll_window") and not is_open:
                    is_fist, is_pinch = True, False
                self.gesture_owner = side
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
                    release_delay = (0.14 if active.get("modifier_drag") else
                                     (0.48 if one_hand_fist_scroll else self.active_release_cooldown_seconds))
                    if now-active["open_since"] >= release_delay:
                        if mode == "right_click":
                            self._activate_canvas_right_click(active.get("origin"), side, now)
                            self.status = "Right-click sent at the mode's starting TPI"
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
                self._maybe_play_timer_during_sound(
                    f"pinch_menu:{side}", self.menu_pinch_started[side], now)
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
                                self._stop_inactive_timer_during_sounds()
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
                                self._stop_inactive_timer_during_sounds()
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
                if self.fist_action.get(side) != "slider" and not self.fist_start_controls.get(side):
                    self._maybe_play_timer_during_sound(
                        f"fist:{side}", self.fist_started_at[side], now)
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
                                self._stop_inactive_timer_during_sounds()
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

        self._stop_inactive_timer_during_sounds()
        self.depth += (self.target_depth-self.depth)*0.16
        self._request_scene_repaint(now)

    def closeEvent(self, event):
        for key in list(self._timer_during_sound_stop_events):
            self._stop_timer_during_sound(key)
        if getattr(self, "virtual_pen_active_stroke", None) is not None:
            self._finish_virtual_pen_stroke()
        for side in ("left", "right", "mouse"):
            self._space_finish_constellation(side)
        pending_virtual_save = getattr(self, "virtual_desktop_save_timer", None)
        if pending_virtual_save is not None and pending_virtual_save.isActive():
            pending_virtual_save.stop()
            self._save_virtual_desktop_data()
        self._stop_native_custom_key_capture()
        self.custom_gesture_key_capture = None
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
        if self.space_drawing_by_side.get("mouse") is not None:
            self._space_extend_constellation("mouse", event.position(), time.monotonic())
        elif self._garden_props["ball"].get("held") == "mouse":
            # The original Garden mouseMoveEvent above is shadowed by this method.
            ball = self._garden_props["ball"]
            point = event.position()
            width, height = max(1., float(self.width())), max(1., float(self.height()))
            x = min(1.0-ball["r"], max(ball["r"], point.x()/width+ball.get("offset_x", 0.0)))
            y = min(1.0-ball["r"], max(ball["r"], point.y()/height+ball.get("offset_y", 0.0)))
            now = time.monotonic()
            dt = max(.008, now-float(ball.get("last_drag_at", now)))
            ball["vx"] = max(-1.8, min(1.8, (x-ball["x"])/dt))
            ball["vy"] = max(-1.8, min(1.8, (y-ball["y"])/dt))
            ball["x"], ball["y"] = x, y
            ball["last_drag_at"] = now
            self.update()
        event.accept()

    def _stop_native_custom_key_capture(self):
        timer = getattr(self, "custom_key_capture_poll_timer", None)
        if timer is not None:
            timer.stop()
        capture = getattr(self, "_native_shortcut_capture", None)
        self._native_shortcut_capture = None
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass

    def _cancel_custom_gesture_key_capture(self, message=None):
        self._stop_native_custom_key_capture()
        self.custom_gesture_key_capture = None
        if message:
            self.custom_gesture_message = message
        self.update()

    def _start_custom_gesture_key_capture(self, index):
        if not (0 <= index < len(self.custom_gestures)):
            return
        self._stop_native_custom_key_capture()
        gesture = self.custom_gestures[index]
        gesture["action"] = "custom_key"
        self.custom_gesture_key_capture = index
        self.custom_gesture_dropdown = None
        self._save_custom_gestures_now()
        self.custom_gesture_message = (
            f"Capturing shortcut for {gesture.get('label', 'gesture')}: press the key combo now. "
            "On Windows, Win+Tab is captured without opening the app switcher. Esc cancels.")
        self.setFocus(Qt.OtherFocusReason)
        self.activateWindow()
        if sys.platform.startswith("win"):
            capture = WindowsShortcutCapture()
            if capture.start():
                self._native_shortcut_capture = capture
                timer = getattr(self, "custom_key_capture_poll_timer", None)
                if timer is not None:
                    timer.start()
            else:
                self.custom_gesture_message = (
                    "Native keyboard capture could not start. Use Ctrl/Alt/Shift shortcuts; "
                    "Windows may intercept Win+Tab.")
        elif sys.platform.startswith("linux"):
            # Qtile consumes many Super shortcuts (e.g. Super+A) before Qt sees
            # keyPressEvent. On X11/XLibre, temporarily grab the keyboard globally
            # so the chosen combo is captured instead of launching the OS shortcut.
            capture = X11ShortcutCapture()
            if capture.start():
                self._native_shortcut_capture = capture
                timer = getattr(self, "custom_key_capture_poll_timer", None)
                if timer is not None:
                    timer.start()
            else:
                self.custom_gesture_message = (
                    "Global shortcut capture is unavailable. On Linux this requires an X11/XLibre "
                    "session; the window manager may intercept Super shortcuts otherwise.")
        self.update()

    def _save_captured_custom_key_combo(self, index, combo):
        if not (0 <= index < len(self.custom_gestures)):
            self._cancel_custom_gesture_key_capture("Shortcut capture cancelled: gesture no longer exists.")
            return
        parts = []
        for raw in str(combo or "").split("+"):
            token = raw.strip()
            if not token:
                continue
            if token.lower() in ("meta", "win", "windows", "super", "super_l", "super_r",
                                 "command", "cmd", "command_l", "command_r"):
                parts.append("META")
            else:
                parts.append(token.upper())
        if not parts:
            self._cancel_custom_gesture_key_capture("No shortcut captured. Click KEY to try again.")
            return
        gesture = self.custom_gestures[index]
        gesture["action"] = "custom_key"
        gesture["key_combo"] = "+".join(parts)
        self._stop_native_custom_key_capture()
        self.custom_gesture_key_capture = None
        self._save_custom_gestures_now()
        self.custom_gesture_message = (f"Saved shortcut {display_key_combo(gesture['key_combo'])} "
                                       f"for {gesture.get('label', 'gesture')}.")
        self.status = self.custom_gesture_message
        self.update()

    def _poll_custom_key_capture(self):
        capture = getattr(self, "_native_shortcut_capture", None)
        if capture is None:
            self.custom_key_capture_poll_timer.stop()
            return
        result, value = capture.poll_result()
        if result == "pending":
            return
        index = self.custom_gesture_key_capture
        self._stop_native_custom_key_capture()
        if result == "combo":
            self._save_captured_custom_key_combo(index, value)
        elif result == "cancel":
            self.custom_gesture_key_capture = None
            self.custom_gesture_message = "Custom key capture cancelled. Click KEY to try again."
            self.update()
        else:
            # Keep the ordinary Qt event-based capture available if the native hook
            # cannot be installed. A warning makes the Win+Tab limitation explicit.
            self.custom_gesture_key_capture = index
            if sys.platform.startswith("linux"):
                limitation = "Super shortcuts may still be intercepted by the window manager."
            else:
                limitation = "Windows may intercept system-reserved shortcuts."
            self.custom_gesture_message = (
                f"Native key capture unavailable ({value}). Ordinary shortcuts can still be captured; "
                f"{limitation} Press Esc to cancel.")
            self.update()

    def _custom_key_name_from_event(self, event):
        """Convert a Qt key event to a pyautogui-compatible shortcut token."""
        key = int(event.key())
        modifiers = event.modifiers()
        modifier_only = {
            int(getattr(Qt, name)) for name in (
                "Key_Control", "Key_Shift", "Key_Alt", "Key_Meta", "Key_AltGr",
                "Key_Super_L", "Key_Super_R", "Key_Command", "Key_Hyper_L", "Key_Hyper_R",
            ) if getattr(Qt, name, None) is not None
        }
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
            # META represents Windows/Super/Command independent of OS; translated
            # to pyautogui's platform-specific name only when the action is fired.
            parts.append("meta")
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
            native_capture = getattr(self, "_native_shortcut_capture", None)
            if native_capture is not None and native_capture.is_active():
                # The native Windows hook suppresses and records keyboard events
                # globally, including system chords such as Win+Tab.
                event.accept()
                return
            if event.key() == Qt.Key_Escape and not event.modifiers():
                self._cancel_custom_gesture_key_capture("Custom key capture cancelled. Click KEY to try again.")
                event.accept()
                return
            if event.isAutoRepeat():
                event.accept()
                return
            combo = self._custom_key_name_from_event(event)
            if combo:
                self._save_captured_custom_key_combo(capture_index, combo)
                event.accept()
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
            self._paint_virtual_background(painter, width, height)
            # Garden is its own scene: keep the saved grid canvas hidden, not layered
            # beneath the garden. Returning to Grid mode reveals those strokes again.
            if self.virtual_scene == "grid":
                canvas_layer = self._ensure_virtual_canvas_layer()
                if canvas_layer is not None and not canvas_layer.isNull():
                    painter.drawImage(QPointF(0, 0), canvas_layer)
            self._paint_virtual_app_dock(painter)
            active_stroke = self.virtual_pen_active_stroke
            if active_stroke is not None and active_stroke.get("points"):
                # The layer is incremental, but replay the active stroke after a resize
                # only when the layer is rebuilt; normally the cached layer already has it.
                pass

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

                # Quit Freehand sits at the top of the main menu and intentionally mirrors
                # the Close button's red gradient, border, and responsive red glow.
                exit_banner = self._main_menu_exit_rect(cx, cy, scale)
                exit_hovered = any(
                    point is not None and exit_banner.contains(point)
                    for point in self.side_cursor_positions.values()
                )
                exit_gradient = QLinearGradient(exit_banner.topLeft(), exit_banner.bottomRight())
                if exit_hovered:
                    for expansion, alpha in ((7*scale, 45), (3.5*scale, 90)):
                        painter.setPen(QPen(QColor(255, 90, 90, alpha), 3*scale))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRoundedRect(exit_banner.adjusted(-expansion, -expansion,
                                                                      expansion, expansion),
                                                12*scale+expansion, 12*scale+expansion)
                    exit_gradient.setColorAt(0.0, QColor(255, 105, 100, 245))
                    exit_gradient.setColorAt(1.0, QColor(215, 30, 40, 250))
                    painter.setPen(QPen(QColor(255, 215, 210, 250), 2.0*scale))
                else:
                    exit_gradient.setColorAt(0.0, QColor(200, 55, 60, 215))
                    exit_gradient.setColorAt(1.0, QColor(95, 12, 22, 235))
                    painter.setPen(QPen(QColor(255, 150, 150, 210), 1.5*scale))
                painter.setBrush(QBrush(exit_gradient))
                painter.drawRoundedRect(exit_banner, 12*scale, 12*scale)
                painter.setPen(QColor(240, 255, 255))
                painter.setFont(self._ui_font(max(8, int(10*scale)), QFont.DemiBold))
                painter.drawText(exit_banner, Qt.AlignCenter, "Quit Freehand")

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

                tab_labels = {"interaction": "INTERACTION", "hands": "HANDS", "tli": "TPI",
                              "notifications": "NOTIFY", "gestures": "GESTURES", "voice": "VOICE",
                              "d1": "ENGINE", "update": "UPDATE", "virtual": "VIRTUAL", "sounds": "SOUNDS"}
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
                            ("Move the TPI with your palm anchor.", False),
                            ("Quick pinch → release: left click.", False),
                            ("Quick grip → open: left click.", False),
                            ("Clicks use the gesture start point.", False),
                            ("", False),
                            ("LEFT-HAND MODIFIER", True),
                            ("Hold LEFT fist to enable the modifier.", False),
                            ("RIGHT pinch → release: right-click.", False),
                            ("RIGHT grip: grab the window under TPI.", False),
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
                            ("Both hands: thumb-middle pinch to zoom.", False),
                            ("Pinch opposite window corners to resize.", False),
                            ("Open hand to end a normal drag/scroll.", False),
                            ("", False),
                            ("SETTINGS", True),
                            ("Pinch/grip near a slider to adjust it.", False),
                            ("TPI is biased slightly toward the thumb.", False),
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
                            ("create new folder · create new text file", False),
                            ("create new powerpoint · new tab · close tab", False),
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
                                            "notifications", "voice", "update", "d1", "virtual", "sounds"):
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
                        self._draw_fitted_button_label(painter, glow_toggle, "TPI GLOW  ·  "+("ON" if self.tli_glow_enabled else "OFF"), base_size=label_size, min_size=6, inset=12*scale)

                        # Independent left/right TLI colour swatches are grouped here.
                        for label, key, color in (("LEFT TPI", "swatch_left", self.left_click_color),
                                                  ("RIGHT TPI", "swatch_right", self.right_click_color)):
                            swatch = controls[key]
                            centre = QPointF(swatch.center().x(), swatch.top()+24*scale)
                            painter.setBrush(QBrush(color))
                            painter.setPen(QPen(QColor(225, 250, 255, 230), 2*scale))
                            painter.drawEllipse(centre, 14*scale, 14*scale)
                            painter.setPen(QColor(210, 245, 240))
                            self._draw_fitted_button_label(
                                painter, QRectF(swatch.left(), swatch.top()+42*scale, swatch.width(), 20*scale),
                                label, base_size=label_size, min_size=6, inset=2)

                    elif self.settings_page == "virtual":
                        # Exactly the same three-way segmented style used by Hands → Render.
                        switch = controls["virtual_scene_switch"]
                        painter.setBrush(QBrush(QColor(10, 37, 55, 225)))
                        painter.setPen(QPen(QColor(125, 235, 245, 210), 1.5*scale))
                        painter.drawRoundedRect(switch, 10*scale, 10*scale)
                        caption_w = switch.width()*self.RENDER_SWITCH_CAPTION
                        painter.setPen(QColor(200, 240, 248))
                        self._draw_fitted_button_label(painter, QRectF(switch.left(), switch.top(), caption_w, switch.height()),
                                                       "SCENE", base_size=label_size, min_size=6, inset=8*scale)
                        for seg_index, (scene_name, seg) in enumerate(self._virtual_scene_segments(switch, scale)):
                            selected = scene_name == self.virtual_scene
                            if selected:
                                painter.setBrush(QBrush(QColor(55, 145, 82, 235)))
                                painter.setPen(QPen(QColor(155, 245, 180, 230), 1.2*scale))
                                painter.drawRoundedRect(seg.adjusted(1*scale, 0, -1*scale, 0), 8*scale, 8*scale)
                            elif seg_index > 0 and VIRTUAL_SCENES[seg_index-1] != self.virtual_scene:
                                painter.setPen(QPen(QColor(125, 235, 245, 90), 1.0*scale))
                                painter.drawLine(QPointF(seg.left(), seg.top()+6*scale),
                                                 QPointF(seg.left(), seg.bottom()-6*scale))
                            painter.setPen(QColor(245, 255, 250) if selected else QColor(175, 210, 220))
                            self._draw_fitted_button_label(painter, seg, VIRTUAL_SCENE_LABELS[scene_name],
                                                           base_size=label_size, min_size=6, inset=3*scale)

                        grid_button = controls["virtual_grid_toggle"]
                        grid_fill = QColor(55, 145, 82, 225) if self.virtual_grid_enabled else QColor(75, 80, 89, 205)
                        grid_edge = QColor(155, 245, 180, 220) if self.virtual_grid_enabled else QColor(155, 162, 172, 200)
                        painter.setBrush(QBrush(grid_fill))
                        painter.setPen(QPen(grid_edge, 1.5*scale))
                        painter.drawRoundedRect(grid_button, 10*scale, 10*scale)
                        painter.setPen(QColor(235, 255, 255))
                        self._draw_fitted_button_label(
                            painter, grid_button, "GRID  ·  "+("ON" if self.virtual_grid_enabled else "OFF"),
                            base_size=label_size, min_size=6, inset=12*scale)

                        style_button = controls["virtual_background_style"]
                        style_data = self._virtual_background_style_data()
                        painter.setBrush(QBrush(QColor(style_data["background"])))
                        painter.setPen(QPen(QColor(style_data["grid"]), 2.0*scale))
                        painter.drawRoundedRect(style_button, 10*scale, 10*scale)
                        painter.setPen(QColor(245, 255, 255))
                        self._draw_fitted_button_label(
                            painter, style_button, "BACKGROUND  ·  "+style_data["name"],
                            base_size=label_size, min_size=6, inset=12*scale)
                        painter.setPen(QColor(185, 225, 235))
                        self._draw_wrapped_fitted_text(
                            painter, text_rects["help"],
                            "Choose Grid for the pen canvas, Garden for interactive nature and weather, or Space for a glowing starfield. In Space, pinch and trace between stars to draw constellations; release to save. Click CLEAR in Space to erase constellations only. Your Grid drawings and Garden scene remain intact.",
                            base_size=max(7, int(8*scale)), min_size=6,
                            align=Qt.AlignLeft | Qt.AlignTop)

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

                        indicator_toggle = controls["disabled_indicator_toggle"]
                        indicator_enabled = self.show_gesture_disabled_indicator
                        painter.setBrush(QBrush(QColor(55, 145, 82, 225) if indicator_enabled else QColor(75, 80, 89, 205)))
                        painter.setPen(QPen(QColor(155, 245, 180, 220) if indicator_enabled else QColor(155, 162, 172, 200), 1.5*scale))
                        painter.drawRoundedRect(indicator_toggle, 10*scale, 10*scale)
                        painter.setPen(QColor(235, 255, 255))
                        self._draw_fitted_button_label(
                            painter, indicator_toggle,
                            "DISABLED GESTURE INDICATOR  ·  "+("ON" if indicator_enabled else "OFF"),
                            base_size=label_size, min_size=6, inset=12*scale)

                        # Each hand's independent calibration opens a temporary target window.
                        for calibration_side, rect in self._tli_calibration_button_rects(cx, cy, scale).items():
                            pending = self.tli_calibration_pending.get(calibration_side, False)
                            calibrated = self.tpi_screen_calibrations.get(calibration_side) is not None
                            if pending:
                                if self.tli_calibration_wait_release.get(calibration_side, False):
                                    label = f"{calibration_side.upper()} · RELEASE FIRST"
                                else:
                                    count = len(self.tli_calibration_samples.get(calibration_side, []))
                                    target_no = len(self.tpi_calibration_draft_points.get(calibration_side, [])) + 1
                                    label = (f"{calibration_side.upper()} · TARGET {target_no}/9" if count == 0 else
                                             f"{calibration_side.upper()} · {target_no}/9 · {count}/{self.tli_calibration_sample_count}")
                                fill = QColor(100, 130, 45, 235)
                            elif calibrated and self.tli_calibration_persisted.get(calibration_side, False):
                                label = f"{calibration_side.upper()} TPI · CALIBRATED"
                                fill = QColor(20, 104, 82, 225)
                            elif calibrated:
                                label = f"{calibration_side.upper()} · SESSION ONLY"
                                fill = QColor(125, 78, 28, 225)
                            else:
                                label = f"CALIBRATE {calibration_side.upper()} TPI · 9 POINTS"
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

                    elif self.settings_page == "sounds":
                        # Render every sound setting as a clearly visible, independent dropdown.
                        # Keep the label and current selection together so the page remains
                        # understandable even when assets/sounds contains no audio files yet.
                        for key, label in SOUND_SETTING_ROWS:
                            rect = controls.get("sound:" + key)
                            if rect is None:
                                continue
                            selected_sound = self.ui_preferences.get(key, "None") or "None"
                            active = key == (self.sound_dropdown_key or SOUND_SETTING_ROWS[0][0])
                            painter.setBrush(QBrush(QColor(45, 118, 78, 242) if active else QColor(10, 47, 67, 242)))
                            painter.setPen(QPen(QColor(160, 245, 185, 235) if active else QColor(140, 230, 240, 235), 1.7*scale))
                            painter.drawRoundedRect(rect, 10*scale, 10*scale)
                            painter.setPen(QColor(237, 252, 255))
                            display_sound = selected_sound if selected_sound != "None" else "None (silent)"
                            self._draw_fitted_button_label(
                                painter, rect, f"{label}  ·  {display_sound}",
                                base_size=max(7, int(8.5*scale)), min_size=6, inset=12*scale)
                        help_rect = text_rects.get("help")
                        if help_rect is not None:
                            painter.setPen(QColor(205, 235, 242))
                            self._draw_wrapped_fitted_text(
                                painter, help_rect,
                                "Select a row to choose a sound from assets/sounds. Add WAV, OGG, MP3, FLAC or AIFF files to populate the lists. Changes save automatically.",
                                base_size=max(7, int(8*scale)), min_size=6)

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
                    if self.settings_page == "sounds":
                        self._paint_sound_dropdown(painter, cx, cy, scale)
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
        # owner of hands/TPIs/timers/Pinch Menu. Painting the same layer here too causes
        # offset duplicate/ghost hands, often noticed around the screen centre.
        search_overlay = getattr(self, "search_hand_overlay", None)
        search_dialog = getattr(self, "search_dialog", None)
        search_overlay_draws_hand_layer = bool(
            search_overlay is not None and search_overlay.isVisible() and (
                (search_dialog is not None and search_dialog.isVisible()) or
                (sys.platform.startswith("win") and
                 getattr(search_overlay, "_global_hand_layer_active", False))
            )
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
            self._paint_floating_clipboard(painter, now)
            self._paint_middle_pinch_zoom_timer(painter, now)

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
                self._paint_gesture_disabled_indicator(painter, points, side)
                # Keep the virtual skeleton visible when a hand is disabled, but remove
                # its moving TLI/cursor until peace sign re-enables tracking.
                if self.gesture_tracking_enabled.get(side, True):
                    cursor = self.side_cursor_positions.get(side)
                    if cursor is None:
                        cursor = self._scale_hand_tli_point(self.interactive_hand_anchor(points, side), points, side)
                    cursor = self._visual_tli_position(side, cursor, points)
                    engaged = side in self.active_push_sides or (self.click_side == side and now < self.click_flash_until)
                    self._paint_tli(painter, cursor, side, engaged)

            # One countdown renderer for both ON->OFF and OFF->ON, even though
            # OFF hands deliberately have no live TLI or side cursor.
            for side in ("left", "right"):
                self._paint_peace_sign_timer(painter, side, now)

            # Custom gestures with their circle timer on show a filling ring at the TLI
            # (midway between both TPIs for the overlap gesture) until the gesture fires.
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
                press_rects.append(self._main_menu_exit_rect(pcx, pcy, pscale))
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
                        press_rects.extend([row["row"], row["side"], row["timer"], row["label"],
                                            row["action"], row["capture"], row["delete"]])
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
