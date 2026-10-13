"""Deterministic, offline phrase routing for Spatial Desktop PC controls.

This deliberately uses aliases/keywords instead of an LLM. Keep commands short,
explicit, and checked before search-box dictation receives final transcripts.
"""
from __future__ import annotations
import re

_ALIASES = {
    # Dictation controls (handled globally before focused-field text routing)
    "stop_dictation": ("stop dictation", "end dictation", "and dictation", "dictation off", "stop typing"),
    "pause_dictation": ("pause dictation", "hold dictation", "dictation pause"),
    "resume_dictation": ("resume dictation", "continue dictation", "dictation resume"),

    # Clipboard and spatial selection
    "copy": ("copy this", "copy text", "copy selection", "copy selected text", "copy selected item", "copy that", "copy this text"),
    "cut": ("cut this", "cut selection", "cut selected text", "cut that"),
    "paste_here": ("paste here", "paste this", "paste at cursor", "paste at t l i", "paste at tap point", "paste at the cursor"),
    "paste": ("paste", "paste clipboard"),
    "copy_image_lasso": ("copy image", "copy picture", "start lasso", "start image lasso", "lasso image", "start image selection", "copy image region"),
    "start_text_selection": ("start text selection", "begin text selection", "start selecting text", "select text with hand", "begin lasso text"),
    "finish_selection": ("finish selection", "end selection", "stop selecting", "finish lasso"),
    "cancel_selection": ("cancel selection", "cancel lasso", "stop selection"),
    "select_all": ("select all", "select everything"),
    "undo": ("undo", "undo that"),
    "redo": ("redo", "redo that"),

    # Media keys
    "media_play_pause": ("play pause", "play or pause", "toggle playback", "pause music", "resume music", "play music", "play media", "pause media", "resume playback"),
    "media_next": ("next track", "next song", "skip track", "skip song", "next media"),
    "media_previous": ("previous track", "previous song", "last track", "previous media"),
    "volume_up": ("volume up", "turn volume up", "increase volume", "louder"),
    "volume_down": ("volume down", "turn volume down", "decrease volume", "quieter"),
    "volume_mute": ("mute", "mute audio", "mute volume", "silence audio"),
    "volume_unmute": ("unmute", "unmute audio", "restore volume"),

    # Window management
    "minimize_window": ("minimize window", "minimise window", "minimize this window", "minimise this window"),
    "maximize_window": ("maximize window", "maximise window", "maximize this window", "maximise this window"),
    "fullscreen_toggle": ("fullscreen", "full screen", "toggle fullscreen", "toggle full screen", "fullscreen toggle", "make this fullscreen", "make this full screen", "exit fullscreen", "exit full screen"),
    "restore_window": ("restore window", "restore this window", "unmaximize window", "unminimize window", "unminimise window"),
    "switch_window": ("switch window", "switch application", "next window", "next app", "alt tab"),
    "show_running_apps": ("show running apps", "show running applications", "show open apps", "show open applications", "show apps", "app switcher", "show app switcher", "task switcher", "show tasks", "show all windows", "windows overview"),
    "open_start_menu": ("open start menu", "show start menu", "start menu", "open windows start menu"),
    "show_desktop": ("show desktop", "minimize all windows", "minimise all windows"),
    "snap_left": ("snap window left", "move window left", "window left"),
    "snap_right": ("snap window right", "move window right", "window right"),

    # File manager / application navigation
    "open_file_explorer": ("open file explorer", "open explorer", "open files", "open file browser"),
    "open_downloads": ("open downloads", "open downloads folder"),
    "open_documents": ("open documents", "open documents folder", "open my documents", "show documents", "show my documents", "show my documents folder", "open the documents folder", "open my documents folder"),
    "open_desktop_folder": ("open desktop folder", "open desktop files"),
    "open_pictures": ("open pictures", "open pictures folder", "open photos folder"),
    "open_videos": ("open videos", "open videos folder"),
    "open_music_folder": ("open music folder", "open music files"),
    "open_notepad": ("open notepad", "launch notepad"),
    "open_calculator": ("open calculator", "launch calculator"),
    "new_folder": ("new folder", "create folder", "make folder"),
    "new_tab": ("new tab", "open new tab"),
    "close_tab": ("close tab", "close this tab"),
    "go_back": ("go back", "back in folder", "back page"),
    "go_forward": ("go forward", "forward page"),
    "go_up": ("go up", "parent folder", "up one folder"),
    "refresh_window": ("refresh", "refresh window", "reload page", "reload window"),
    "rename_selected": ("rename", "rename selected", "rename item"),
    "delete_selected": ("delete selected", "delete this", "delete item", "move to recycle bin"),
}

_LOOKUP = {alias: action for action, aliases in _ALIASES.items() for alias in aliases}


def normalize_transcript(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_pc_command(text: str) -> str | None:
    """Return the deterministic action ID for an exact supported voice phrase."""
    return _LOOKUP.get(normalize_transcript(text))


def aliases_for_help() -> dict[str, tuple[str, ...]]:
    return dict(_ALIASES)
