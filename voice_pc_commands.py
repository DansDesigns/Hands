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
    "show_clipboard": ("open clipboard", "show clipboard", "open floating clipboard", "show floating clipboard"),
    "hide_clipboard": ("close clipboard", "hide clipboard", "close floating clipboard", "hide floating clipboard"),
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
    "open_my_computer": ("open my computer", "show my computer", "open this pc", "show this pc", "open computer", "go to my computer", "go to this pc"),
    "open_downloads": ("open downloads", "open downloads folder"),
    "open_documents": ("open documents", "open documents folder", "open my documents", "show documents", "show my documents", "show my documents folder", "open the documents folder", "open my documents folder"),
    "open_desktop_folder": ("open desktop folder", "open desktop files"),
    "open_pictures": ("open pictures", "open pictures folder", "open photos folder"),
    "open_videos": ("open videos", "open videos folder"),
    "open_music_folder": ("open music folder", "open music files"),
    "open_notepad": ("open notepad", "launch notepad"),
    "open_calculator": ("open calculator", "launch calculator"),
    # Creating files/folders must be explicit: a selection name like "new folder 2"
    # must never be misclassified as a request to create a folder.
    "new_folder": ("create folder", "create new folder", "create a folder", "create a new folder"),
    "create_new_text_file": ("create new text file", "create text file", "create a new text file"),
    "create_new_powerpoint": ("create new powerpoint", "create powerpoint", "create a new powerpoint",
                              "create new powerpoint presentation", "create powerpoint presentation",
                              "create a new powerpoint presentation"),
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


def _strip_polite_navigation_prefix(text: str) -> str:
    """Remove common spoken politeness without changing the requested item name."""
    result = (text or "").strip()
    prefixes = r"(?:(?:please|could you|can you|would you|will you|hey freehand|freehand)\s+)+"
    # Repeat because speech commonly produces variants such as "could you please select".
    while True:
        cleaned = re.sub(r"^" + prefixes, "", result, flags=re.IGNORECASE).strip()
        if cleaned == result:
            return result
        result = cleaned


def has_explicit_create_prefix(text: str) -> bool:
    """Creation intents must literally start with "create" after polite prefixes."""
    cleaned = _strip_polite_navigation_prefix(text)
    return bool(re.match(r"^create\b", cleaned, re.IGNORECASE))


def is_navigation_intent(text: str) -> bool:
    """Identify file/folder/creation requests that must not reach generative TTS."""
    raw = _strip_polite_navigation_prefix(text)
    cleaned = normalize_transcript(raw)
    if not cleaned:
        return False

    # Creation is always opt-in with the word "create". This guard also catches
    # malformed/partial create requests so they cannot fall through to the AI.
    if re.match(r"^create\b", cleaned) and re.search(
            r"\b(?:folder|directory|file|text|powerpoint|presentation)\b", cleaned):
        return True

    if not re.match(r"^(?:select|choose|open|go to|navigate to|find)\b", cleaned):
        return False
    if re.search(r"\b(?:folder|directory|file|files|drive|document|documents|item|items|all)\b", cleaned):
        return True

    # File extensions and literal filename separators are reliable navigation
    # signals even when the speaker omits punctuation such as '_' or '-'.
    if re.search(r"(?i)\.[a-z0-9]{1,12}(?:\s|$)", raw) or re.search(r"[_-]", raw):
        return True

    # Generic one/two-word names (e.g. "select ponder") should be resolved in
    # the active file manager, not sent to the generative assistant. Keep a few
    # text-selection phrases out of this classification.
    remainder = re.sub(r"^(?:select|choose|open|go to|navigate to|find)\s+", "", cleaned).strip()
    if cleaned.startswith(("select ", "choose ")) and remainder not in {
            "text", "text with hand", "all", "everything", "this", "that"}:
        return bool(remainder)
    return False


def parse_navigation_command(text: str) -> dict | None:
    """Parse parameterised file-manager commands while preserving item names verbatim."""
    raw = _strip_polite_navigation_prefix(text)
    clean = normalize_transcript(raw)
    if not clean:
        return None

    creator = re.fullmatch(
        r"create\s+(?:a\s+)?(?:(?:new)\s+)?(folder|directory|text\s+file|plain\s+text\s+file|"
        r"power\s*point(?:\s+presentation)?|powerpoint(?:\s+presentation)?|presentation)",
        clean, re.IGNORECASE)
    if creator:
        requested = re.sub(r"\s+", " ", creator.group(1).lower())
        kind = "folder" if requested in {"folder", "directory"} else (
            "text" if "text" in requested else "powerpoint")
        return {"action": "create_item", "kind": kind}

    drive = re.fullmatch(r"(?:go to|navigate to|open) ([a-z])(?: drive)?", clean)
    if drive:
        return {"action": "navigate_drive", "drive": drive.group(1).upper()}

    if clean in {"select all files", "select every file", "select all the files"}:
        return {"action": "select_all_kind", "kind": "file"}
    if clean in {"select all folders", "select all directories", "select every folder", "select all the folders"}:
        return {"action": "select_all_kind", "kind": "folder"}
    if clean in {"select all files and folders", "select all files and directories", "select everything in this folder"}:
        return {"action": "select_all_kind", "kind": "all"}

    verb_match = re.match(r"^\s*(select|choose|open|go to|navigate to)\s+(.+?)\s*$", raw, re.IGNORECASE)
    if not verb_match:
        return None
    verb = verb_match.group(1).lower()
    remainder = verb_match.group(2).strip()
    if not remainder:
        return None

    # Handle explicit spoken constructions like "select a folder called llama.cpp".
    # Keep punctuation in the captured item name; dots are common in folder names.
    named_kind = re.match(
        r"^(?:(?:the|a|an)\s+)?(folder|directory|file|document)\s+(?:called|named)\s+(.+?)\s*[.!?]?$",
        remainder, re.IGNORECASE)
    if named_kind:
        kind = "folder" if named_kind.group(1).lower() in {"folder", "directory"} else "file"
        name = named_kind.group(2).strip().strip("\"'“”‘’")
        if not name:
            return None
        if verb in {"go to", "navigate to"}:
            return {"action": "navigate_item", "name": name, "kind": "folder"}
        if verb == "open":
            return {"action": "open_item", "name": name, "kind": kind}
        return {"action": "select_items", "names": [name], "kind": kind}

    # Also allow "select the folder llama.cpp" / "select folder llama.cpp".
    leading_kind = re.match(
        r"^(?:(?:the|a|an)\s+)?(folder|directory|file|document)\s+(.+?)\s*[.!?]?$",
        remainder, re.IGNORECASE)
    if leading_kind and not re.match(r"^(?:called|named)\b", leading_kind.group(2), re.IGNORECASE):
        kind = "folder" if leading_kind.group(1).lower() in {"folder", "directory"} else "file"
        name_part = leading_kind.group(2).strip()
        if name_part:
            if verb in {"go to", "navigate to"}:
                return {"action": "navigate_item", "name": name_part.strip("\"'“”‘’"), "kind": "folder"}
            if verb == "open":
                return {"action": "open_item", "name": name_part.strip("\"'“”‘’"), "kind": kind}
            return {"action": "select_items", "names": [name_part.strip("\"'“”‘’")], "kind": kind}

    # Quoted names take priority, allowing multiple explicit names without splitting
    # punctuation (or the word "and" inside a file/folder name).
    quoted_names = re.findall(r"[\"'“‘](.+?)[\"'”’]", remainder)
    if verb in {"select", "choose"} and len(quoted_names) >= 2:
        names = [name.strip() for name in quoted_names if name.strip()]
    elif verb in {"select", "choose"} and re.search(r"\s+and\s+", remainder, re.IGNORECASE) and not quoted_names:
        names = [part.strip().strip("\"'“‘’") for part in re.split(r"\s+and\s+", remainder, flags=re.IGNORECASE) if part.strip()]
    elif len(quoted_names) == 1:
        names = [quoted_names[0].strip()]
    else:
        names = [re.sub(r"^(?:the|a|an)\s+", "", remainder.strip().strip("\"'“‘’"), flags=re.IGNORECASE)]

    names = [name.strip() for name in names if name and name.strip()]
    if not names:
        return None

    kind_hint = None
    trailing_kind = re.search(r"\b(folder|directory|file|document)$", remainder, flags=re.IGNORECASE)
    if trailing_kind:
        suffix_kind = trailing_kind.group(1).lower()
        kind_hint = "folder" if suffix_kind in {"folder", "directory"} else "file"

    if not quoted_names:
        names = [re.sub(r"\s+(?:folder|directory|file|document)$", "", name, flags=re.IGNORECASE).strip()
                 for name in names]
        names = [re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE).strip() for name in names]

    if verb in {"select", "choose"}:
        return {"action": "select_items", "names": names, "kind": kind_hint}
    if len(names) != 1:
        return None
    action = "navigate_item" if verb in {"go to", "navigate to"} else "open_item"
    return {"action": action, "name": names[0],
            "kind": kind_hint or ("folder" if action == "navigate_item" else None)}
