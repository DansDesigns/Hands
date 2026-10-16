"""Per-user storage paths for settings that must survive application updates."""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path


def user_data_dir() -> Path:
    """Return a writable, per-user application data directory on each platform."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / "Freehand"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Freehand"
    base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "freehand"


def user_file(name: str, legacy_dir: Path | None = None) -> Path:
    """Resolve a user-owned file and migrate the old in-install copy once."""
    directory = user_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if not destination.exists() and legacy_dir is not None:
        legacy = Path(legacy_dir) / name
        if legacy.is_file():
            try:
                shutil.copy2(legacy, destination)
            except OSError as exc:
                print(f"Could not migrate {name} to user settings directory: {exc}")
    return destination
