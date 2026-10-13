"""Small, dependency-free version checker for Spatial Desktop."""
from __future__ import annotations

import re
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

REPOSITORY_URL = "https://github.com/DansDesigns/Hands"
REMOTE_VERSION_URL = "https://raw.githubusercontent.com/DansDesigns/Hands/main/version.txt"
SOURCE_ZIP_URL = "https://github.com/DansDesigns/Hands/archive/refs/heads/main.zip"
VERSION_FILE = Path(__file__).resolve().with_name("version.txt")


def read_version(path: str | Path = VERSION_FILE) -> str:
    """Read a plain version.txt value, falling back to 0.8 for an incomplete checkout."""
    try:
        value = Path(path).read_text(encoding="utf-8-sig").strip()
    except OSError:
        value = "0.8"
    value = value.removeprefix("v").strip()
    if not value:
        raise ValueError("version.txt is empty")
    return value


def _version_parts(version: str) -> tuple[int, ...]:
    """Convert dotted numeric versions (optionally prefixed with v) for comparison."""
    value = str(version).strip().removeprefix("v")
    if not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise ValueError(f"Invalid version string: {version!r}")
    return tuple(int(part) for part in value.split("."))


def compare_versions(left: str, right: str) -> int:
    """Return 1 if left is newer, -1 if older, or 0 if equal."""
    a, b = _version_parts(left), _version_parts(right)
    width = max(len(a), len(b))
    a += (0,) * (width-len(a))
    b += (0,) * (width-len(b))
    return (a > b) - (a < b)


def fetch_remote_version(timeout: float = 8.0) -> str:
    """Fetch and validate the repository's raw version.txt without blocking the GUI."""
    request = urllib.request.Request(
        REMOTE_VERSION_URL,
        headers={"User-Agent": "Spatial-Desktop-Updater/0.8", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(256).decode("utf-8-sig").strip()
    return raw.removeprefix("v").strip()


def download_update(timeout: float = 30.0, progress_callback=None) -> Path:
    """Download and validate the repository source archive, reporting byte progress when possible."""
    temp_root = Path(tempfile.mkdtemp(prefix="spatial-desktop-update-"))
    archive = temp_root / "update.zip"
    request = urllib.request.Request(
        SOURCE_ZIP_URL,
        headers={"User-Agent": "Spatial-Desktop-Updater/0.8", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if progress_callback is not None:
                    try:
                        progress_callback(downloaded, total)
                    except Exception:
                        pass
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            roots = {name.split("/", 1)[0] for name in names if "/" in name}
            if not names or not roots:
                raise ValueError("GitHub returned an empty or unexpected source archive")
            root = temp_root / next(iter(roots))
            # Reject path traversal before extracting the archive.
            for name in names:
                target = (temp_root / name).resolve()
                if temp_root.resolve() not in target.parents and target != temp_root.resolve():
                    raise ValueError("The update archive contains an unsafe path")
            zf.extractall(temp_root)
        source_root = root
        if not (source_root / "main.py").is_file() or not (source_root / "version.txt").is_file():
            raise ValueError("The downloaded archive is missing main.py or version.txt")
        remote_version = read_version(source_root / "version.txt")
        # Return a path to the validated extracted source tree; it remains in temp storage.
        return source_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def install_update(source_root: str | Path, destination: str | Path | None = None) -> Path:
    """Install source files in place while preserving settings, calibration and user data.

    Existing files are backed up under the system temporary directory for rollback.
    The running process should be restarted after this function completes.
    """
    source_root = Path(source_root).resolve()
    destination = Path(destination or Path(__file__).resolve().parent).resolve()
    if not (source_root / "main.py").is_file() or not (source_root / "version.txt").is_file():
        raise ValueError("The staged update is incomplete")
    protected_names = {
        "ui_settings.json", "voice_settings.json", "gestures.txt", "hand_calibration.json",
        "tli_calibration.json", "custom_gestures.json", "local_search_index.sqlite3",
    }
    protected_dirs = {"models", "face_models", ".venv", "venv", "__pycache__", ".git",
                      ".pytest_cache", "update_backups"}
    backup_root = Path(tempfile.gettempdir()) / "SpatialDesktopUpdateBackups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source_root.rglob("*"):
        relative = item.relative_to(source_root)
        if (any(part in protected_dirs for part in relative.parts) or item.name in protected_names
                or item.name.startswith("local_search_index.sqlite3")):
            continue
        if item.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
            continue
        if item.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_file():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        shutil.copy2(item, target)
        copied += 1
    if copied == 0 or not (destination / "main.py").is_file():
        raise RuntimeError("No application files were installed")
    return backup_root
