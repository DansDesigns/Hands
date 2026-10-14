"""Lightweight GitHub version checker for StreamTTS.

The remote version.txt is intentionally not bundled with this application; the
repository owner supplies it at the root of the main or master branch.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY_URL = "https://github.com/dansdesigns/streamtts"
VERSION_URLS = (
    ("main", "https://raw.githubusercontent.com/dansdesigns/streamtts/main/version.txt"),
    ("master", "https://raw.githubusercontent.com/dansdesigns/streamtts/master/version.txt"),
)
DOWNLOAD_URL_TEMPLATE = "https://github.com/dansdesigns/streamtts/archive/refs/heads/{branch}.zip"
_VERSION_RE = re.compile(
    r"^\s*[vV]?(\d+(?:\.\d+){1,3})(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?\s*$"
)


class UpdateCheckError(RuntimeError):
    """Raised when no valid version.txt can be read from the repository."""


def parse_version(value: str) -> tuple:
    """Convert a numeric version string to a comparison tuple.

    Accepts e.g. 0.1.5, v0.1.5, 0.1.5-beta. Numeric components are padded,
    and a final release sorts after a prerelease with the same numeric version.
    """
    if not isinstance(value, str):
        raise ValueError("Version must be text")
    match = _VERSION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid version string: {value!r}")
    numbers = tuple(int(part) for part in match.group(1).split("."))
    numbers = numbers + (0,) * (4 - len(numbers))
    prerelease = match.group(2)
    # A stable release sorts above any prerelease with matching numbers.
    return numbers, (1, "") if prerelease is None else (0, prerelease.lower())


def read_current_version() -> str:
    """Read the version bundled with the app from ../VERSION."""
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        value = version_file.read_text(encoding="utf-8").strip().splitlines()[0]
        parse_version(value)
        return value.lstrip("vV")
    except (OSError, IndexError, ValueError):
        return "0.0.0"


def is_newer_version(latest: str, current: str) -> bool:
    return parse_version(latest) > parse_version(current)


def fetch_latest_version(timeout: float = 6.0) -> tuple[str, str]:
    """Return (version, branch) from remote version.txt.

    Tries main then master, allowing either branch layout without embedding a
    version.txt in the distribution. Network errors are reported to the GUI.
    """
    errors: list[str] = []
    for branch, url in VERSION_URLS:
        request = Request(url, headers={"User-Agent": "StreamTTS-Updater/1.0", "Cache-Control": "no-cache"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(512).decode("utf-8-sig", errors="strict")
            lines = [line.strip() for line in payload.splitlines() if line.strip()]
            if not lines:
                raise UpdateCheckError("version.txt is empty")
            version = lines[0]
            parse_version(version)
            return version.lstrip("vV"), branch
        except HTTPError as exc:
            errors.append(f"{branch}: HTTP {exc.code}")
            if exc.code != 404:
                break
        except (URLError, TimeoutError, UnicodeError, ValueError, UpdateCheckError) as exc:
            errors.append(f"{branch}: {exc}")
            # A malformed / unreachable version file is not normally helped by
            # trying the other branch, but a 404 above explicitly is.
            if not isinstance(exc, (HTTPError,)):
                break
    details = "; ".join(errors) if errors else "no response"
    raise UpdateCheckError(f"Unable to read a valid version.txt from GitHub ({details}).")
