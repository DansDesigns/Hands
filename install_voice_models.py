#!/usr/bin/env python3
"""Download the default StreamTTS UK voice model for offline speech output."""
from __future__ import annotations
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
VOICE = "vits-piper-en_GB-alan-medium"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{VOICE}.tar.bz2"


def main():
    target = MODELS / VOICE
    if target.is_dir() and list(target.glob("*.onnx")) and (target / "tokens.txt").is_file():
        print(f"StreamTTS voice already installed: {target}")
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    archive = MODELS / f"{VOICE}.tar.bz2"
    temporary = archive.with_suffix(archive.suffix + ".part")
    try:
        print(f"Downloading StreamTTS voice model: {VOICE}")
        with urlopen(URL, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(archive)
        base = MODELS.resolve()
        with tarfile.open(archive, "r:bz2") as bundle:
            for member in bundle.getmembers():
                destination = (MODELS / member.name).resolve()
                if destination != base and base not in destination.parents:
                    raise RuntimeError(f"Unsafe path in model archive: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Refusing link in model archive: {member.name}")
            bundle.extractall(MODELS)
    finally:
        temporary.unlink(missing_ok=True)
    if not (target / "tokens.txt").is_file() or not list(target.glob("*.onnx")):
        raise RuntimeError(f"Voice download did not create the expected model directory: {target}")
    print(f"StreamTTS voice installed: {target}")


if __name__ == "__main__":
    main()
