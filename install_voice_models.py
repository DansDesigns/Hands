#!/usr/bin/env python3
"""Download the optional Kokoro voice model files once for offline use."""
from __future__ import annotations
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "tts"
FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}

class Progress:
    def __init__(self, filename): self.filename = filename
    def __call__(self, blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r{self.filename}: {pct:3d}%", end="", flush=True)

def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        target = TARGET / filename
        if target.exists() and target.stat().st_size > 1024 * 1024:
            print(f"Already downloaded: {target}")
            continue
        temp = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {filename} for local TTS…")
        try:
            urllib.request.urlretrieve(url, temp, Progress(filename))
            print()
            if temp.stat().st_size < 1024 * 1024:
                raise RuntimeError("Download was unexpectedly small; check the URL/network.")
            os.replace(temp, target)
            print(f"Saved: {target}")
        except Exception:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
            raise
    print("Kokoro model files installed. They can now be used offline.")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
