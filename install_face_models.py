#!/usr/bin/env python3
"""Install optional local Face Welcome models into the per-user cache.

The app also runs this download flow automatically when Face Welcome is enrolled.
This script remains as a convenient manual retry from the installer/setup guide.
"""
from __future__ import annotations
import sys

from face_welcome import FACE_MODELS, download_face_models


def main():
    print("Preparing optional local face-recognition models...")
    print(f"Cache location: {FACE_MODELS}")
    detector, recognizer = download_face_models(FACE_MODELS, print)
    print("Face models installed successfully.")
    print(f"  Detector:   {detector}")
    print(f"  Recognizer: {recognizer}")
    print("Recognition remains disabled until a user explicitly enrolls in Settings.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Face model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
