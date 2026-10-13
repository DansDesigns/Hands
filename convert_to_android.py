#!/usr/bin/env python3
"""Validate a Spatial Desktop ZIP and point to the included native Android project.

This helper deliberately does not pretend to transpile PySide6/Windows APIs to Android.
It can be run against a project ZIP to verify that its desktop files and Android scaffold
are present in a package produced from the current source.
"""
import argparse
import os
import sys
import zipfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zipfile", nargs="?", help="Spatial Desktop source ZIP to inspect")
    parser.add_argument("--android-dir", default="android_app", help="Path to the Android project")
    args = parser.parse_args()
    if args.zipfile:
        if not os.path.isfile(args.zipfile):
            print(f"ZIP not found: {args.zipfile}", file=sys.stderr)
            return 2
        try:
            with zipfile.ZipFile(args.zipfile) as z:
                names = z.namelist()
                py_count = sum(n.endswith(".py") for n in names)
                print(f"Source ZIP valid: {len(names)} entries, {py_count} Python files")
        except zipfile.BadZipFile:
            print("Input is not a valid ZIP archive", file=sys.stderr)
            return 2
    required = [
        "settings.gradle.kts", "build.gradle.kts",
        "app/build.gradle.kts", "app/src/main/AndroidManifest.xml",
        "app/src/main/java/com/spatialdesktop/android/MainActivity.kt",
    ]
    missing = [p for p in required if not os.path.isfile(os.path.join(args.android_dir, p))]
    if missing:
        print("Android project incomplete; missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    print(f"Android project scaffold found: {os.path.abspath(args.android_dir)}")
    print("Open that folder in Android Studio; add hand_landmarker.task before testing hand tracking.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
