# Freehand — Spatial Computing

Freehand is an open-source, camera-based **hand interaction** application for Windows and Linux. It uses MediaPipe hand tracking to control native desktop input and render virtual hands, a Glass Menu, the Garden/Space virtual desktop and a drawing whiteboard/blackboard. **D1, Vosk, STT, TTS, facial recognition and assistant file search are not part of Freehand.** They belong to a separate AI assistant project.

## Free source and optional paid installers

The Freehand **application source** is licensed under **GPL-3.0-or-later** (see [`LICENSE`](LICENSE) and [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md)), subject to separate licences for third-party media and dependencies. The code is available publicly and can be built locally. The official prebundled Windows installer and Linux AppImage may be sold for convenience; recipients still have their GPL rights to obtain and redistribute the corresponding GPL-covered software. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Install from source

1. Download and extract the repository. **Use a Python release compatible with the available MediaPipe wheels** (the included installers support Python 3.10–3.12 on Windows; Linux also checks 3.13).
2. On Windows, run `install_windows.bat`, then `run_windows.bat`, or launch from the installed Start Menu shortcut.
3. On Linux, run `sh install_linux.sh` and then `bash run_linux.sh`, or use the application menu. For native two-hand touch injection, the installer configures restricted `/dev/uinput` access; a new login may be required for group membership.
4. Allow camera access. On first launch, Freehand downloads Google's MediaPipe hand-landmarker `.task` file if it isn't already cached in `models/`. For offline deployment, distribute a verified permitted copy of that model as part of the installer.
5. For optional 3D hand models, install `requirements-3d.txt`. The included WebXR hand meshes have separate MIT licence notices.

## Core features

- Two-hand gesture tracking; independent pinch, drag, scrolling, calibration, TPI indicators and native desktop control.
- Gesture-controlled Glass Menu, gesture map, per-hand settings and clipboard.
- Full-screen whiteboard/blackboard with palette, eraser and PNG export.
- Garden and Space virtual desktop and interactive onboarding imagery.
- Windows/Linux camera handling, automatic startup overlays, sounds and updater.

The settings and calibration are stored locally. The updater reads `version.txt` for the installed application version. Release updates require properly published corresponding installers for packaged builds.

## Custom 3D hands

Create hand models in glTF Binary (`.glb`) format. Put custom models in a local `models/hands/` folder, with `custom-left.glb` or `custom-right.glb` (one side can be mirrored). Bones can use common WebXR, Rigify or Mixamo names, or user-provided `.joints.json` mappings. The optional renderer needs ModernGL. Check licences before distributing custom models.

## Licences and attributions

- Freehand source: GPL-3.0-or-later — [`LICENSE`](LICENSE).
- Generic 3D hands: Amazon/WebXR Input Profiles, MIT — [`assets/hands/LICENSE-webxr-input-profiles.md`](assets/hands/LICENSE-webxr-input-profiles.md).
- Interface sound assets: sourced from Pixabay under their applicable Content License — [`assets/sounds/LICENSE-PIXABAY.md`](assets/sounds/LICENSE-PIXABAY.md).
- Python packages/native libraries/models: see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). Paid binary releases must also provide exact licence notices from the packaged build.

See [`changelog.txt`](changelog.txt) for release history and [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidance.
