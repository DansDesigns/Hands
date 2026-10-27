# Freehand — hand-only repository verification and update (27 October 2026)

Audited input: `Hands_current.zip` from the user, including its **current** `main.py`.

## Verified present and preserved

- Hand-tracking and gesture code: `gesture_rules.py`, `custom_gestures.py`, `hand_occlusion.py`, `hand_contact.py`, `hand_calibration.py`, `tli_calibration.py`, `tpi_screen_calibration.py`.
- UI and desktop: `main.py`, `garden_weather.py`, `windows_virtual_desktop.py`, `linux_screen_geometry.py`, `hand_mesh_3d.py`.
- Configuration and updater: `updater.py`, `user_paths.py`, `version.txt`, `ui_settings.json`.
- Both WebXR generic-hand `.glb` files and original MIT notice.
- All nine onboarding pictures, both gesture-map images, nine sound MP3s, and the original full-resolution splash logo, PNG and ICO icons.
- Existing Windows VBS launcher, Windows/Linux launch scripts and optional 3D requirements.
- Main.py and all media, images, 3D hands, other source modules and unchanged launchers are **byte-for-byte identical** to the uploaded ZIP.

`models/hand_landmarker.task` **is not in this ZIP**, but current Freehand downloads it using Google's MediaPipe URL on first use. A truly offline paid installer should include a verified licensed copy and its notices.

## Changed

- `install_windows.bat`: remove Search/Indexing module deployment, Mutagen indexing, TTS/Vosk, face models and D1 downloads. Retain optional ModernGL, Start Menu/launcher handling and Python environment setup.
- `install_linux.sh`: remove voice/face/D1 setup; retain restricted `/dev/uinput` permissions and desktop launcher setup, plus optional ModernGL.
- `requirements.txt`: hand-only runtime; removes Vosk, sounddevice, document-indexing packages. Kept `opencv-contrib-python` unchanged for compatibility pending full platform build tests.
- `LICENSE`: replace old source-available-only terms with **full GNU GPLv3** text; `LICENSE_SCOPE.md` declares **GPL-3.0-or-later** for project-owned source and excludes separately licensed third-party media.
- Update `README.md`, `CONTRIBUTING.md` and CLA references to the new hand-only GPL/free-source/paid-prebuilt model.
- `THIRD_PARTY_LICENSES.md`, `assets/sounds/LICENSE-PIXABAY.md`: document licensing and the remaining binary-licence/provenance checklist.
- `install_desktop_launcher.sh`: update obsolete Spatial Desktop Prototype description to Freehand.

## Archived separately, not shipped in cleaned Freehand

- `search_launcher.py`
- `freehand_search_index.py`
- `apply_freehand_search_settings.py`

These were unreferenced by the supplied hand-only main.py and are backed up in `Freehand_legacy_search_modules_backup.zip` for reuse in the standalone assistant.

## Still requiring manual confirmation before publishing paid installers

1. **Licence choice:** This draft uses **GPL-3.0-or-later**, following the earlier recommendation. The actual rights holder must approve/commit that licence change; check any third-party source contributions before relicensing. GPL allows paid installers, but buyers can redistribute GPL-covered copies.
2. **Pixabay MP3 provenance:** the repo's `Attributes.txt` only says all sounds are from Pixabay. Record URLs, creators and licensing evidence for each sound; avoid standalone redistribution.
3. **Bundled Qt/OpenCV/native wheels, MediaPipe model and other assets:** generate a complete SBOM/notice bundle from each finished Windows and Linux build. The source requirements list is not enough.
4. **Distribution Builder:** ensure it bundles `LICENSE`, `LICENSE_SCOPE.md`, third-party licence/asset notices, original splash logo and verified MediaPipe model (if promising offline first launch).
5. **Testing:** shell syntax, Python syntax, resource hashes, Windows batch labels and static module imports were validated here. Running the actual Windows installer or Linux AppImage requires testing on target machines.
6. **Legacy references:** the unchanged `main.py` still contains some obsolete voice-related comments/constants. They were intentionally not changed in this task; current local imports do not include the removed assistant modules.
