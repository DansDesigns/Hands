# Third-party dependencies and asset notices — Freehand

Freehand source is GPL-3.0-or-later; external components retain their own terms. This document provides a **source-tree inventory**, not a complete final binary SBOM or substitute for licences to be distributed with compiled releases.

| Component | Typical upstream licence | Commercial distribution / action |
| --- | --- | --- |
| Python | PSF License | Permits paid distribution; retain notices for bundled runtimes |
| PySide6 / Qt 6 (including QtMultimedia) | LGPLv3 / GPLv3 / commercial, by component | Allowed subject to the selected licences. Check exact Qt modules, provide licence texts, replacement/relink rights and required notices; GPL-covered Freehand source does not remove Qt obligations |
| MediaPipe | Apache-2.0 (library) | Keep notices; check **exact hand_landmarker.task artifact** licence separately |
| OpenCV (`opencv-contrib-python`) | Apache-2.0 for current OpenCV, with separately licensed bundled native parts | Inspect distributed wheel, including possible FFmpeg/Qt components, and retain all required notices |
| NumPy | BSD-3-Clause | Retain copyright and licence text |
| PyAutoGUI | BSD-3-Clause | Retain copyright and licence text; inspect transitives |
| ModernGL (optional) | MIT | Retain notice if bundled |
| WebXR Input Profiles generic hand assets | MIT | Existing attribution and complete MIT licence: `assets/hands/LICENSE-webxr-input-profiles.md` |
| Pixabay UI MP3 assets | Pixabay Content License (source note) | May be integrated into the app; do not independently sell/distribute sound pack. Record each original download and check applicable terms; see `assets/sounds/LICENSE-PIXABAY.md` |
| GNU/Linux virtual-input and camera libraries, where separately installed | Varies by package | Use the distribution's packages where possible; inspect dynamically linked and packaged files |

Additional packaging tools have their own licences: PyInstaller's GPL/bootloader exception, Inno Setup's own licence, AppImage tooling, Windows SDK signing utilities and platform drivers. These are **not automatically Freehand dependencies** just because the Release Builder uses them.

## Checklist for paid ready-to-install builds

1. Publish/offer the **complete corresponding GPL-covered source**, build recipes and installer scripts corresponding to the binary release; reference the matching release tag.
2. Include `LICENSE`, `LICENSE_SCOPE.md`, this file and `assets/hands/LICENSE-webxr-input-profiles.md` with the installer/AppImage.
3. Inventory **actual packaged** Python wheels, Qt shared libraries, native DLLs and model weights; include applicable copyright notices, full third-party licences, and any required attribution/source/relink information.
4. For Pixabay MP3s, preserve source page URLs, creator and download dates. Their terms are not GPL; do not offer the MP3s as a standalone sound collection.
5. Confirm rights to all artwork, icons, logos, and optional custom hand models before publishing.
6. If distributing the downloaded MediaPipe hand model, verify licence/provenance for that **specific binary model**, and include the necessary notices.
7. Check licences and signatures of the *finished* Windows `.exe` and Linux `.AppImage`, not just this repository.

Sources: https://doc.qt.io/qt-6/licensing.html , https://pixabay.com/service/terms/ , https://www.gnu.org/licenses/gpl-3.0.html .
