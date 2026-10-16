# Freehand — Spatial Desktop

## Short description

Freehand is a camera-based spatial desktop that lets you control your computer with hand gestures and voice. It includes a virtual desktop, optional Garden Mode, hand visualisation, local speech recognition and speech output, and local file search. Most interaction and speech processing runs locally.

## How to install

1. Download and extract the project ZIP.
2. Install Python 3 and the dependencies for your platform.
3. On Windows, run `run_windows.bat`. On Linux, run `run_linux.sh` (or install `requirements.txt` and run `python main.py`).
4. Allow camera access and place your hands where the camera can see them.
5. Optional: install `requirements-3d.txt` for rigged 3D hand rendering, and follow `VOICE_SETUP.md` for offline speech output/voice setup.

The app stores settings and calibration locally. Use **Settings → Update** to check for an available project update. The root `version.txt` is the version source used by the updater.

## Features, speech commands & gestures

### Features

- Camera-based two-hand tracking with Skeleton, Flat and rigged 3D hand styles.
- Tap Point Indicator (TPI), per-hand calibration, pinch menu, pointer interaction, drag-and-drop, scrolling and window manipulation.
- Virtual desktop with persistent drawing, plus Garden Mode with animated scenery, hand-reactive grass, birds, a pick-up-and-move ball, a movable sundial, day/night lighting and weather when network access is available.
- Local Vosk speech recognition, optional offline StreamTTS speech, dictation, local file search, clipboard tools and optional face welcome.
- Customisable gestures and interaction settings.

### Gestures

- **Quick thumb/index pinch and release** or **quick fist/grip and open**: click at the TPI.
- **Hold a thumb/index pinch**: open the Pinch Menu after its confirmation delay; move the TPI to an option and make a fresh pinch/grip to select it.
- **Move while pinching**: drag-and-drop; release to drop.
- **Right-hand fist with movement**: scroll. A left-hand fist can act as a modifier for right-click and window/canvas dragging.
- **Peace sign held for about 1.5 seconds**: detach or reattach that hand from native pointer movement.
- **Garden props**: pinch a ball or sundial, move it, then release.

Gestures can vary with settings and platform. Open the app's **Gesture Map** and **Speech Commands** pages for the current in-app reference.

### Speech command examples

Enable voice input, then try phrases such as:

- `search bar` / `search files` — open local search.
- `pause tracking` / `resume tracking` — pause or resume hand input.
- `begin dictation` / `end dictation` — dictate into the focused field.
- `close window` / `open user folder` — common desktop actions.
- `start text selection`, then `copy this` or `finish selection` — select and copy text.
- `copy image` / `start image lasso`, then `copy this` — copy a screen region as an image.

The app's **Speech Commands** page lists additional commands and requirements. Exact availability depends on platform and enabled components.

## Attributes & inspiration

Freehand is inspired by spatial-computing interfaces, gesture-driven interaction, desktop launchers, and natural hand interaction. It uses MediaPipe hand landmarks and local Vosk speech recognition, with optional StreamTTS and OpenGL-based 3D rendering.

Bundled hand models are from the [WebXR Input Profiles](https://github.com/immersive-web/webxr-input-profiles) project; see `assets/hands/LICENSE-webxr-input-profiles.md`. Third-party libraries and assets retain their own licences. No project-wide licence has been selected in this repository, so check with the maintainer before redistributing or commercially using the project.

Release notes are in [`changelog.txt`](changelog.txt).

## How to use your own hand 3D models

1. Export your hand model from Blender or another 3D tool as **glTF Binary (`.glb`)**. FBX, OBJ and BLEND files should be exported to glTF first.
2. Create the ignored local folder `models/hands/` in the project root.
3. Add `custom-right.glb` and/or `custom-left.glb` to that folder. If you supply only one side, Freehand mirrors it for the other.
4. Use a rigged hand with bones for the wrist and finger segments. Common WebXR, Blender/Rigify (for example `f_index.01.R`) and Mixamo (for example `RightHandIndex1`) bone naming conventions are recognised.
5. For a custom rig whose bone names are not recognised, add `custom-right.joints.json` (and/or the matching left file) mapping each bone name to `wrist`, `<finger>-meta`, or `<finger>-1` through `<finger>-4`. Finger names are `thumb`, `index`, `middle`, `ring` and `pinky`; `1` is the knuckle-side segment and `4` the tip.
6. Enable **Settings → Hands → RENDER → 3D**. Install the optional dependencies in `requirements-3d.txt` if needed. If 3D rendering is unavailable, the app falls back to Flat mode.

The `models/` directory is git-ignored so personal/licensed models are not accidentally added to source control. Check the model licence before using or sharing it; some licences allow use but prohibit redistribution.
