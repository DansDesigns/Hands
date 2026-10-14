# Spatial Desktop v0.8

**Release version: 0.8.** Spatial Desktop provides camera-based hand tracking, gesture-controlled desktop interaction, per-hand TLI calibration, local Vosk speech recognition, optional offline StreamTTS speech output, local file search, face welcome, and an optional local d1 decision runtime. The sections below document the interaction and calibration milestones carried forward into this release; the root `version.txt` is the single version source used by the Settings → Update checker.

## Version checks and updates

The installed app version is recorded in the root `version.txt` file. Open **Settings → Update** and choose **Check for Updates** to compare that file with the repository copy at the raw `main/version.txt`. If a newer version is found, choose **Download Update**, then **Install Update**. The updater stages and validates the source archive, backs up replaced files, preserves settings and calibration, and asks you to restart Spatial Desktop after installation.

When publishing a release, update `version.txt` in the repository to the new dotted numeric version (for example `0.8.1`). Keep the version file at the repository root on the `main` branch so the in-app checker can find it.


## Hand render styles: Skeleton, Flat and 3D

Settings → Hands → **RENDER** is a three-way switch:

- **Skeleton** draws the tracked joints as lines.
- **Flat** is the painted hand: depth-ordered palm and tapered, shaded fingers that stay readable at any angle.
- **3D** poses a rigged 3D hand mesh from the 21 MediaPipe landmarks and renders it with OpenGL (`moderngl`, installed by the optional `requirements-3d.txt` step). If OpenGL or `moderngl` is unavailable, 3D falls back to Flat and the reason is shown once in the status/console.

The bundled 3D hands are `assets/hands/generic-hand-left.glb` and `generic-hand-right.glb` from the [WebXR Input Profiles](https://github.com/immersive-web/webxr-input-profiles) project, MIT licensed (see `assets/hands/LICENSE-webxr-input-profiles.md`). Hands are shown from your own viewpoint, not the camera's: with your palms toward the camera you see the backs of the virtual hands, just as you see your real ones.

**Touching hands are drawn touching.** Each hand is tracked on its own, and the drawn fingers are never exactly as thick as yours. So when your thumbs or fingertips touch, the drawn ones can stop a few pixels apart.
- When a fingertip of one hand comes within about 0.15 palm lengths (surface to surface) of a fingertip of the other, the ends of those two fingers are pulled together until they meet.
- The pull fades out smoothly by 0.35 palm lengths, so hands further apart aren't affected.
- This applies to the flat and 3D styles at any hand scale. Like tucking, it only changes the drawing: the TLI and gestures are untouched.

**Hidden fingertips are drawn tucked in.** The tracker always reports every fingertip, even ones the camera can't see; for those it guesses, often a half-open finger. A fingertip or thumb tip counts as hidden when it's farther from the camera than the palm or another finger in front of it.
- With the flat and 3D styles, hidden digits are drawn folded into a fist, and a hidden thumb rests over the curled fingers. This is the common case of a fist or pointing hand seen from the back. A fingertip outside the camera view is cut off rather than tucked, so it keeps its tracked position. Touching fingertips (a pinch) are never tucked.
- With the palm turned away from the camera, any finger that's folded is treated as hidden behind the hand. Once one finger is hidden, the other folded fingers are tucked with it, so a fist is drawn as a fist instead of each finger following the tracker's guess.
- The thumb only counts as folded when its tip lies across the palm, over the fingers, not just near it. A thumb out to the side after a pinch is never tucked.
- Which side of the palm the fingers fold to comes from the hand's own handedness, not from the guessed tips, so tucked fingers can't flip to the back of the hand. As a safety net, if two or more clearly folded fingers all bend the other way (a hand whose left/right label is briefly wrong), the fingers win.
- A tucked digit stays tucked until it's clearly out again: it has to be seen out for 4 camera frames in a row (about 130 ms), and it gets a looser reach limit while tucked. Without this, the tracker's wandering guesses made the drawn fingers flap.
- Digits fade in and out of the tucked pose over about 80 ms.
- This only changes the drawing. Gestures, the TLI and the cursor still use the tracked landmarks, and the skeleton style still shows the raw tracking.

**Using your own hand model:** save a rigged glTF/GLB hand as `models/hands/custom-right.glb` and/or `custom-left.glb` (`models/` is git-ignored, so licensed downloads stay out of the repository). If only one side is provided, it is mirrored for the other hand. Bone names from WebXR, Blender/Rigify (`f_index.01.R`) and Mixamo (`RightHandIndex1`) rigs are recognised automatically; for other rigs add `custom-right.joints.json` mapping each bone name to `wrist`, `<finger>-meta` or `<finger>-1`…`<finger>-4` (finger = thumb/index/middle/ring/pinky, 1 = knuckle bone, 4 = tip). FBX/OBJ/BLEND models need exporting to glTF from Blender first. Check each model's licence before sharing it: many marketplace "royalty free" licences allow use in a product but not redistributing the model file.

## v61 overlay and menu layout updates

- On Windows, the active hand-visual overlay periodically reasserts topmost z-order without activating or stealing keyboard focus. This helps keep hands/TLI visible above ordinary apps, Task Manager, and windowed/borderless video players. Exclusive-fullscreen applications can still suppress ordinary desktop overlays.
- The main glass menu Close banner is at the bottom of the panel. The left/right TLI calibration buttons have a wider gap so they are easier to distinguish.
- Interaction Settings allow `scroll_acceleration` up to `10.0×`.

## v55 Pinch drag-and-drop

- A thumb/index pinch that stays stationary continues through the fixed 0.8-second confirmation buffer and circular timer, then opens the Pinch Menu as before.
- In OS Desktop mode, if the held pinch moves at least 22 screen pixels before the timer completes, Spatial Desktop presses and holds the native left mouse button at the TLI. The pointer continues following the hand, allowing ordinary drag-and-drop in file managers, PowerPoint, image editors, 3D Builder and other applications that accept native mouse dragging.
- Releasing the pinch drops the item and releases the mouse button. The drag gesture is consumed, so it cannot also fire a quick click or open the Pinch Menu. If tracking is lost, hand tracking is paused, or the application closes, the held button is released safely.
- This is application-agnostic mouse input; it does not require custom integration with each application. Use OS Desktop mode for dragging objects in external applications.

## v54 changes

- Calibration target centre is transparent so the live hand can be seen behind it; requesting calibration re-enables the selected hand if peace-sign tracking was paused.
- Palm width defaults to 1.26×. Settings tabs have additional spacing and bottom-centred Back buttons. The Pinch Menu has extra padding below its last action.
- Skeleton mode includes an adjustable line-thickness slider. Interaction settings expose TLI glow intensity, hue/colour, border thickness, and diameter. Glass and Pinch Menu tints are 15% transparent.

This release carries forward the interaction baseline from Spatial Desktop v40 and later stability patches.

Spatial Desktop uses MediaPipe hand landmarks with a large, vibrant Tap Location Indicator (TLI). Before calibration, the TLI is centred midway between the thumb tip and index fingertip. After calibration, it uses the registered pinch point in a palm-relative 3D coordinate frame, so the TLI follows the hand as it translates, rotates, and changes apparent size. The displayed TLI and activation/click point share the same transformed, smoothed coordinates.


## TLI calibration (v52)

1. Open **Settings → Interaction** and select **CALIBRATE LEFT TLI** or **CALIBRATE RIGHT TLI**. Calibrate each physical hand separately.
2. Release the pinch/grab used to select the button. The button changes to **PINCH NOW** when ready.
3. Bring thumb and index fingertip together, with the other fingers extended enough for pinch recognition, and hold the pinch steady for roughly half a second. Twelve consecutive frames are averaged to reduce tracking jitter.
4. The button shows **CALIBRATED** when the new palm-relative point has been saved.

Calibration stores the thumb/index midpoint relative to the wrist-to-middle-MCP and across-palm axes, normalized by palm length and width. During use, that local point is reconstructed from the current hand landmarks, then passed through the same hand-scale transform and smoothing used by the visible TLI and actual activation point. The calibration is saved in `tli_calibration.json` beside `main.py`, separately for the left and right hands. If no valid calibration exists, the app falls back to the live thumb/index midpoint. Recalibrate after changing camera position or if you want a different natural pinch posture. This is hand-relative calibration, not camera-depth activation.

## Hand-interaction baseline carried forward from v25

- **Close the Pinch Menu outside:** after releasing the pinch that opened the Pinch Menu, make a fresh fist/grip or pinch while the TLI is outside the popup bounds. The popup closes, and that gesture is consumed so it cannot click or drag through to the desktop. Gestures inside the popup continue to select options.
- **Movement-driven right-fist scrolling:** a RIGHT-only fist starts smooth vertical scrolling as soon as deliberate movement is detected over page content; there is no fist timer circle or 0.8-second wait. The LEFT fist remains the modifier, and two-fist window drag stays separate.
- **Fist timer removed:** quick fist/open still clicks, but right-fist movement—not a completed countdown—starts scrolling. Opening the hand remains stable for 0.20 seconds to release active drag/scroll. Timed pinches require a continuous 0.8-second confirmation before the ring fills; a release before confirmation cancels the sequence.

## v43 UI and rendering fixes

- The main glass window and the topmost Search hand overlay no longer paint the same hand/TLI/timer layer simultaneously; Search overlay becomes the sole owner of that layer while visible. The overlay also avoids fabricating a fallback TLI when no cursor position is available.
- The home glass tile is now **Search** and opens the same local search launcher. Its magnifier icon is distinct from the **Speech Commands** list icon and the live Voice/STT audio-wave icon.
- Home tiles and floating Pinch Menu labels use bounded font fitting and line wrapping so longer labels remain inside their buttons.
- The fixed 0.8-second gesture confirmation is enforced as a continuous hold, not a wall-clock timer that keeps running between separate quick pinches.

## Gestures

- **Quick fist/grip → open:** left-click at the TLI location captured when the fist began. In Right-click Mode, this becomes a right-click.
- **Quick thumb/index pinch → release:** left-click at the TLI location captured when the pinch began.
  - A pinch that releases into folded fingers is not a click. Closing your hand into a thumbs up, thumbs down or fist sweeps the index finger past the thumb, which briefly reads as a pinch.
  - When the released hand is neither clearly open nor closed, the click waits up to 0.3 s to see which way it goes. A release into an open hand still clicks straight away.
  - A pinch must also last 0.15 s before moving it can start a drag, so that brief pinch can't press the mouse button either.
- **Timed thumb/index pinch:** after a 0.8-second confirmation delay, the ring fills for `menu_pinch_hold_seconds` (0.5–5.0 seconds; default 0.6). This opens the Pinch Menu in normal mode, or exits Right-click/Scroll Window mode. If the Pinch Menu is already open and selection-ready, a fresh pinch/grip outside the popup closes it.
- **Right fist scrolling:** a RIGHT-only fist begins smooth vertical scrolling when deliberate movement is detected over page content. No fist countdown or hold timer is needed; the existing smoothing and spike clamp keep movement gentle. The LEFT fist remains the modifier, and the separate two-fist window drag is unchanged.
- **Left-fist modifier:** hold the LEFT hand in a fist. A quick RIGHT-hand pinch and release sends a right-click at the pinch start point. A held RIGHT-hand grip starts dragging the native window in OS mode or the virtual canvas/application surface in 3D mode; open the right hand to release. Modifier gestures do not accumulate the Pinch Menu timer.
- **Peace-sign toggle (mouse detach):** hold an index-and-middle-finger V-sign for 1.5 seconds to detach or reattach that hand from native OS mouse movement. The TLI keeps updating and the hand can still interact with glass controls and Pinch Menus. A circular countdown appears around its TLI; the detached notice persists until reattached. No toggle TTS is queued.
- **Pinch Menu:** release the opening pinch, move the TLI to `RIGHT-CLICK MENU` or `GLASS MENU` (Dictate and STT are side by side), then use a fresh grab or pinch for about 0.1 seconds. The selection gesture is consumed. Right-click and Scroll Window modes are mutually exclusive; Glass Menu only changes visibility.
- **Settings sliders:** a fist/grip near a slider snaps the TLI to the current knob and starts dragging immediately, without a gesture timer. Initial pickup preserves the value.

## Camera mapping and settings

The camera's central X range `0.20–0.80` and Y range `0.16–0.84` map to the full display. Positions outside that region clamp to screen edges while the camera may continue tracking the hand in the outer overscan area. Settings slider travel is reduced to `0.42x` cursor travel for finer adjustment.

The tracker recovers the camera by itself after the PC sleeps, or if the driver stops sending frames. It releases the camera and opens it again when any of these happen:
- the tracking loop was paused for more than 3 seconds (sleep/resume)
- frames have failed for 2 seconds
- the same picture has been repeated for 2 seconds

Retries back off up to 30 seconds. If the camera doesn't come back under its usual index after repeated tries, the next few indexes are tried as well. Hands reappear within a second or two of the camera returning, with no restart needed.

`gestures.txt` uses one `setting_name=value` per line; blank lines and lines beginning with `#` are ignored. Supported settings:

- `hand_visual_scale`: `0.3–1.0`
- `finger_thickness`: `0.8–10.0` (default `8.0`)
- `hand_tracking_sensitivity`: `0.5–1.5`
- `cursor_smoothing`: `0.0–0.95`
- `click_debounce_seconds`: `0.04–0.25`
- `menu_pinch_hold_seconds`: `0.5–5.0` after the 0.8-second confirmation delay; default `0.6`
- `scroll_acceleration`: `0.5–10.0×`; controls how quickly drag scrolling accelerates with distance from the grab point; default `1.0×`
- `cursor_lock_priority`: `right` or `left`; controls which hand owns idle cursor tracking when both are visible
- `left_click_color` and `right_click_color`: `#RRGGBB`

The RIGHT and LEFT colour swatches are positioned on their respective sides without changing their existing colour assignments or hand functionality. Scroll direction can be inverted from Interaction Settings.

## Run

On Windows, run `run_windows.bat`. On Linux, use `run_linux.sh` or install `requirements.txt` and run `python main.py`. Camera access is required. `F11` toggles fullscreen and `Esc` exits.

## Licence and contributions

A project licence has not yet been selected. Check the repository for the licence added by the maintainer before reusing, redistributing, or commercially using this software. Contributions are welcome through pull requests; contributors should only submit work they have the right to contribute. Review third-party licences for dependencies and models separately.


## Offline voice and fast local search (v29)

- **Vosk STT retained:** audio runs in short bounded blocks. Partial transcripts feed the search box word-by-word as they become available; final recognition still handles voice commands.
- **Ulauncher-style local search:** say `search bar` or `search files` while voice input is enabled, press `Ctrl+Space`, or click the Search tile. Results update as you type. Press **Dictate** or launch it by voice to stream Vosk partial transcripts into the query field. Use **Add Folder** to include locations; use Up/Down and Enter to select/open. The **Speak** button uses StreamTTS to read the selected filename and excerpt.
- **No model needed for search:** Python plus SQLite FTS5 indexes file names, paths and bounded text excerpts. Ranking combines BM25, file-name/path weighting, exact phrase boosts, fuzzy filename matching and a small transparent synonym table. It runs locally and does not download embeddings or use cloud services. The initial index is built in a background thread and later updates changed files incrementally.
- **Indexed file types:** file names for all types; contents for common text/source files, PDF and DOCX. The installer includes the small PDF/Word parsing dependencies. Common personal folders are indexed by default; use Add Folder to choose others. The index is stored under `~/.spatial_desktop`.
- **Optional local TTS (StreamTTS):** the Windows/Linux installer installs `requirements-voice.txt` and attempts to download StreamTTS ONNX voice files via `install_voice_models.py`. If network/download fails, the installer warns and the core app/search still works. British voice `vits-piper-en_GB-alan-medium` is the default; see `VOICE_SETUP.md`.
- **Why no EmbeddingGemma/OpenJev yet:** true embeddings can match descriptions with little word overlap, but require a large model/runtime. OpenJev is a local decision-model wrapper rather than a file retrieval index, and the JEV Search project's optional reranking uses a remote API. For this phase we prefer responsive, transparent, local ranking. If testing shows synonym/BM25 search is insufficient, a small embedding model can be evaluated later.

## Start and shortcuts

- Windows: run `install_windows.bat` once, then the Start Menu shortcut or `run_windows.bat`.
- Linux: run `./install_linux.sh` once, then the app launcher or `./run_linux.sh`.
- **Ctrl+Space** opens/closes local search; say **“search bar”** while Vosk input is enabled to open it directly and begin live dictation.



## Compact search, voice commands, and startup greeting (v29)

- **Search window close safety:** the search launcher is a child dialog and closing its title-bar X or its own × button only closes that launcher. It must not exit Spatial Desktop. The application explicitly disables Qt's quit-on-last-window-closed behaviour for the launcher and quits only when the main overlay itself closes.
- **Compact launcher:** search opens as a short bar with a dropdown arrow. The arrow reveals results and utility actions; typing or live Vosk dictation expands results automatically unless the user deliberately collapses the details.
- **Voice commands:** `close window` requests WM_CLOSE for the last focused external Windows window. When Search is open, it targets the external window that was active before Search took focus. `open user folder` opens the user's home directory in Explorer/Finder/the default file manager. The commands are handled before search dictation consumes recognised text.
- **Startup speech:** StreamTTS speaks a time-of-day greeting after launch (morning/afternoon/evening/night), using the enrolled display name if available and the operating-system username otherwise. The greeting is skipped if local TTS is not ready.

## Optional local face welcome

- Open Settings and select **ENROL FACE FOR WELCOME-BACK**. Enrollment requires an explicit confirmation and a display name. Recognition stays disabled until the user enrolls.
- OpenCV Zoo dynamic-input YuNet/SFace models are downloaded by the installer when available or automatically on first enrolment into a per-user cache. They run locally; face recognition is not an authentication or security feature.
- On enrollment, the service samples several face embeddings from the existing camera stream. It stores only a numeric embedding and display name in `~/.spatial_desktop/face_profile.json`, not raw images or video. The profile can be re-enrolled or deleted from the same Settings button.
- Once enabled, face checks run at low frequency to reduce CPU use. If the enrolled person reappears after more than 30 minutes without a successful match, StreamTTS says `Welcome back, <name>`. Face embeddings are biometric information: enrollment is opt-in, local, and removable.
- Face Welcome downloads its local models automatically the first time you enrol (internet is needed only for this one-time setup). Downloads run in a background worker and do not block or close Spatial Desktop. If setup fails due to connectivity, it retries automatically while enrollment is pending; the optional `install_face_models.py` script is available only as a manual retry. Without the models, all other features continue working.

## Validation note

Python syntax and the dependency-free gesture decision rules can be tested without a webcam. Real landmark recognition, native window dragging, and scroll routing still need to be tested with your camera and desktop environment.

## Edge tracking and deliberate fist-shake grab (v30)

- **Taskbar/edge clicks:** MediaPipe tracking and pose thresholds are more tolerant near screen edges, and cursor smoothing uses a smaller edge band so the pointer does not feel stuck near minimise/restore controls. Long holds directly over the Windows taskbar are not treated as window drags; use a quick grasp/open or quick pinch to activate taskbar icons.
- **Skeleton overscan:** actual cursor positions remain clamped to the physical display, but skeleton joints use an unclamped visual transform. When a hand reaches beyond the calibrated camera region, joints are clipped by the screen rather than all collapsing onto an artificial horizontal/vertical wall.
- **Window drag:** hold a fist until the confirmation timer appears, then shake side-to-side. Three clear horizontal direction changes over a meaningful span arm move mode; the window does not move during the shake. Once recognised, a cyan highlighted border identifies the grabbed window. Subsequent fist movement moves it; opening the hand releases it.
- **Scroll disambiguation:** page scrolling starts only after a clearly vertical trajectory in content or over a scrollbar. Title-bar/non-client areas do not start scrolling. Horizontal window movement requires the explicit shake gesture, reducing accidental window drags during page scrolling.
- **Release cooldown:** an open-hand pose must remain stable for 0.20 seconds to end an active drag/scroll. Short false-open classifications reset the release timer, and missing camera frames retain the separate tracking-loss grace.

The edge pose thresholds are intentionally only slightly more permissive than the central-screen thresholds. This helps targeting near the taskbar without broadly increasing false fist/pinch activations. The shake detector and scroll arbitration have dependency-free unit tests; real camera and native-window behaviour still need to be tested on Windows.

## Voice-controlled hand tracking pause/resume

With Vosk voice input enabled, say **“pause tracking”** or **“pause hand tracking”** to temporarily disable hand-landmark inference and all hand-driven cursor/gesture input. Any active drag or scroll is cancelled safely and held mouse buttons are released. A visible banner reminds you to say **“resume tracking”** or **“resume hand tracking”** to restore hand input. Voice recognition itself remains active throughout, so you do not need to use the keyboard or hands to resume.

The camera continues delivering frames for the optional, explicitly enrolled local face-welcome feature, but MediaPipe hand inference is paused to reduce CPU use. Your settings and persistent interaction modes are preserved; transient gesture state is cleared to avoid a stale gesture starting when tracking resumes.



## Voice-controlled PC commands and clipboard lasso (v32)

All commands are handled locally by Vosk phrase matching; no LLM or cloud API is used. Global commands are checked before search dictation so they still work while the local search launcher is open.

### Clipboard and TLI selection

- **`copy this` / `copy selection`** sends the normal Copy shortcut to the active external app. It works with text or an object the app has selected.
- **`paste here`** closes the search launcher if needed, restores the previous external window, clicks once at the current TLI position, then sends Paste. Use **OS Desktop** mode so the click reaches the underlying app.
- **`start text selection`** presses and holds the left mouse button at the current TLI. Move your hand to select text, then say **`copy this`** to release and copy, or **`finish selection`** to release without copying. **`cancel selection`** also releases the mouse button.
- **`copy image` / `copy picture` / `start image lasso`** starts a rectangular image-region capture at the current TLI. Move to the opposite corner and say **`copy this`** or **`finish lasso`**. The screen region is placed on the system clipboard as an image; say **`paste here`** at an image-capable destination. Hover-only image copying depends on the source app; the lasso is the reliable general fallback.
- **`cut this`**, **`paste`**, **`select all`**, **`undo`**, and **`redo`** send their conventional keyboard shortcuts to the external app. The lasso mode suppresses competing hand gestures and always releases a held mouse button when finished, cancelled, paused or when the app closes.

### Media commands

- `play pause`, `play or pause`, `pause music`, `resume music`
- `next track`, `next song`, `previous track`, `previous song`
- `volume up`, `volume down`, `mute`, `unmute`

Windows uses media virtual keys. Linux uses `playerctl` for media when available and `pactl` for audio volume when available. Unsupported controls report a status message rather than crashing the app.

### Window commands

- `minimize window`, `maximize window`, `restore window`
- `switch window`, `show desktop`, `snap window left`, `snap window right`
- `close window` continues to target the external focused window and avoids closing Spatial Desktop.

### File browser and navigation commands

- `open file explorer`, `open downloads`, `open documents`, `open desktop folder`, `open pictures`, `open videos`, `open music folder`
- `new folder`, `new tab`, `close tab`, `go back`, `go forward`, `go up`, `refresh`
- `rename selected`, `delete selected` (standard Delete key; in Windows Explorer this normally moves the item to the Recycle Bin)
- `open notepad`, `open calculator`, `open user folder`

Some navigation shortcuts depend on the active application; e.g. `new tab` is useful in browsers and Explorer, while `new folder` is intended for a file manager. Always verify the selected target before saying `delete selected`.

### Selection safety

`start text selection` and image lasso are supported in **OS Desktop** mode. The text workflow uses a real held mouse button so the target app performs native text selection. The image workflow copies a screen rectangle to the clipboard, not the original embedded image object. Clipboard actions remain local; no clipboard contents are uploaded or indexed automatically.

## Voice PC controls and spatial clipboard selection (v32)

The PC voice-command router is local and rule-based; it does not use a language model. Commands are processed before search dictation.

**Clipboard and selection:**
- `copy this` / `copy selection`: send the standard Copy shortcut to the external app's current selection.
- `paste here`: in OS Desktop mode, focus the external app, click once at the current TLI, then paste clipboard contents.
- `start text selection`: begin a native mouse drag at the TLI; move the hand to select text and say `copy this` or `finish selection`.
- `copy image` / `copy picture` / `start image lasso`: begin a rectangular screen-region selection at the current TLI. Move to the opposite corner and say `copy this` (or `finish lasso`) to copy that screen crop as an image. This is a bitmap capture of a region, not the embedded image object itself.
- `cut this`, `paste`, `select all`, `undo`, `redo`, and `cancel selection` operate on the current external app. The lasso locks out competing hand gestures and releases the mouse button when finished or cancelled.

**Media:** `play pause`, `next track`, `previous track`, `volume up`, `volume down`, `mute`, `unmute`.

**Window management:** `minimize window`, `maximize window`, `restore window`, `switch window`, `show desktop`, `snap window left`, `snap window right`, `close window`.

**File manager/navigation:** `open file explorer`, `open downloads`, `open documents`, `open desktop folder`, `open pictures`, `open videos`, `open music folder`, `new folder`, `new tab`, `close tab`, `go back`, `go forward`, `go up`, `refresh`, `rename selected`, `delete selected`, `open notepad`, `open calculator`, and `open user folder`.

Text selection uses the app's normal mouse-drag selection, so the source application decides what text is selected. Image lasso captures the rectangular pixels between two TLI positions to the local system clipboard. `paste here` clicks the TLI location first; use it over a text insertion point or an app that accepts pasted images. Lasso and TLI paste require OS Desktop mode. Unknown phrases are not treated as commands, and a command is not sent if Windows/Linux can't safely identify an external target window.


## Settings and interaction feedback (v33)

- **Settings sub-pages:** Interaction, Notifications, and Voice keep the glass settings panel compact and make controls easier to find.
- **Notification placement:** the tracking-paused banner and Vosk heard/transcript bar can each cycle independently through top-left, top-centre, top-right, bottom-left, bottom-centre and bottom-right. Positions are saved in `ui_settings.json`.
- **Voice settings:** cycle through the installed StreamTTS voice IDs, enable/disable local TTS, set speech speed from `0.7x` to `1.5x`, test the voice, and manage optional face welcome from the Voice page. The first engine load starts in a background thread at application startup (or when TTS is enabled) to reduce first-speech delay.
- **Voice navigation:** `show menu` toggles the glass menu; `show context` toggles the Pinch Menu at the current TLI/cursor position. Both are handled before search dictation.
- **Scroll direction:** the Interaction page includes a `SCROLL · NORMAL/INVERTED` toggle. Its state persists in `gestures.txt` and applies to Windows targeted scrolling and the PyAutoGUI fallback.
- **Move-mode feedback:** after fist-shake recognition has armed a window drag, expanding cyan waves and sparkles animate from the hand/TLI in addition to the target window highlight. The effect is absent during the shake itself.


## v34 interaction and dictation updates

- **Visible TLI over Search:** a click-through, topmost overlay reasserts its z-order while the search launcher is open. It draws the hand skeleton, TLI, progress ring and Pinch Menu above the launcher without taking keyboard focus or intercepting clicks.
- **More reliable Pinch Menu:** the menu timer defaults to a 1.0-second fill after the existing 0.8-second confirmation pause (about 1.8 seconds total). Pinch detection allows a naturally bent index finger and requires at least two of the other three long fingers to remain extended. Short tracking/pinch recognition gaps reuse the same timer for up to 0.70 seconds.
- **Progressive grip-scroll:** small movements remain slow and precise; accumulated hand travel ramps scroll gain smoothly for longer movement. Scroll inversion remains configurable.
- **Focused-field dictation:** say `begin dictation` to stream recognised words into the currently focused application field, then say `end dictation` to stop. Stable words are typed while Vosk emits partials; the stop phrase is not typed into the field.
- **Pinch Menu controls:** the menu includes persistent `DICTATION` and `STT VOICE INPUT` options. The former toggles dictation to the search field or current focused OS field; the latter starts/stops Vosk itself.

When the search launcher is open, the transparent visual layer is periodically reasserted as a non-activating topmost window. This keeps the hands/TLI and Pinch Menu visible above the compact search bar while preserving mouse and keyboard input to the underlying search controls.


## v35 interaction fixes

- Window dragging now uses a two-fist clutch: make both hands fists, hold one still and move the other. The moving hand must travel decisively while the other remains steady; the movement gesture itself does not arm drag until the distinction is clear. Opening either hand releases the drag.
- Cursor lock priority is configurable in Interaction Settings and persists as `cursor_lock_priority=right` or `cursor_lock_priority=left` in `gestures.txt`. Right remains the default. An already active gesture retains its owner.
- TLI anchor is the thumb/index base midpoint with a small outward-from-wrist offset to keep it away from the index joint.
- Voice commands include `pause dictation`, `resume dictation`, and `stop dictation`. The terminal phrase recognises both “end dictation” and the common Vosk misrecognition “and dictation”.
- The Pinch Menu's Right Click item performs a right-click at the selection TLI position immediately instead of toggling persistent right-click mode.


## v36 gesture stability updates

- Pinch can grab Settings sliders using the same knob snap/pinned TLI flow as a fist.
- Cursor smoothing slider ignores isolated large cursor jumps so a transient malformed hand doesn't slam it to an endpoint.
- Right/left handedness labels are normalised once at the camera tracker boundary; colour swatches and their click targets use matching physical-hand labels. Cursor priority now uses the selected priority consistently for idle gesture initiation.
- Stronger confidence, hand-geometry and consecutive-frame checks reject stray/phantom hands when no hand is actually in view.
- Glass menu, Pinch Menu and Settings sheets have a frosted translucent treatment and soft layered edge blur.


## v36 interaction reliability

- Pinch and fist both snap to Settings sliders and pin the TLI to the knob while dragging.
- The cursor-smoothing slider rejects sudden single-frame cursor/landmark jumps; a new location must remain consistent for three frames before it can move the value.
- MediaPipe's handedness labels are normalized at the tracker boundary for the physical LEFT/RIGHT hands, and the cursor-priority setting controls gesture initiation consistently.
- LEFT and RIGHT colour swatches, click targets, and displayed labels are matched to the same physical-hand keys.
- Stray hands are filtered by confidence, plausible landmark extent/palm geometry, and consecutive-frame position stability.
- The glass menu, Pinch Menu and Settings pages use a more translucent frosted-glass finish with soft halo edges.


## v37 frosted backdrop, v38 in-menu help and modifier drag

- **Help pages are in the glass menu:** Gesture Map and Speech Commands each open as their own glass-menu page with a Back control; neither uses a system message box that can appear behind the app.
- Hold the **LEFT hand in a fist** as a temporary modifier. A quick **RIGHT-hand pinch and release** sends a right-click at the pinch start point. A held **RIGHT-hand grip** starts dragging the native window under the TLI; open the right hand to release. The left fist itself does not trigger a click or hold timer.
- Modifier inputs use a dedicated path so rapid right-hand pinches never build the Pinch Menu timer. The existing Pinch Menu and native drag paths remain otherwise independent.
- Hold a peace sign (index and middle fingers raised) for **1.5 seconds** to detach/reattach that hand from native OS mouse movement. A ring and numeric countdown show progress; TLI tracking and glass/Pinch Menu gestures remain active.
- The TLI has a small bias toward the thumb side of the thumb/index web space.
- The main glass menu and floating Pinch Menu use a Gaussian-blurred backdrop clipped to their rounded panels, plus a darker frosted tint for stronger text contrast.


## v39 tracking, 3D modifier, timer and glass updates

- A left fist now modifies right-hand gestures in both OS Desktop and virtual/3D mode. In OS mode, right pinch-release performs a native right-click and right grip drags the native window. In virtual/3D mode, right pinch-release opens the floating 3D context menu and right grip drags the virtual canvas/application surface; opening the right hand releases the drag.
- A hand paused by the peace-sign toggle no longer contributes a cursor point or moves the OS mouse. Its last TLI stays frozen while the hand remains visible for the peace-sign re-enable gesture. Active interactions are cancelled safely when a hand is disabled.
- A detached-hand notice remains on screen while either hand is detached, at the configured notice position. Reattaching the final detached hand shows a brief “Hand mouse attached” notice. Toggles do not queue delayed TTS.
- Fist/grip and timed pinch progress now show a visible ring and numeric seconds-to-trigger from the start of their timer, matching the peace-sign countdown style.
- Glass backdrop Gaussian blur is stronger in OS and virtual modes, and the modifier notification now follows the configured notification anchor instead of being fixed at top-centre.


## v40 Pinch Menu, hand visual and tracking feedback fixes

- The provided CGTrader mannequin asset is listed as unrigged and untextured, so it cannot deform directly from MediaPipe landmarks without a rig/skin-weight conversion pipeline. The live hand display now uses a thickened, rounded, warm skin-toned rendering with a shaded palm instead of thin skeleton sticks. See the [CGTrader hand model listing](https://www.cgtrader.com/free-3d-models/character/human-anatomy/hand-model-5007cc3c-effe-4560-b50e-719d56c8e37d).
- Pinch detection is more tolerant of natural tracking jitter, while still requiring two extended non-index fingers so a closed fist alone does not qualify. A stable pinch can re-arm after a fist; a bounded 0.8-second soft buffer avoids getting stuck waiting for a perfect open-hand classification.
- Active drag/scroll stops after a stable open hand for 0.20 seconds, while the Pinch Menu keeps a separate 0.8-second soft re-arm/sequence buffer for gesture jitter.
- The Pinch Menu fades in on completion while its full timer ring fades out; closing the menu also fades out rather than disappearing abruptly.
- Mouse-detachment state is visual-only. If either hand is detached, the notification remains until that hand is reattached. Reattaching the final detached hand shows a brief attached notice. TLI and glass gestures remain live while detached.


## v40.1 pinch transition and responsiveness fixes

- Pinch Menu no longer blocks on a synchronous full-screen capture when its hold timer completes. It fades in immediately using the cached backdrop or a blurred fallback, while the finished timer ring fades out.
- Desktop blur now uses a stronger reduced-resolution Gaussian pass, preserving a wide soft blur radius while reducing UI stalls.
- A confirmed pinch can transition directly out of a previous fist timer. If scrolling is active, a stable pinch immediately stops scroll input and starts the Pinch Menu timer.
- The 0.8-second pinch re-arm/sequence buffer is restored. Active drag/scroll ends after a stable open hand for 0.20 seconds, so page scrolling does not feel stuck after release.
- Mouse detachment remains visual-only: a detached hand keeps its persistent notice while TLI and glass interactions remain live.

## v41 updates

- Interaction settings include a **Skin Tone Darkness** slider (0–100%) for the live hand mesh; it is saved in `gestures.txt`.
- Default TLI colours are **left red** and **right blue**; both swatches remain independently configurable.
- Side-on hand views use relaxed projected-geometry checks and lower MediaPipe confidence thresholds, with two stable frames and a 160ms confirmed-landmark grace period to bridge brief dropouts.
- The Pinch Menu's **0.8-second soft gesture/re-arm buffer remains fixed**, independent of the 0.20-second active drag/scroll release cooldown. It is intentionally not configurable or removed by the new settings.

## v42 phantom-hand stabilisation

The tracker now uses more conservative MediaPipe confidence thresholds and requires three spatially consistent detections before drawing a hand. It no longer paints a cached skeleton after detection is lost, preventing stale landmarks from appearing as a phantom hand. Side-on hand geometry is still supported through a narrow exception that requires a substantial other image axis, non-trivial landmark depth, a valid wrist-to-palm span, and higher handedness confidence. The fixed 0.8-second gesture confirmation and pinch re-arm buffers are unchanged.


## v44 interaction refinements

- **Finger thickness control (historical v44):** Interaction Settings includes a saved slider; the current v51 range is documented below and adjusts the connected hand mesh.
- **Gentler grip-scroll:** a scroll action starts from the current hand position rather than replaying travel from the gesture-confirmation period. Frame-to-frame vertical deltas are clamped and low-pass filtered; scroll gain starts low and ramps gradually over sustained movement. Edge auto-scroll adds fractional movement instead of jumping by a whole wheel notch after only a few frames.
- **Translucent, non-overlapping notifications:** modifier, transcript, tracking-state and paused-tracking bars now use semi-transparent fills. Notifications sharing the same configured anchor are stacked vertically with a consistent gap, and the saved top/bottom and left/centre/right placement preferences are retained.
- The separate 0.8-second gesture confirmation and Pinch Menu re-arm buffers remain unchanged.


## v46 hand sizing and right-fist scrolling

- **Hand scale:** the saved hand skeleton/TLI scale slider now spans 0.1×–2.0×. At the small end, the rendered hand remains anchored around the fingertip group as in v45.
- **Finger thickness:** the saved finger-only thickness slider now spans 0.8×–5.0×.
- **Right-fist scrolling:** there is no fist countdown ring. With only the RIGHT hand in a fist, deliberate vertical movement over page content arms scrolling directly; frame deltas still use the prior smoothing and spike clamp. The LEFT fist remains the modifier, and the fixed Pinch Menu 0.8-second buffers are unchanged.


## v47 hand scaling and pinch timer fixes

- **Hand scale:** the slider is limited to 0.3×–1.0×. The hand and TLI share one transform, so interaction stays attached to the shrunken hand. The later v49 fix changes the transform origin to the wrist to stop grip-induced jumps.
- **Finger thickness:** the saved Interaction Settings slider now spans 0.8×–8.0×.
- **Pinch timer:** the circle remains empty for a continuous 0.8-second confirmation hold. Any confirmed release before the buffer expires cancels that sequence; a quick second pinch starts a new 0.8-second buffer and cannot inherit the first pinch's elapsed time. Once confirmed, the circle accumulates only actual pinch-down time; release gaps and duplicate frames contribute nothing.

## v48 changes

- The live TLI interaction point follows the same scaled transform as the visible hand, including clicks and hit-testing.
- Default hand size is 0.3× and startup mode is OS Desktop.

## v49 changes

- **Stable hand mapping:** the wrist is the fixed scale origin. The same wrist-anchored transform is applied to all 21 tracked landmarks and the actual TLI interaction point, including the points used for clicks, drags, and hit-testing. Curling the fingers into a grip no longer changes the transform origin and shifts the whole hand.
- **Continuous pinch buffer:** a real pinch release before the initial 0.8 seconds cancels the timer sequence. A quick double-pinch cannot pool elapsed time across releases; the next pinch starts a fresh buffer. The ring stays empty and shows the buffer countdown first, then shows the remaining fill duration.
- Cursor smoothing is applied after mapping the hand anchor through the common wrist scale transform, so the OS pointer and the displayed TLI follow the same mapped point.


## v50 changes

- **Finger thickness:** Interaction Settings now ranges from **0.8× to 10×**, with **8× as the default**. Thickness is scaled uniformly with a palm-size estimate from the tracked landmarks, so apparent finger width grows as the hand gets closer and shrinks as it moves farther away.
- **Continuous hand colouring:** rounded finger shafts, tips, knuckles and palm are merged into one connected silhouette before a single skin gradient is applied. This removes the visibly separate coloured circles formerly painted over line ends.
- **TLI placement and feedback:** the TLI is centred midway between the thumb tip and index fingertip; the same point drives the activation/click location. Its 54px diameter, vivid colour and layered glow improve visibility, while the progress ring is moved outward to remain visible.


## v51 contact tracking and quick-pinch double-click

- **Contact-aware hand scaling:** each hand normally scales around its wrist to prevent grip-induced drift. When both hands approach, the pivots blend toward the nearest pair of real fingertips; at fingertip contact, that contact remains fixed during shrinking. The same pivot and scale transform is used for all 21 landmarks and the transformed TLI/activation point.
- **Quick-pinch double-click:** two distinct short pinches at approximately the same point send two ordinary native click events at the same location, within the platform's double-click interval. A release gap as short as one tracked frame can complete the first click before the second pinch returns; the pinch click path is separate from quick-grip clicks.
- **Strict confirmation buffer:** any detected pinch release before the initial 0.8 seconds completes cancels that timer sequence, even when the release gap is shorter than the click debounce. Each re-pinch starts a fresh buffer; quick pulses cannot pool time against an old start.
- **Two-hand fingertip contact:** the contact-aware pivot blends toward the nearest real fingertips as the hands converge. A wider near-contact zone keeps their drawn tips aligned when perspective and landmark error mean the detected points do not measure exactly zero distance.
- **Distance-scaled thickness:** one direct geometric-mean palm-size factor multiplies every rendered finger/joint/palm stroke width. Larger projected hands make the mesh thicker; smaller projected hands make it thinner. One merged silhouette keeps the skin gradient continuous at line ends.
- **TLI:** diameter increased to 54px with a stronger multilayer glow. The visible TLI, transformed input cursor and activation point still share one position; the timer ring is larger to leave a clear margin around it.


## v53 hand controls, TLI calibration window, and responsive settings

- **Hand Settings tab:** hand scale, finger thickness, palm width, skin-tone darkness, per-hand TLI colour swatches, skin/skeleton rendering toggle, and separate left/right TLI calibration buttons now live together on a dedicated tab.
- **Palm width:** the 0.65×–1.80× slider expands/contracts the palm contour across the hand's transverse axis without moving the live tracker landmarks or TLI activation point. Default is 1.00×.
- **Skin/skeleton effect:** `SKIN EFFECT` renders the joined gradient hand surface; `SKELETON` switches to connected landmark lines and joints while keeping the same live tracking transform.
- **Interaction tab:** drag/scroll sensitivity defaults to 1.50×, cursor smoothing defaults to 0.90, and controls are available for the pinch-menu fill duration, TLI glow intensity, TLI hue shift, and TLI border thickness.
- **Default finger thickness:** 4.24×. The saved slider range remains 0.8×–10×.
- **TLI calibration:** selecting a left/right calibration button opens a separate temporary target window. Release the selecting pinch, then hold a fresh pinch inside the outlined area for 12 stable frames. The target window closes automatically on success; Escape cancels it. While the window is active, hand gestures cannot move the OS pointer or activate other controls. Calibration remains independent for each hand and is stored in `tli_calibration.json`.
- **Responsive glass settings:** panel dimensions are selected per page and scaled to the display size, with four non-overlapping tabs for Interaction, Hands, Notifications, and Voice.


## v54 visual and calibration updates

- The TLI calibration target has a transparent interior so live hand tracking remains visible through the window.
- Starting calibration for a peace-paused hand automatically resumes that hand and clears stale cursor smoothing.
- Palm width defaults to 1.26x.
- Interaction settings now expose TLI hue, glow, border thickness and TLI diameter (24–96px); Hands settings expose skeleton stroke thickness.
- Settings sliders have more room below the page tabs; the Back control is bottom-centred on each page.
- The Pinch Menu has additional padding below the lowest buttons. Glass and Pinch Menu panel tints are set to approximately 15% transparency.


## v56 — hand onboarding and more compatible scrolling

The first launch now opens a dedicated hand setup wizard. It captures a screen-touch anchor (extend an arm and touch the display just under the camera), a palm-facing size reference, and the relative distance between two hands when their fingertips meet. Samples are averaged over 20 valid camera frames. The resulting `hand_calibration.json` stores a screen translation anchor, per-hand apparent palm references, and palm-relative proximity thresholds. It is created on successful calibration and is intentionally not bundled, so a fresh install offers onboarding once. Skipping also records that choice; use **Settings → Hands → Run Hand Scale & Proximity Calibration** to run it again.

During onboarding all normal mouse/gesture actions are suspended. Live landmarks and the TLI are rendered directly over the wizard/target window so they remain visible above the calibration outline. TLI target hit-testing now uses the same transformed activation point as the visible indicator. Two-hand pivot blending and rendered hand-stroke width use the current palm size relative to calibration references instead of relying only on display-sized constants.

Scroll input is sent as native wheel input at the live TLI position instead of posting `WM_MOUSEWHEEL` directly to a captured top-level window. This allows Windows to route the wheel event to the child content control under the pointer, which is more compatible with File Explorer and PowerPoint.


## v60 interaction refinements

- **Scroll acceleration:** the Interaction tab adds a saved `SCROLL ACCELERATION` slider from `0.5×` to `10.0×`. It controls how quickly fist-drag scrolling ramps up as the hand moves farther from its original grab point; default is `1.0×`.
- **Main glass menu Close:** a wide `CLOSE ×` banner at the top of the main menu hides the glass menu without entering another page.
- **Gesture timer default:** the Pinch Menu ring-fill duration defaults to `0.6` seconds. The separate `0.8`-second confirmation buffer remains unchanged.


## Tests

Automated tests are collected in `tests/`. Run them from the project root with `python -m pytest -q tests` (voice-environment-specific tests may require optional voice dependencies and models).

## Custom gestures (Settings → Gestures)

Record your own hand poses and give each one an action.

- **Record Left Hand / Record Right Hand**: after a 2-second countdown, hold the pose still while 15 frames are captured. The pose is stored as how bent each finger is (measured in 3D), so it works wherever your hand is and copes with moderate rotation. The row label shows which fingers are out, e.g. `T·P` for thumb and little finger, or `I·M·R` for three fingers.
- Poses that clash with built-in gestures (fist, open hand, pinch, peace sign) are refused, as are poses too similar to one you've already recorded.
- **Split poses (Vulcan salute)**: the gaps between neighbouring fingers are measured too. These are the angles between index–middle, middle–ring and ring–pinky, taken flat across the palm so turning the hand doesn't change them. A pose counts as a *split* when one inner gap (index–middle or middle–ring) is held apart (12° or more) while the other is held together (8° or less).
  - Split poses can be recorded even with all five fingers out, because they're clearly different from the ordinary open hand. The classic index+middle | ring+pinky salute is labelled **VULCAN**; other splits show the gap with a bar, e.g. `T·I|M·R·P`.
  - A split gesture only fires when your live hand has the same gaps: each inner gap must agree to within 10°, and a gap held apart in one must not be closed in the other. A normal open hand never triggers a Vulcan gesture, and making a Vulcan salute doesn't trigger your other open-finger gestures.
  - The pinky gap isn't used to define a split, because the little finger drifts outwards on its own in a relaxed hand.
  - Gestures recorded before spread was measured behave exactly as before.
- A **fist** only counts with the thumb tucked in. Fingers closed with the thumb out is free for custom gestures: record **THUMBS UP** and **THUMBS DOWN** as separate gestures (thumb direction is compared only when the thumb is out).
- **Built-in TLI overlap**: bring both TLIs together (open hands) and hold for the hold time to toggle the glass menu. A ring shows progress. It isn't a recordable gesture.
- On each row, tap the hand chip to cycle LEFT / RIGHT / ANY, and tap the action (**▾**) to open a list of every action and pick one. **×** deletes the row.
- **Custom key shortcut**: choose **CUSTOM KEY**, then press the keyboard shortcut you want (for example `Ctrl+Alt+Shift+K`). The row shows the saved combination; selecting CUSTOM KEY again lets you record a replacement. Press unmodified `Esc` to cancel capture.
- **Circle timer (ON/OFF on each row)**: ON shows a filling ring at the TLI and fires after the hold time. OFF fires almost instantly (after a 0.12 s stability check) with no ring.
- **Hold time**: how long the pose must be held before it fires when its circle timer is on. It fires once, then re-arms when you release the pose. A green bar fills along the row while you hold it.
- **Match tolerance**: raise it if your pose isn't recognised, lower it if the wrong pose fires.
- Actions include mouse buttons (right click, middle click, mouse back, mouse forward), pressed at that hand's TLI. Like pinch clicks, they do nothing for a detached hand or over the glass menu.
- Gestures are saved in `custom_gestures.json`. A pose is ignored while that hand is in a fist, a pinch, a drag or a scroll. Holding a thumbs-up for 2 seconds disables that hand’s gestures **and cursor/TLI tracking together**; the camera-rendered virtual hand remains visible, and another 2-second thumbs-up re-enables both.

Settings → Interaction → **Scroll Dead Zone** sets how far (in pixels) the hand must move from the grab point before grip scrolling starts. Past it, scrolling starts around 2 wheel notches per second and speeds up with distance; Drag/Scroll Sensitivity and Scroll Acceleration tune it. A detached hand cannot drag-scroll.

## Detached notice fading (Settings → Notifications)

The persistent "LEFT/RIGHT MOUSE DETACHED" notice stays fully visible while the glass menu is open. Five seconds after the menu closes it fades to the **Detached Notice Fades To** value (0–100%, default 30%). Whenever its text changes (for example the other hand also detaches), or another notification appears in the stack (the tracking-paused bar or the voice transcript bar), it returns to full visibility for 5 seconds and then fades again.


## Liquid AI d1 decision layer

The Windows main installer automatically downloads the default quantized d1-omni-600M model and a local llama.cpp runtime. See `D1_SETUP.md` for setup and model-switching details. If d1 setup fails, Spatial Desktop continues to use its existing voice matching and local-search behavior.

## Local d1 decision layer

The Windows installer downloads the quantized Liquid AI d1-omni-600M model (about 670 MB including its projector) and a local llama.cpp runtime. The runtime is launched automatically by Spatial Desktop; no separate server needs to be left running. Open Glass Menu → Settings → ENGINE for the local d1 and llama.cpp controls. The d1-omni-600M Q8 model is the only available model option for now; the larger Q4 option has been removed because its download was unreliable. See `D1_SETUP.md` for details.
