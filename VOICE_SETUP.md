# Offline voice and fast local file search

Spatial Desktop keeps Vosk for offline speech recognition and StreamTTS ONNX for optional natural-sounding local speech. The installer now installs the StreamTTS runtime and attempts to download its voice files automatically. Search itself does **not** need an embedding model, LLM, or cloud service.

## Fast launcher-style file search

Open the search bar with any of these:

- Say **“search bar”** or **“search files”** while Vosk voice input is enabled.
- Press **Ctrl+Space** while Spatial Desktop has focus.
- Click the **Search** tile in the glass menu.

The launcher behaves like a compact application search bar:

- Results update as you type with a short debounce.
- Click **Dictate** or open the launcher by saying “search bar” to use Vosk dictation. Partial transcripts replace the search field as Vosk hears new words, so results update before a full sentence is finalised.
- Use the Up/Down arrows to choose a result and Enter or double-click to open it.
- Click **Add Folder** to include other locations. The index is incremental and is stored locally in `~/.spatial_desktop` (on Windows this folder is under your user profile).
- Click **Speak** to have StreamTTS read the selected file name and a matching excerpt when the local TTS runtime and model files are available.
- Say “close search” to dismiss the launcher, “open top result” to open the current selection, or “stop dictation” to stop updating the query from speech.

The default index roots are common personal folders such as Desktop, Documents, Downloads, Pictures, Videos and Music (where they exist). It never indexes the whole home folder by default. Hidden and common development/cache folders are skipped. It indexes file names and paths for all files; small text and source files plus PDF and DOCX files also contribute searchable content. PDF and DOCX parsers are installed by the main installer.

## How local search ranks files

The search system is ordinary Python plus SQLite FTS5. It incrementally caches file name, parent path and bounded text excerpts, then combines BM25 full-text ranking, stronger filename/path weights, exact phrase boosts, fuzzy filename similarity, and a small transparent synonym list (for example, “photo” ↔ “picture/image” and “meeting” ↔ “call/minutes”). It uses no embedding vectors and downloads no search model. The first index pass runs in the background; later opens only update changed files.

This is a quick local *semantic-style* finder, not a neural semantic embedding system. It works best when a query includes at least some related words or known synonyms. It will not reliably understand a description with no lexical overlap (for example, “the memory of last summer” when the file only says “holiday”), as an embedding model might. We can expand and tune synonyms/ranking based on your real searches without adding a large model.

### Why not Jev/OpenJev or EmbeddingGemma yet?

- **JEV Search** demonstrates a useful hybrid: local filename/content ranking first, with optional remote Jev reranking. Its project documentation says enabling the remote reranker sends search queries, candidate paths and excerpts to its API; the app is currently presented as a macOS/Electron project. That does not match an always-local, low-overhead Windows-first requirement.
- **OpenJev** is a local decision-model approach built on top of a small language model (the documented quick-start uses Qwen3-0.6B, with around a 1.5 GB model download). It is designed to classify/score choices, not to replace a file index. It adds model memory and inference cost without being the simplest fit for incremental file search.
- **EmbeddingGemma 2** is a proper embedding/retrieval model and is the more relevant of the model-based options for semantic file matching, but its checkpoint is large compared with SQLite FTS and it needs a model inference runtime. We are deliberately leaving it out until the fast indexed baseline has been tested.

## StreamTTS local TTS

The installer runs `pip install -r requirements-voice.txt` and `python install_voice_models.py`. If either optional step fails because a network/package is unavailable, the installer warns and keeps the hand controls and search usable; retry manually from the project folder:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-voice.txt
.venv\Scripts\python.exe install_voice_models.py
.venv\Scripts\python.exe test_tts.py
```

On Linux:

```bash
.venv/bin/python -m pip install -r requirements-voice.txt
.venv/bin/python install_voice_models.py
.venv/bin/python test_tts.py
```

The default voice is British English (`bf_emma`). Change the voice or speech speed in **Settings → Voice** (changes are saved to `voice_settings.json`); editing that file manually is also supported. Model files are downloaded once and remain local afterwards. When model files are installed, the ONNX engine warms in a background thread during startup (or when TTS is enabled); synthesis still runs away from the UI thread and is only requested for spoken replies or the **Speak** button.

## Voice commands

With Voice Input enabled from the glass menu, built-in commands include `pause tracking`, `resume tracking`, `search bar`, `close window`, `open user folder`, `next desktop`, `previous desktop`, `home`, `help`, and `stop listening`. The search field uses Vosk partial text for live dictation; final recognition confirms the latest text rather than delaying search until the whole utterance has finished.

- Say **“pause tracking”** (also “pause hand tracking”) to suspend hand inference and all hand-driven cursor/gesture input. Any active drag/scroll is cancelled safely, mouse buttons are released, and a visible banner explains how to resume. Vosk stays active so you can say **“resume tracking”** (also “resume hand tracking”) to turn hand input back on. The camera continues providing frames to the explicitly enrolled, local face-welcome feature, but MediaPipe hand inference is paused to reduce CPU usage.


## v29 additions

The Windows/Linux installer attempts to download the OpenCV Zoo dynamic-input YuNet face detector and quantized SFace recognizer into the per-user cache. If that initial download fails or was skipped, pressing Enrol Face downloads the models automatically in a background worker; you do not need to run a Python script manually. Model downloading happens off the GUI thread and the app stays running. Face Welcome is inert until you explicitly enroll a face through Settings. The saved profile contains only a numeric embedding and a display name at `~/.spatial_desktop/face_profile.json`; it does not store camera images. You can remove it via the same Settings button.

At startup, StreamTTS speaks a time-of-day greeting when its voice runtime and model files are ready. The text includes the enrolled display name if present, otherwise the operating-system username. Voice commands now include `close window` and `open user folder`.


## Voice-controlled PC commands (v32)

The PC command router is deterministic and offline. It handles exact phrase aliases before launcher dictation, so commands are not typed into the search box when the launcher is open.

| Say | Action |
|---|---|
| `pause tracking` / `resume tracking` | Stop or resume hand inference while keeping Vosk running |
| `copy this` / `copy selection` | Copy the current native selection or selected object |
| `paste here` | Click at the TLI and paste clipboard contents into the target |
| `start text selection` | Mouse-down at the TLI; move hand to extend native text selection |
| `finish selection` | Release the held selection drag |
| `copy image` / `start image lasso` | Start a rectangular screen-image selection; move to the opposite corner then say `copy this` |
| `cancel selection` | Release held mouse input and cancel lasso |
| `cut this`, `paste`, `select all`, `undo`, `redo` | Standard editing shortcuts |
| `play pause`, `next track`, `previous track` | Media controls |
| `volume up`, `volume down`, `mute`, `unmute` | System audio controls where the OS exposes them |
| `minimize window`, `maximize window`, `restore window` | Control the external active window |
| `switch window`, `show desktop`, `snap window left/right` | Window management shortcuts |
| `open file explorer`, `open downloads`, `open documents`, `open pictures`, `open videos`, `open music folder` | Open common user folders |
| `create new folder`, `create new text file`, `create new powerpoint`, `new tab`, `close tab`, `go back`, `go forward`, `go up`, `refresh` | File-browser/browser navigation shortcuts |
| `rename selected`, `delete selected` | Standard rename/delete actions in apps that support them |
| `open notepad`, `open calculator`, `open user folder` | Launch app or open home folder |

For **text selection**, say `start text selection`, move the TLI from the beginning to the end of the desired text, then say `copy this`. For a **picture or screen region**, say `copy image` while the TLI is at one corner, move to the opposite corner, then say `copy this`; this places a bitmap of that region on the clipboard. To paste, move the TLI to the destination text field or image-capable app and say `paste here`.

TLI paste and lasso operations require **OS Desktop** mode. `paste here` clicks at the TLI first, so aim at a text field or document insertion point rather than a button. `delete selected` sends the normal Delete key; confirm the selected item before using it. All matching is offline and rule-based; no generative model is involved.

## Desktop voice command reference (v32)

The command router is deterministic and offline. It is evaluated before search dictation to keep command phrases from being typed into the launcher.

- **Clipboard:** `open clipboard` / `show clipboard`, `close clipboard` / `hide clipboard`, `copy this`, `copy selection`, `cut this`, `paste`, `paste here`, `select all`, `undo`, `redo`.
- **Text lasso:** say `start text selection`, move your hand to extend the native selection, then say `copy this` or `finish selection`. Say `cancel selection` to release the mouse and cancel.
- **Image lasso:** say `copy image` or `start image lasso` at the first corner, move the TLI to the opposite corner, then say `copy this` or `finish lasso`. A screen crop goes to the clipboard as an image.
- **Media:** `play pause`, `next track`, `previous track`, `volume up`, `volume down`, `mute`, `unmute`.
- **Windows:** `minimize window`, `maximize window`, `restore window`, `switch window`, `show desktop`, `snap window left`, `snap window right`, `close window`.
- **Folders/apps:** `open file explorer`, `open user folder`, `open downloads`, `open documents`, `open desktop folder`, `open pictures`, `open videos`, `open music folder`, `open notepad`, `open calculator`.
- **Navigation:** `create new folder`, `create new text file`, `create new powerpoint`, `new tab`, `close tab`, `go back`, `go forward`, `go up`, `refresh`, `rename selected`, `delete selected`.

`paste here` restores the last external application, clicks at the TLI and presses Paste. Use OS Desktop mode. Always inspect the target before saying `delete selected`; it sends the normal Delete key. On Linux, media controls use `playerctl` and `pactl` when installed. Voice commands execute locally without an LLM.


## v33 voice and overlay settings

In the glass menu open **Settings**, then choose **Interaction**, **Notifications**, or **Voice** from the tabs. The Notifications page independently controls the `HAND TRACKING PAUSED` banner and the Vosk heard/transcript bar. Each button cycles between six screen corners/centres and saves to `ui_settings.json`.

The Voice page lets you toggle local TTS, cycle installed StreamTTS voice IDs, adjust speech speed from `0.7x` to `1.5x`, play a test phrase, and enrol/manage Face Welcome. StreamTTS's ONNX engine begins loading in the background when the installed model files are available, reducing first-speech initialisation delay.

Say **`show menu`** to toggle the glass menu, or **`show context`** to toggle the Pinch Menu near the current TLI/cursor. Say **`help`** for the command summary.

The Interaction page's **Scroll · Normal/Inverted** button reverses vertical scroll direction and saves the choice in `gestures.txt`. When a fist-shake successfully arms a native window drag, cyan waves and small sparkles radiate from the TLI to indicate that move mode is engaged.


## v34 voice and interaction shortcuts

- Say **`begin dictation`** to type Vosk-recognised words into the currently focused OS text field. Partial transcripts stream stable words as they arrive; the newest word is held until stable to avoid repeated/incorrect typing. Say **`end dictation`** to finish; the phrase itself is not typed.
- In the search launcher, use its Dictate control (or open it with `search bar`) to dictate into the search field specifically.
- The Pinch Menu adds **Dictation** and **STT Voice Input** switches. Dictation toggles routing into the active search field/focused field; STT Voice Input starts or stops Vosk recognition.
- The Pinch Menu hold defaults to a **1.0-second fill after a 0.8-second confirmation delay** (about 1.8 seconds total). Pinch recognition now allows natural index bending and tolerates brief landmark dropouts so the timer is less likely to restart.
- With Search open, the skeleton, TLI, hold progress and Pinch Menu are drawn on a transparent, click-through layer above the launcher.
- Grab-scroll is progressive: short travel remains precise, and gain increases as accumulated vertical movement grows. `invert_scroll` still reverses direction.

## StreamTTS Config GUI

Open **Settings → Engine → Open StreamTTS Config** to launch the standalone `streamtts_config.py` window. It lets you enable/disable speech output, choose an installed StreamTTS/Piper voice model, and adjust speech speed. **Save settings** writes to `voice_settings.json`, the same settings file used by Spatial Desktop. The GUI uses Python's built-in Tkinter, so no extra GUI package is required; on some Linux distributions Tkinter may need to be installed separately (for example, the `python3-tk` package).

## StreamTTS voice configuration

Open Settings → Voice → **OPEN STREAMTTS CONFIG** to access the full StreamTTS voice controls: the complete bundled Piper/VITS voice catalog, installed operating-system voices (when `pyttsx3` is available), downloading another model by ID, importing a local model folder, speech rate, pitch shift, streaming mode, voice test, and saving settings. Changes to voice, pitch, and streaming mode are read by Spatial Desktop at startup; restart Spatial Desktop after saving. Missing neural voice models can be downloaded with **Apply selected voice** or on first speech use.


## Voice-driven file navigation and selection

With Voice Input enabled, the file-navigation commands can operate on the active file manager:

| Say | Action |
|---|---|
| `open my computer` / `open this PC` | Open the system's computer/drives view (This PC on Windows; filesystem root on Linux/macOS) |
| `go to C drive` / `open D drive` | Open that drive in Windows File Explorer |
| `open Agents folder` / `go to Agents folder` | Open or navigate into a named folder in the current file manager |
| `open report.docx` | Open a named file from the current folder |
| `select the Agents folder` | Select a named folder without opening it |
| `select report.docx and notes.pdf` | Select multiple named items as one multi-selection |
| `select all files` | Select files in the current Windows File Explorer folder, excluding folders |
| `select all folders` | Select folders in the current Windows File Explorer folder, excluding files |
| `select all files and folders` | Select every item in the current folder |

On Windows, named-item selection uses Windows Shell automation against the active File Explorer folder. Names are matched case-insensitively; if a filename extension is omitted, a unique basename match is allowed. When names are ambiguous, say the full filename or specify `file`/`folder`. For named item and type-filtered selection commands, keep the target File Explorer folder active. `open my computer` and drive navigation can be used from elsewhere.

On macOS, Finder name selection/opening uses AppleScript. On Linux, named selection is best-effort using X11 type-ahead via `xdotool`, and support for filtering only files versus folders depends on the file manager; Wayland sessions or file managers without type-ahead support may not support those operations. These voice commands are local and rule-based.
