# StreamTTS — cross-platform incremental speech core (C++ / no Rust)

StreamTTS is a small native C++17 library that accepts text fragments through
`pushText()` and synthesizes short chunks in the background. It is designed for
LLM/API streams, sockets, pipes, and other programs that receive text a little at
a time. It is intended to run locally/offline after the voice model and native
libraries have been installed.

**Targets:** Linux, Windows, Android (NDK + JNI). **No Rust toolchain or Rust
bindings are needed or enabled by this project.** The inference backend is
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), pinned to v1.13.8, and the
optional reference playback backend is the C library miniaudio. Audio output is
an interface callback at the core boundary, so a host can replace miniaudio with
its own WASAPI, ALSA/PulseAudio, AAudio, Oboe, or other device integration.

## Current design

- One `OfflineTts` model instance is created and held for the lifetime of the
  `StreamingTts` object (no model loading per word).
- `pushText(fragment)` is safe to call repeatedly with partial tokens, including
  fragments that split in the middle of a word.
- A segmenter releases chunks at completed-word thresholds and at punctuation.
  Balanced mode starts after two complete words, then targets three-word chunks;
  Natural mode targets longer chunks; Immediate mode targets one word.
- One worker serializes inference and delivers mono float PCM to the registered
  audio callback. The caller's callback should copy/enqueue audio before it
  returns; the supplied miniaudio adapter does this through an SPSC ring buffer.
- `flush()` emits the final trailing word even if the stream ends without a
  space. `waitUntilIdle()` waits for synthesis/audio-callback delivery, not for
  the downstream speaker to finish playback.
- Both C++ (`StreamingTts`) and C ABI (`streamtts_push_text`) are provided. A
  zero-dependency Python `ctypes` wrapper lives in `python/streamtts.py`.

### Latency and voice quality

The default sample profile is Piper `en_US-amy-low`, chosen as a lightweight
starting point and easy benchmark. Its ONNX archive is downloaded by setup; model
weights are not embedded in the source ZIP. Replace it with another sherpa-onnx TTS model
by setting `Options::model_path`, `tokens_path`, and `data_dir` appropriately.
The library does not pretend that a non-streaming VITS/Piper model can generate
its first waveform sample before inference: it synthesizes each small text chunk
then queues the resulting audio. That is incremental/chunked speech, not true
frame-level streaming from an unfinished word. Shorter chunks usually reduce
text-buffering latency but can make word stress and intonation less natural.

## Quick start: Linux / Windows

Prerequisites: CMake 3.24+, a C/C++17 compiler, Git, and Python 3 for the small
setup scripts. On the first build, CMake FetchContent retrieves the sherpa-onnx
source and its native dependencies; internet access is needed unless you already
have local copies.

1. Download the reference audio backend header and voice model:

   ```bash
   python scripts/download_miniaudio.py
   python scripts/download_model.py
   ```

2. Configure and build the native C++ core on Linux:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ./build/streamtts_cli models/vits-piper-en_US-amy-low
   ```

3. **Windows without Visual Studio:** use the Python test application below. It
   uses sherpa-onnx's prebuilt Python/native wheels, so it does not compile this
   project's C++ DLL and does not need Visual Studio or a C++ compiler. Open a
   normal Command Prompt or Windows Terminal in this folder and run:

   ```bat
   setup_windows.bat
   run_windows_test.bat
   ```

   The setup creates a local `.venv`, installs the prebuilt runtime packages and
   downloads the sample voice. In the app, type normally; it starts speaking
   after word boundaries without waiting for Enter. Backspace edits the current
   unsubmitted word, Enter flushes the line, and Ctrl+C exits.

   **Native Windows C++ build note:** sherpa-onnx's own Windows instructions say
   MinGW is not supported. Building the native C++ library from source therefore
   requires the Microsoft C++ Build Tools (the full Visual Studio IDE is not
   required). If those are installed and CMake is on PATH, run
   `build_windows_native.bat` from an ordinary CMD window. This build-tool
   requirement is separate from running the Python test application.

The native C++ CLI accepts lines and is intended as a smoke test. Real
integrations should call `pushText` as each fragment arrives rather than
waiting for an entire line.

To use an already checked-out sherpa-onnx tree instead of CMake downloading it,
pass `-DSHERPA_ONNX_SOURCE_DIR=/absolute/path/to/sherpa-onnx` to CMake.

To build only the library and segmenter tests without the example device backend,
pass `-DSTREAMTTS_BUILD_DEMO=OFF`. The demo backend downloads `miniaudio.h` during
configure if it is not already present; the script above can download it first.


## Interactive Python tester (Windows CMD / Windows Terminal)

The easiest way to try the voice on a Windows computer without a C++ compiler is:

```bat
setup_windows.bat
run_windows_test.bat
```

`python/interactive_test.py` reads keystrokes directly, so typing a sentence
feeds completed words to the TTS pipeline as you go. Balanced mode buffers two
words for the first chunk and then targets three-word chunks; `.`, `!`, and `?`
are sent immediately. The synthesizer runs on a worker thread and audio plays
from an independent output stream, so you can keep typing while speech is being
generated and played.

### Voice library expansion (v0.1.4)

The GUI's built-in Piper catalog now lists 71 official sherpa-onnx voice profiles,
including many English variants and several additional languages. Only the catalog
metadata is included in the ZIP; each voice model is downloaded from the official
sherpa-onnx `tts-models` release when first selected, so the project download stays
small. All downloaded voices are cached locally for offline use.

### Graphical settings app (v0.1.4 and later)

Launch the Tkinter GUI with `run_windows_gui.bat` after setup. Re-run `setup_windows.bat` after updating so it installs the OS-voice packages (`pyttsx3` and `pywin32`). It includes a live
text area, a test-voice button, voice selection, speech-rate control (0.5×–2.0×),
pitch shift (-6 to +6 semitones), Immediate/Balanced/Natural chunking, manual flush
(Ctrl+Enter), and a Save settings button. It remembers these preferences in
`settings.json`.

The built-in catalog now contains 71 curated Piper/VITS profiles across
English (UK and US), French, German, Dutch, Spanish, Portuguese, Italian, and
Polish. It includes many voice and quality variants, such as Alan, Alba, Aru,
Cori, Jenny Dioco, Northern/Southern English, Amy, Arctic, Bryce, Danny, GLaDOS,
Lessac, Ryan, Thorsten, Siwis, Nathalie, Daniela, and others. The list is defined
in `python/voice_catalog.py`, so additional compatible profile IDs and friendly
labels can be added easily.

To keep the project ZIP small, the catalog metadata is bundled but individual
ONNX model archives are **downloaded on first use**. Uninstalled choices are
marked `(download)` in the GUI; select one and click **Apply selected voice**.
Voice downloads are saved in `models/` and remain available offline afterward.
Available profiles are based on the official sherpa-onnx Piper model index at
https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html and other language
pages beneath https://k2-fsa.github.io/sherpa/onnx/tts/all/ . **Download other…**
also accepts a compatible Piper/VITS model ID from the official sherpa-onnx
`tts-models` release, for example `vits-piper-en_GB-cori-high`. **Import folder…**
accepts a local model folder containing an ONNX file and `tokens.txt`. The GUI
also discovers installed operating-system voices through
`pyttsx3` (SAPI5 on Windows); these use the OS engine and do not need ONNX models.
Native system voices support rate adjustment but generally do not expose pitch control.
For neural model voices, pitch control uses librosa pitch shifting on each chunk;
0 semitones bypasses the effect. Pitch shifting can add processing delay.

The pure-Python front-end in `python/streamtts_python.py` exposes `push_text()`
(and the alias `pushText()`), `flush()`, and `wait_until_idle()`. This path uses
the official prebuilt `sherpa-onnx` Python package on Windows. It exercises the
same incremental chunking policy but is not the compiled C ABI wrapper in
`python/streamtts.py`; build the native shared library before using that wrapper.

Dependencies installed into the local virtual environment are `sherpa-onnx`,
`sherpa-onnx-bin`, `numpy`, `sounddevice`, `librosa` (optional pitch shift), and `pyttsx3`/`pywin32` (system voices).
No Rust is installed or used.

### Update checker (v0.1.5)

The GUI checks for updates at startup and has an **Updates** section with
**Check now**, **Download ZIP**, and **GitHub** controls. The check runs in a
background thread, so it does not block typing or speech. It fetches the plain
text version from:

- `https://raw.githubusercontent.com/dansdesigns/streamtts/main/version.txt`
- If the `main` path returns 404, it also tries the `master` branch.

**Repository-owner setup:** publish a root-level `version.txt` in the repository
with only the version on its first line (for example, `0.1.5`). The `version.txt`
file is intentionally **not included** in this application ZIP. The app compares
that remote version with the local `VERSION` file. If GitHub has a newer version,
**Download ZIP** opens the current branch's source archive in the browser; this
checker does not replace running files automatically. Back up `settings.json`
and `models/` before manually replacing an existing installation.

## C++ API example

```cpp
#include "streamtts/streaming_tts.h"
#include "streamtts/miniaudio_sink.h"

streamtts::Options options;
options.model_path = "models/vits-piper-en_US-amy-low/en_US-amy-low.onnx";
options.tokens_path = "models/vits-piper-en_US-amy-low/tokens.txt";
options.data_dir = "models/vits-piper-en_US-amy-low/espeak-ng-data";
options.chunking = streamtts::ChunkingMode::Balanced;

streamtts::StreamingTts tts(options);             // load/warm model once
streamtts::MiniaudioSink speaker(tts.sampleRate());
tts.setAudioCallback([&](const float* pcm, size_t n, int32_t) {
  speaker.enqueue(pcm, n);                         // copy/enqueue during callback
});

tts.pushText("The first ");                       // still an unfinished 2nd word
// Your producer continues to call pushText() for each received text fragment.
tts.pushText("words can arrive ");
tts.pushText("before the full sentence is ready.");
tts.flush();                                      // emit trailing partial word
// Keep tts alive while synthesis drains. Destruction stops/joins the worker.
tts.stop();                                      // join worker before sink leaves scope
speaker.waitUntilDrained();
```

## C ABI / Python

The C ABI in `include/streamtts/c_api.h` exposes `streamtts_create`,
`streamtts_push_text`, `streamtts_flush`, `streamtts_wait_until_idle`, and
`streamtts_destroy`. Create a shared build with `-DBUILD_SHARED_LIBS=ON`, then:

```python
from streamtts import StreamTTS

# Supply your own playback callback. `samples` points to transient float PCM;
# copy it during the callback if another thread/device will use it later.
with StreamTTS(
    "build/libstreamtts.so",
    "models/vits-piper-en_US-amy-low/en_US-amy-low.onnx",
    "models/vits-piper-en_US-amy-low/tokens.txt",
    "models/vits-piper-en_US-amy-low/espeak-ng-data",
    audio_callback=lambda samples, count, rate: consume_audio(samples, count, rate),
) as tts:
    tts.push_text("Hello ")
    tts.push_text("from an incremental stream.")
    tts.flush()
    tts.wait_until_idle()
```

On Windows, pass the path to `streamtts.dll`; on Linux, use the generated
`libstreamtts.so` path. The project is native C++ but C ABI + `ctypes` lets Python
programs push text into the same engine. The callback runs on the native worker
thread, not necessarily the caller's thread.

## Android sample

Open `android/` in Android Studio with Android SDK, NDK and CMake 3.22.1
installed. The app compiles the same C++ core and `streamtts_jni` bridge; it does
not package the voice model in the APK. Copy these files to the app's private
`filesDir/models/` directory:

- `en_US-amy-low.onnx`
- `tokens.txt`
- `espeak-ng-data/` (when required by the selected voice)

The small sample activity includes `Load local voice` and `Speak` buttons.
Production code should receive text on a background/network thread and call the
JNI `NativeTts.pushText(fragment)` as fragments arrive. Avoid blocking the UI
thread on `waitUntilIdle()` for long content.

Android Gradle uses ABI filters for `arm64-v8a`, `armeabi-v7a`, and `x86_64`.
For smaller APKs, remove ABIs that your target devices do not need.

## Correctness boundaries / next steps

- This first prototype is English/Piper-oriented in its sample paths. Different
  sherpa-onnx model families can require different files/config fields; test a
  target voice before shipping it.
- The reference audio callback receives complete synthesized chunks. If a host
  requires a bounded audio queue, implement backpressure/drop policy in its
  sink; do not silently grow memory without limits.
- Current segmentation recognizes ASCII spaces and common ASCII punctuation as
  boundaries, while preserving UTF-8 bytes in words. A production multilingual
  version should use ICU or another Unicode word-break implementation.
- StreamingTts instances serialize inference on a single worker each. Avoid
  concurrent `stop()` calls from multiple threads; stop is idempotent for normal
  sequential lifetime management.
- Benchmark on actual desktop and Android hardware. Text-buffering latency is
  only one part of first-audio latency; model initialization, inference and
  device scheduling also matter.

## Dependency notes

- sherpa-onnx: Apache-2.0; see its upstream repository and the license for each
  downloaded model.
- miniaudio: see its dual MIT / Unlicense notice in the downloaded header.
- Piper voice files are separate model artifacts; check their individual
  licenses and redistribution terms before bundling them in a release.

