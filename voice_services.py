"""Optional offline speech services for Spatial Desktop.

Speech recognition stays on Vosk. StreamTTS provides optional offline, low-latency streaming speech output.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
VOSK_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
STREAMTTS_DIR = ROOT / "streamtts"
TTS_MODELS_DIR = ROOT / "models"
TTS_SETTINGS = ROOT / "voice_settings.json"

try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    import vosk
except Exception:
    vosk = None


DEFAULT_STREAMTTS_VOICE = "vits-piper-en_GB-alan-medium"
STREAMTTS_VOICES = (
    "vits-piper-en_GB-alan-medium",
    "vits-piper-en_GB-alba-medium",
    "vits-piper-en_GB-cori-medium",
    "vits-piper-en_GB-jenny_dioco-medium",
    "vits-piper-en_US-amy-low",
    "vits-piper-en_US-ryan-medium",
)


def load_tts_settings() -> dict:
    defaults = {"enabled": True, "voice": DEFAULT_STREAMTTS_VOICE, "speed": 1.0, "pitch": 0.0, "mode": "balanced", "language": "en-gb"}
    try:
        supplied = json.loads(TTS_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return defaults
    if isinstance(supplied, dict):
        defaults["enabled"] = bool(supplied.get("enabled", defaults["enabled"]))
        voice = supplied.get("voice")
        # Migrate the former Kokoro voice setting to the new British English default.
        if isinstance(voice, str) and voice:
            if voice in STREAMTTS_VOICES or voice.startswith("vits-piper-") or voice.startswith("system:"):
                defaults["voice"] = voice
            else:
                # Permit locally imported model folders created by StreamTTS Config.
                candidate = TTS_MODELS_DIR / voice
                if candidate.is_dir() and (candidate / "tokens.txt").is_file() and list(candidate.glob("*.onnx")):
                    defaults["voice"] = voice
        try:
            speed = float(supplied.get("speed", defaults["speed"]))
            if speed == speed and 0.5 <= speed <= 2.0:
                defaults["speed"] = speed
        except (TypeError, ValueError):
            pass
        try:
            pitch = float(supplied.get("pitch", defaults["pitch"]))
            if pitch == pitch and -6.0 <= pitch <= 6.0:
                defaults["pitch"] = pitch
        except (TypeError, ValueError):
            pass
        mode = supplied.get("mode", defaults["mode"])
        if mode in ("immediate", "balanced", "natural"):
            defaults["mode"] = mode
        lang = supplied.get("language", defaults["language"])
        if lang in ("en-gb", "en-us"):
            defaults["language"] = lang
    return defaults


class Speech:
    """Threaded, bounded-queue Vosk STT with partial transcript updates.

    The audio callback never waits on decoding. If the decoder falls behind,
    the oldest block is dropped rather than growing latency indefinitely.
    """
    def __init__(self, callback: Callable[[str], None],
                 partial_callback: Optional[Callable[[str], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.partial_callback = partial_callback or (lambda _text: None)
        self.status_callback = status_callback or (lambda _text: None)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.q: queue.Queue = queue.Queue(maxsize=12)
        self.model = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._resume_reset = threading.Event()
        self._last_partial = ""
        self._last_final = ""
        self._last_final_at = 0.0
        if vosk and sd and VOSK_MODEL.exists():
            try:
                try:
                    vosk.SetLogLevel(-1)
                except Exception:
                    pass
                self.model = vosk.Model(str(VOSK_MODEL))
            except Exception as exc:
                self.status_callback(f"Vosk model unavailable: {exc}")

    def start(self) -> bool:
        if self.running:
            return True
        if sd is None:
            self.status_callback("Microphone unavailable: install sounddevice/PortAudio")
            return False
        if self.model is None:
            self.status_callback(f"Vosk model not found at {VOSK_MODEL}")
            return False
        self._stop.clear()
        self._clear_queue()
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="Vosk-STT", daemon=True)
        self.thread.start()
        self.status_callback("Voice input listening")
        return True

    def _clear_queue(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def set_paused(self, paused: bool):
        if paused:
            self._paused.set()
            self._clear_queue()
        else:
            self._paused.clear()
            self._resume_reset.set()

    def _audio_callback(self, indata, frames, timing, status):
        if not self.running or self._stop.is_set() or self._paused.is_set():
            return
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            # Prioritise current speech over delayed audio blocks.
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(bytes(indata))
            except queue.Full:
                pass

    def _loop(self):
        recognizer = None
        try:
            recognizer = vosk.KaldiRecognizer(self.model, 16000)
            try:
                # Faster end-of-utterance response, without making pauses overly eager.
                recognizer.SetEndpointerDelays(0.5, 0.45, 8.0)
            except Exception:
                pass
            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16",
                                   channels=1, callback=self._audio_callback):
                while not self._stop.is_set():
                    try:
                        data = self.q.get(timeout=0.10)
                    except queue.Empty:
                        continue
                    if data is None:
                        break
                    if self._paused.is_set():
                        continue
                    if self._resume_reset.is_set():
                        try:
                            recognizer.Reset()
                        except Exception:
                            pass
                        self._resume_reset.clear()
                    if recognizer.AcceptWaveform(data):
                        try:
                            text = json.loads(recognizer.Result()).get("text", "").strip()
                        except (ValueError, TypeError):
                            text = ""
                        self._publish_partial("")
                        if text:
                            now = time.monotonic()
                            # Avoid duplicate final callbacks for an identical endpoint.
                            if text != self._last_final or now - self._last_final_at > 1.2:
                                self._last_final = text
                                self._last_final_at = now
                                self.callback(text)
                    else:
                        try:
                            partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                        except (ValueError, TypeError):
                            partial = ""
                        if partial != self._last_partial:
                            self._publish_partial(partial)
        except Exception as exc:
            if not self._stop.is_set():
                self.status_callback(f"Voice input error: {exc}")
        finally:
            self.running = False
            self._publish_partial("")
            self.status_callback("Voice input stopped")

    def _publish_partial(self, text: str):
        self._last_partial = text
        try:
            self.partial_callback(text)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        self.running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(None)
            except queue.Empty:
                pass
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.thread = None
        self._clear_queue()



class KokoroTTS:
    """Compatibility facade for Spatial Desktop, backed by StreamTTS/Piper.

    Existing Spatial Desktop call sites keep using speak(), set_voice(), and
    update_settings(); speech is segmented and synthesized by StreamTTS.
    Voice models are downloaded on demand and kept under models/.
    """
    def __init__(self, status_callback=None, activity_callback=None):
        self.status_callback = status_callback or (lambda _text: None)
        self.activity_callback = activity_callback or (lambda _active: None)
        self.settings = load_tts_settings()
        self.enabled = bool(self.settings["enabled"])
        self.status = "StreamTTS starting"
        self._lock = threading.RLock()
        self._engine = None
        self._thread = None
        self._stop = threading.Event()
        self._pending = queue.Queue(maxsize=20)
        self._downloading = False
        self._download_error = ""
        self._worker = threading.Thread(target=self._run, name="StreamTTS-output", daemon=True)
        self._worker.start()

    @property
    def ready(self):
        return self.enabled and not self._stop.is_set()

    @property
    def model_files_present(self):
        voice = str(self.settings["voice"])
        if voice.startswith("system:"):
            return True
        folder = TTS_MODELS_DIR / voice
        return folder.is_dir() and any(folder.glob("*.onnx")) and (folder / "tokens.txt").is_file()

    def _set_status(self, message):
        self.status = str(message)
        try:
            self.status_callback(self.status)
        except Exception:
            pass

    def save_settings(self):
        try:
            TTS_SETTINGS.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def update_settings(self, *, enabled=None, voice=None, speed=None, pitch=None, mode=None, language=None, persist=True):
        changed_voice = False
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
                self.settings["enabled"] = self.enabled
            if isinstance(voice, str) and voice and voice != self.settings["voice"]:
                self.settings["voice"] = voice
                changed_voice = True
            if speed is not None:
                try:
                    value = float(speed)
                    if 0.5 <= value <= 2.0:
                        self.settings["speed"] = round(value, 2)
                        if self._engine is not None:
                            self._engine.set_speed(value)
                except (TypeError, ValueError, RuntimeError):
                    pass
            if pitch is not None:
                try:
                    value = float(pitch)
                    if -6.0 <= value <= 6.0:
                        self.settings["pitch"] = value
                        if self._engine is not None and hasattr(self._engine, "set_pitch"):
                            self._engine.set_pitch(value)
                except (TypeError, ValueError, RuntimeError):
                    pass
            if mode in ("immediate", "balanced", "natural"):
                self.settings["mode"] = mode
                if self._engine is not None and hasattr(self._engine, "set_mode"):
                    self._engine.set_mode(mode)
            if language in ("en-gb", "en-us"):
                self.settings["language"] = language
            if changed_voice:
                old, self._engine = self._engine, None
                if old is not None:
                    threading.Thread(target=self._close_engine, args=(old,), daemon=True).start()
        if persist:
            self.save_settings()
        if self.enabled:
            self._set_status("StreamTTS ready" if self.model_files_present else "StreamTTS downloading voice model…")

    def set_enabled(self, enabled):
        self.update_settings(enabled=enabled)

    def set_voice(self, voice):
        self.update_settings(voice=voice)

    def set_speed(self, speed):
        self.update_settings(speed=speed)

    def set_pitch(self, pitch):
        self.update_settings(pitch=pitch)

    def set_mode(self, mode):
        self.update_settings(mode=mode)

    def _close_engine(self, engine):
        try:
            engine.close()
        except Exception:
            pass

    def _download_model(self, voice):
        import tarfile
        import urllib.request
        import shutil
        import tempfile

        TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        archive = TTS_MODELS_DIR / (voice + ".tar.bz2")
        temp = archive.with_suffix(archive.suffix + ".part")
        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{voice}.tar.bz2"
        try:
            self._set_status(f"Downloading StreamTTS voice: {voice}…")
            with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as dest:
                shutil.copyfileobj(response, dest, length=1024 * 1024)
            temp.replace(archive)
            base = TTS_MODELS_DIR.resolve()
            with tarfile.open(archive, "r:bz2") as tf:
                for member in tf.getmembers():
                    target = (TTS_MODELS_DIR / member.name).resolve()
                    if target != base and base not in target.parents:
                        raise RuntimeError("Unsafe path in voice model archive")
                    if member.issym() or member.islnk():
                        raise RuntimeError("Voice model archive contains a link")
                tf.extractall(TTS_MODELS_DIR)
        finally:
            temp.unlink(missing_ok=True)
        folder = TTS_MODELS_DIR / voice
        if not (folder / "tokens.txt").is_file() or not list(folder.glob("*.onnx")):
            # Archives occasionally wrap files in a differently named directory.
            matches = list(TTS_MODELS_DIR.rglob("tokens.txt"))
            folder = next((p.parent for p in matches if list(p.parent.glob("*.onnx")) and p.parent.name == voice), folder)
        if not (folder / "tokens.txt").is_file() or not list(folder.glob("*.onnx")):
            raise RuntimeError(f"Downloaded model {voice} is missing its ONNX model or tokens.txt")
        return folder

    def _ensure_engine(self, voice):
        with self._lock:
            if self._engine is not None and self._engine_voice == voice:
                return self._engine
        is_system_voice = str(voice).startswith("system:")
        folder = TTS_MODELS_DIR / voice
        if not is_system_voice and not (folder.is_dir() and list(folder.glob("*.onnx")) and (folder / "tokens.txt").is_file()):
            folder = self._download_model(voice)
        import sys
        python_dir = str(STREAMTTS_DIR / "python")
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
        voice = str(voice)
        if voice.startswith("system:"):
            from system_voice import SystemVoiceTTS
            engine = SystemVoiceTTS(
                voice[len("system:"):],
                mode=self.settings.get("mode", "balanced"),
                speed=float(self.settings["speed"]),
                pitch_semitones=float(self.settings.get("pitch", 0.0)),
            )
        else:
            from streamtts_python import StreamingTTS
            engine = StreamingTTS(
                folder, mode=self.settings.get("mode", "balanced"), num_threads=2,
                speed=float(self.settings["speed"]),
                pitch_semitones=float(self.settings.get("pitch", 0.0)),
            )
        with self._lock:
            old = self._engine
            self._engine = engine
            self._engine_voice = voice
        if old is not None:
            threading.Thread(target=self._close_engine, args=(old,), daemon=True).start()
        self._set_status("StreamTTS ready")
        return engine

    def speak(self, text):
        text = " ".join(str(text).split())[:1000]
        if not text or not self.enabled or self._stop.is_set():
            return False
        try:
            self._pending.put_nowait(text)
        except queue.Full:
            try:
                self._pending.get_nowait()
                self._pending.put_nowait(text)
            except queue.Full:
                return False
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                text = self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            if text is None:
                self._pending.task_done()
                break
            try:
                if not self.enabled:
                    continue
                voice = self.settings["voice"]
                engine = self._ensure_engine(voice)
                self.activity_callback(True)
                self._set_status("Speaking with StreamTTS…")
                engine.push_text(text)
                engine.flush()
                engine.wait_until_idle(timeout=120)
                self._set_status("StreamTTS ready")
            except Exception as exc:
                self._set_status(f"StreamTTS unavailable: {exc}")
            finally:
                try:
                    self.activity_callback(False)
                except Exception:
                    pass
                self._pending.task_done()

    def close(self):
        self._stop.set()
        try:
            self._pending.put_nowait(None)
        except queue.Full:
            try:
                self._pending.get_nowait()
                self._pending.put_nowait(None)
            except queue.Empty:
                pass
        engine = self._engine
        if engine is not None:
            self._close_engine(engine)
            self._engine = None
        if self._worker.is_alive() and self._worker is not threading.current_thread():
            self._worker.join(timeout=1.0)
