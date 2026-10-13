"""Optional offline speech services for Spatial Desktop.

Speech recognition stays on Vosk. Kokoro TTS is optional and only loads its
model when the local model files and `kokoro-onnx` package are available.
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
TTS_DIR = ROOT / "models" / "tts"
TTS_MODEL = TTS_DIR / "kokoro-v1.0.onnx"
TTS_VOICES = TTS_DIR / "voices-v1.0.bin"
TTS_SETTINGS = ROOT / "voice_settings.json"

try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    import vosk
except Exception:
    vosk = None
try:
    from kokoro_onnx import Kokoro
except Exception:
    Kokoro = None


def load_tts_settings() -> dict:
    defaults = {"enabled": True, "voice": "bf_emma", "speed": 1.0, "language": "en-gb"}
    try:
        supplied = json.loads(TTS_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return defaults
    result = defaults.copy()
    if isinstance(supplied, dict):
        result["enabled"] = bool(supplied.get("enabled", defaults["enabled"]))
        voice = supplied.get("voice", defaults["voice"])
        if isinstance(voice, str) and voice in {
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "af_heart", "af_bella", "af_sarah", "am_adam", "am_michael",
        }:
            result["voice"] = voice
        try:
            speed = float(supplied.get("speed", defaults["speed"]))
            if speed == speed and 0.7 <= speed <= 1.5:
                result["speed"] = speed
        except (TypeError, ValueError):
            pass
        lang = supplied.get("language", defaults["language"])
        if lang in ("en-gb", "en-us"):
            result["language"] = lang
    return result


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
    """Optional, local Kokoro ONNX TTS with queued, non-blocking synthesis."""
    def __init__(self, status_callback: Optional[Callable[[str], None]] = None,
                 activity_callback: Optional[Callable[[bool], None]] = None):
        self.status_callback = status_callback or (lambda _text: None)
        self.activity_callback = activity_callback or (lambda _active: None)
        self.settings = load_tts_settings()
        self.enabled = bool(self.settings["enabled"])
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._engine_lock = threading.Lock()
        self._engine_ready = threading.Event()
        self._error_reported = False
        self._prewarm_thread = None
        # Start loading the small local model immediately in the background. The first
        # spoken reply no longer pays the full ONNX/model initialisation cost.
        if self.ready:
            self._prewarm_thread = threading.Thread(target=self._preload_engine, name="Kokoro-preload", daemon=True)
            self._prewarm_thread.start()

    @property
    def model_files_present(self) -> bool:
        return TTS_MODEL.is_file() and TTS_VOICES.is_file()

    @property
    def ready(self) -> bool:
        return bool(self.enabled and Kokoro is not None and sd is not None and self.model_files_present)

    @property
    def status(self) -> str:
        if not self.enabled:
            return "TTS disabled in voice_settings.json"
        if Kokoro is None:
            return "TTS package missing; install requirements-voice.txt"
        if sd is None:
            return "TTS audio output unavailable; install sounddevice"
        if not self.model_files_present:
            return "TTS model missing; run install_voice_models.py"
        return "Kokoro TTS ready (local/offline)"

    def save_settings(self):
        try:
            TTS_SETTINGS.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def update_settings(self, *, enabled=None, voice=None, speed=None, language=None, persist=True):
        voices = {"bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "af_heart", "af_bella", "af_sarah", "am_adam", "am_michael"}
        if enabled is not None:
            self.enabled = bool(enabled)
            self.settings["enabled"] = self.enabled
        if voice in voices:
            self.settings["voice"] = voice
        if speed is not None:
            try:
                value = float(speed)
                if 0.7 <= value <= 1.5:
                    self.settings["speed"] = round(value, 2)
            except (TypeError, ValueError):
                pass
        if language in ("en-gb", "en-us"):
            self.settings["language"] = language
        if persist:
            self.save_settings()
        # If TTS was disabled at startup, enabling it in Settings should also warm
        # the engine in the background rather than making the next phrase wait.
        if self.enabled and self.ready and self._engine is None and (self._prewarm_thread is None or not self._prewarm_thread.is_alive()):
            self._prewarm_thread = threading.Thread(target=self._preload_engine, name="Kokoro-preload", daemon=True)
            self._prewarm_thread.start()

    def set_enabled(self, enabled: bool):
        self.update_settings(enabled=enabled)

    def set_voice(self, voice: str):
        self.update_settings(voice=voice)

    def set_speed(self, speed: float):
        self.update_settings(speed=speed)

    def _ensure_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from kokoro_onnx import Kokoro as KokoroEngine
                self._engine = KokoroEngine(str(TTS_MODEL), str(TTS_VOICES))
                self._engine_ready.set()
            return self._engine

    def _preload_engine(self):
        try:
            self._ensure_engine()
        except Exception as exc:
            self._engine_ready.set()
            if not self._error_reported:
                self._error_reported = True
                self.status_callback(f"TTS preload failed: {exc}")

    def speak(self, text: str) -> bool:
        text = " ".join(str(text).split())
        if not text or not self.ready or self._stop.is_set():
            return False
        # Keep spoken replies short and responsive on lower-end CPUs.
        text = text[:500]
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(text)
            except (queue.Empty, queue.Full):
                return False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="Kokoro-TTS", daemon=True)
            self._thread.start()
        return True

    def _run(self):
        active = False
        try:
            while not self._stop.is_set():
                try:
                    text = self._queue.get(timeout=0.5)
                except queue.Empty:
                    break
                if text is None:
                    break
                if not active:
                    active = True
                    try:
                        self.activity_callback(True)
                    except Exception:
                        pass
                    self.status_callback("Speaking locally…")
                try:
                    # Engine is usually already warm from the background preload thread.
                    engine = self._ensure_engine()
                    audio, sample_rate = engine.create(
                        text, voice=self.settings["voice"], speed=float(self.settings["speed"]),
                        lang=self.settings["language"]
                    )
                    if not self._stop.is_set():
                        sd.play(audio, sample_rate)
                        sd.wait()
                except Exception as exc:
                    if not self._error_reported:
                        self._error_reported = True
                        self.status_callback(f"TTS unavailable: {exc}")
                if self._queue.empty() and active:
                    active = False
                    try:
                        self.activity_callback(False)
                    except Exception:
                        pass
                    self.status_callback("TTS ready")
        finally:
            if active:
                try:
                    self.activity_callback(False)
                except Exception:
                    pass
            self._thread = None

    def close(self):
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except queue.Empty:
                pass
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        prewarm = self._prewarm_thread
        if prewarm and prewarm.is_alive() and prewarm is not threading.current_thread():
            prewarm.join(timeout=0.3)
