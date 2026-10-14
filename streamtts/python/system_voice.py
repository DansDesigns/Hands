"""System-installed voice backend using pyttsx3 (SAPI5 on Windows).

This backend uses voices provided by the operating system, not ONNX models.
It shares the StreamTTS text segmenter and push_text/flush interface.
Pitch shifting is not available through the native pyttsx3/SAPI interface.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Deque, Optional
from collections import deque

from streamtts_python import TextSegmenter


class SystemVoiceTTS:
    """Incremental speech using a voice installed in the operating system."""

    supports_pitch = False

    def __init__(
        self,
        voice_id: str,
        *,
        mode: str = "balanced",
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> None:
        self.voice_id = voice_id
        self._settings_lock = threading.Lock()
        self._speed = float(speed)
        self._pitch_semitones = float(pitch_semitones)
        self._segmenter = TextSegmenter(mode)
        self._segmenter_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._errors: Deque[str] = deque(maxlen=16)
        self._error_lock = threading.Lock()
        self._ready = threading.Event()
        self._init_error: Optional[str] = None
        self._worker = threading.Thread(target=self._run, name="StreamTTS-SystemVoice", daemon=True)
        self._worker.start()
        if not self._ready.wait(20.0):
            self._queue.put(("stop", None))
            raise RuntimeError("Timed out initializing the operating-system speech engine")
        if self._init_error:
            self._worker.join(timeout=1.0)
            raise RuntimeError(self._init_error)

    @property
    def last_error(self) -> str:
        with self._error_lock:
            return self._errors[-1] if self._errors else ""

    def _report_error(self, message: str) -> None:
        with self._error_lock:
            self._errors.append(message)
        print(f"\n[System voice error] {message}", flush=True)

    @staticmethod
    def _rate_for_speed(speed: float) -> int:
        # pyttsx3 rates are approximate words/minute; native engines differ.
        return max(70, min(330, int(round(180 * speed))))

    def _run(self) -> None:
        engine = None
        pythoncom = None
        com_initialized = False
        try:
            if os.name == "nt":
                # pywin32 initializes COM on the main Python thread only. This
                # speech worker owns the SAPI engine, so initialize COM here.
                import pythoncom as _pythoncom
                pythoncom = _pythoncom
                pythoncom.CoInitialize()
                com_initialized = True
            import pyttsx3
            driver = "sapi5" if os.name == "nt" else None
            engine = pyttsx3.init(driverName=driver)
            voices = engine.getProperty("voices") or []
            matched = next((v for v in voices if str(getattr(v, "id", "")) == self.voice_id), None)
            if matched is None:
                available = [str(getattr(v, "name", getattr(v, "id", "voice"))) for v in voices]
                detail = ", ".join(available[:8]) or "none detected"
                raise RuntimeError(f"System voice ID is no longer available. Detected voices: {detail}")
            engine.setProperty("voice", self.voice_id)
            with self._settings_lock:
                speed = self._speed
            engine.setProperty("rate", self._rate_for_speed(speed))
            self._ready.set()
        except Exception as exc:
            self._init_error = (
                f"Could not initialize system voice '{self.voice_id}': {exc}. "
                "On Windows, run setup_windows.bat to install pyttsx3 and its SAPI5 support."
            )
            self._ready.set()
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if com_initialized and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            return

        try:
            while True:
                kind, value = self._queue.get()
                try:
                    if kind == "stop":
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        return
                    if kind == "text":
                        try:
                            engine.say(value)
                            engine.runAndWait()
                        except Exception as exc:
                            self._report_error(f"could not speak {value!r}: {exc}")
                    elif kind == "speed":
                        try:
                            engine.setProperty("rate", self._rate_for_speed(float(value)))
                        except Exception as exc:
                            self._report_error(f"could not change system speech rate: {exc}")
                finally:
                    self._queue.task_done()
        finally:
            if com_initialized and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        with self._settings_lock:
            self._speed = speed
        if not self._closed:
            self._queue.put(("speed", speed))

    def set_pitch(self, semitones: float) -> None:
        # Accepted so the GUI can share one settings interface. Native SAPI
        # voices do not expose pitch; the GUI disables the control for this backend.
        self._pitch_semitones = float(semitones)

    def set_mode(self, mode: str) -> None:
        with self._segmenter_lock:
            if mode not in ("immediate", "balanced", "natural"):
                raise ValueError("mode must be immediate, balanced, or natural")
            defaults = {"immediate": (1, 1), "balanced": (3, 2), "natural": (6, 4)}
            self._segmenter.max_words, self._segmenter.first_words = defaults[mode]
            chunks = self._segmenter._drain(final=False)
        for chunk in chunks:
            self._queue.put(("text", chunk))

    def push_text(self, fragment: str) -> None:
        if self._closed:
            raise RuntimeError("SystemVoiceTTS is closed")
        if not isinstance(fragment, str):
            raise TypeError("fragment must be str")
        with self._segmenter_lock:
            chunks = self._segmenter.push(fragment)
        for chunk in chunks:
            self._queue.put(("text", chunk))

    def pushText(self, fragment: str) -> None:  # noqa: N802
        self.push_text(fragment)

    def flush(self) -> None:
        if self._closed:
            return
        with self._segmenter_lock:
            chunks = self._segmenter.flush()
        for chunk in chunks:
            self._queue.put(("text", chunk))

    def wait_until_idle(self, timeout: float = 120.0) -> None:
        # runAndWait is synchronous on the worker, so queue.join includes
        # synthesis and playback completed by the native speech engine.
        if not self._closed:
            finished = threading.Event()
            timer = threading.Thread(target=lambda: (self._queue.join(), finished.set()), daemon=True)
            timer.start()
            if not finished.wait(timeout):
                raise TimeoutError("Timed out waiting for system speech to finish")

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self.wait_until_idle()
        self._closed = True
        self._queue.put(("stop", None))
        self._worker.join(timeout=5.0)

    def __enter__(self) -> "SystemVoiceTTS":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
