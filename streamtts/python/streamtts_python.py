"""Pure-Python streaming front-end to sherpa-onnx's prebuilt native Python package.

This module provides the same useful push_text/flush/wait_until_idle shape as
our C++ API, but it does not require compiling StreamTTS itself. It is useful on
Windows machines without Visual Studio or a C++ compiler.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import threading
import time
from typing import Deque, List, Optional


def _is_space(ch: str) -> bool:
    return ch.isspace()


def _is_hard_punctuation(ch: str) -> bool:
    return ch in ".!?"


def _is_clause_punctuation(ch: str) -> bool:
    return ch in ",;:"


def _trim(text: str) -> str:
    return text.strip()


class TextSegmenter:
    """Incremental word/chunk segmenter matching the C++ core's policy."""

    def __init__(self, mode: str = "balanced", max_words_per_chunk: int = 0) -> None:
        defaults = {
            "immediate": (1, 1),
            "balanced": (3, 2),
            "natural": (6, 4),
        }
        if mode not in defaults:
            raise ValueError("mode must be immediate, balanced, or natural")
        max_words, first_words = defaults[mode]
        if max_words_per_chunk > 0:
            max_words = max_words_per_chunk
            first_words = min(first_words, max_words)
        self.max_words = max_words
        self.first_words = first_words
        self.emitted_first = False
        self.pending = ""

    def push(self, fragment: str) -> List[str]:
        self.pending += fragment
        return self._drain(final=False)

    def flush(self) -> List[str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> List[str]:
        ready: List[str] = []
        while self.pending:
            target = self.max_words if self.emitted_first else self.first_words
            chunk_end: Optional[int] = None
            consume_end: Optional[int] = None
            words = 0
            in_word = False

            for index, ch in enumerate(self.pending):
                if _is_space(ch):
                    if in_word:
                        words += 1
                        in_word = False
                    if words >= target:
                        chunk_end = index
                        consume_end = index + 1
                        break
                elif _is_hard_punctuation(ch) or _is_clause_punctuation(ch):
                    if in_word:
                        words += 1
                        in_word = False
                    if _is_hard_punctuation(ch) or (_is_clause_punctuation(ch) and words >= 2):
                        chunk_end = index + 1
                        consume_end = index + 1
                        break
                else:
                    in_word = True

            if chunk_end is None and final:
                chunk_end = len(self.pending)
                consume_end = len(self.pending)
            if chunk_end is None:
                break

            chunk = _trim(self.pending[:chunk_end])
            self.pending = self.pending[consume_end:]
            if chunk:
                ready.append(chunk)
                self.emitted_first = True
        return ready


class StreamingTTS:
    """Incremental offline TTS using the prebuilt sherpa-onnx Python wheel.

    push_text() may receive arbitrary fragments (even fragments in the middle
    of a word). Model inference runs on a worker thread while an independent
    sounddevice output stream plays queued PCM.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        mode: str = "balanced",
        num_threads: int = 2,
        speaker_id: int = 0,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        max_words_per_chunk: int = 0,
    ) -> None:
        model_dir = Path(model_dir).resolve()
        model_candidates = sorted(model_dir.glob("*.onnx"))
        if not model_candidates:
            raise FileNotFoundError(
                f"No .onnx TTS model found in {model_dir}. Install a voice from the GUI or run setup_windows.bat."
            )
        # Most sherpa-onnx Piper packages contain a single ONNX voice model.
        model_path = model_candidates[0]
        tokens_path = model_dir / "tokens.txt"
        data_dir = model_dir / "espeak-ng-data"
        if not model_path.is_file() or not tokens_path.is_file():
            raise FileNotFoundError(
                "Voice model files are missing. Run setup_windows.bat or "
                "python scripts/download_model.py first. Expected: "
                f"{model_path} and {tokens_path}"
            )

        try:
            import numpy as np
            import sherpa_onnx
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Python dependencies are missing. Run setup_windows.bat "
                "(or pip install sherpa-onnx sherpa-onnx-bin numpy sounddevice)."
            ) from exc

        self._np = np
        self._sd = sd
        self._speaker_id = int(speaker_id)
        self._settings_lock = threading.Lock()
        self._speed = float(speed)
        self._pitch_semitones = float(pitch_semitones)
        self._segmenter = TextSegmenter(mode, max_words_per_chunk)
        self._segmenter_lock = threading.Lock()
        self._text_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._worker_busy = threading.Event()
        self._errors: Deque[str] = deque(maxlen=16)
        self._error_lock = threading.Lock()
        self._closed = False

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_dir) if data_dir.is_dir() else "",
                ),
                num_threads=max(1, int(num_threads)),
                provider="cpu",
                debug=False,
            ),
            max_num_sentences=1,
        )
        if not config.validate():
            raise RuntimeError("sherpa-onnx rejected the voice configuration; check the model files")
        self._tts = sherpa_onnx.OfflineTts(config)

        self._audio_lock = threading.Lock()
        self._audio_queue: Deque[object] = deque()
        self._active_audio = None
        self._active_pos = 0
        self._audio_queued_samples = 0
        self._output_stream = None
        self._output_started = False
        self._sample_rate: Optional[int] = None

        self._worker = threading.Thread(target=self._synthesis_loop, name="StreamTTS-Python", daemon=True)
        self._worker.start()

    @property
    def last_error(self) -> str:
        with self._error_lock:
            return self._errors[-1] if self._errors else ""

    def _report_error(self, message: str) -> None:
        with self._error_lock:
            self._errors.append(message)
        print(f"\n[TTS error] {message}", flush=True)

    def set_speed(self, speed: float) -> None:
        """Set the synthesis rate for chunks that have not started synthesis yet."""
        speed = float(speed)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        with self._settings_lock:
            self._speed = speed

    def set_pitch(self, semitones: float) -> None:
        """Set pitch shift in semitones (-6 to +6); duration is preserved."""
        semitones = float(semitones)
        if not -6.0 <= semitones <= 6.0:
            raise ValueError("pitch must be between -6 and +6 semitones")
        with self._settings_lock:
            self._pitch_semitones = semitones

    def set_mode(self, mode: str) -> None:
        """Change chunking policy without replacing the loaded voice model."""
        with self._segmenter_lock:
            if mode not in ("immediate", "balanced", "natural"):
                raise ValueError("mode must be immediate, balanced, or natural")
            defaults = {"immediate": (1, 1), "balanced": (3, 2), "natural": (6, 4)}
            max_words, first_words = defaults[mode]
            self._segmenter.max_words = max_words
            self._segmenter.first_words = first_words
            chunks = self._segmenter._drain(final=False)
        for chunk in chunks:
            self._text_queue.put(chunk)

    def push_text(self, fragment: str) -> None:
        if self._closed:
            raise RuntimeError("StreamingTTS is closed")
        if not isinstance(fragment, str):
            raise TypeError("fragment must be str")
        with self._segmenter_lock:
            chunks = self._segmenter.push(fragment)
        for chunk in chunks:
            self._text_queue.put(chunk)

    # Match the spelling in the C++ API for easy adaptation.
    def pushText(self, fragment: str) -> None:  # noqa: N802
        self.push_text(fragment)

    def flush(self) -> None:
        if self._closed:
            return
        with self._segmenter_lock:
            chunks = self._segmenter.flush()
        for chunk in chunks:
            self._text_queue.put(chunk)

    def _synthesis_loop(self) -> None:
        while True:
            text = self._text_queue.get()
            if text is None:
                self._text_queue.task_done()
                break
            self._worker_busy.set()
            try:
                with self._settings_lock:
                    speed = self._speed
                    pitch_semitones = self._pitch_semitones
                audio = self._tts.generate(text=text, sid=self._speaker_id, speed=speed)
                samples = self._np.asarray(audio.samples, dtype=self._np.float32).reshape(-1).copy()
                sample_rate = int(audio.sample_rate)
                if samples.size and abs(pitch_semitones) >= 0.01:
                    try:
                        import librosa
                    except ImportError as exc:
                        raise RuntimeError(
                            "Pitch shifting needs librosa. Run setup_windows.bat or install librosa in this Python environment."
                        ) from exc
                    samples = librosa.effects.pitch_shift(
                        y=samples, sr=sample_rate, n_steps=pitch_semitones
                    ).astype(self._np.float32, copy=False)
                if samples.size:
                    self._enqueue_audio(samples, sample_rate)
            except Exception as exc:
                self._report_error(f"synthesis failed for {text!r}: {exc}")
            finally:
                self._worker_busy.clear()
                self._text_queue.task_done()

    def _enqueue_audio(self, samples, sample_rate: int) -> None:
        need_start = False
        with self._audio_lock:
            if self._sample_rate is None:
                self._sample_rate = sample_rate
            elif self._sample_rate != sample_rate:
                raise RuntimeError(f"Voice sample rate changed from {self._sample_rate} to {sample_rate}")

            self._audio_queue.append(samples)
            self._audio_queued_samples += int(samples.size)
            if self._output_stream is None:
                stream = self._sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=0,
                )
                self._output_stream = stream
                need_start = True
            else:
                stream = self._output_stream

        if need_start:
            try:
                stream.start()
                self._output_started = True
            except Exception as exc:
                with self._audio_lock:
                    self._audio_queue.clear()
                    self._active_audio = None
                    self._active_pos = 0
                    self._audio_queued_samples = 0
                self._report_error(f"could not start audio output: {exc}")

    def _audio_callback(self, outdata, frames, _time_info, _status) -> None:
        outdata.fill(0.0)
        written = 0
        with self._audio_lock:
            while written < frames:
                if self._active_audio is None:
                    if not self._audio_queue:
                        break
                    self._active_audio = self._audio_queue.popleft()
                    self._active_pos = 0
                remaining = int(self._active_audio.size) - self._active_pos
                take = min(frames - written, remaining)
                outdata[written:written + take, 0] = self._active_audio[self._active_pos:self._active_pos + take]
                self._active_pos += take
                written += take
                self._audio_queued_samples -= take
                if self._active_pos >= int(self._active_audio.size):
                    self._active_audio = None
                    self._active_pos = 0

    def wait_until_idle(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        self._text_queue.join()
        while True:
            with self._audio_lock:
                remaining = self._audio_queued_samples
            if remaining <= 0 and not self._worker_busy.is_set() and self._text_queue.unfinished_tasks == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for synthesized audio to finish playing")
            time.sleep(0.02)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self.wait_until_idle()
        self._closed = True
        self._text_queue.put(None)
        self._text_queue.join()
        self._worker.join(timeout=5.0)
        if self._output_stream is not None:
            try:
                if self._output_started:
                    self._output_stream.stop()
                self._output_stream.close()
            except Exception as exc:
                self._report_error(f"error closing audio output: {exc}")
            self._output_stream = None

    def __enter__(self) -> "StreamingTTS":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
