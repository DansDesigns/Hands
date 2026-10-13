"""Opt-in local face welcome-back monitor for Spatial Desktop.

Uses OpenCV Zoo YuNet + quantized SFace ONNX models. Camera frames and face
embeddings stay on this machine; only a numeric embedding and chosen display
name are saved. No face photos or video are written to disk.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Must be set before main.py imports cv2. The OpenCV Zoo dynamic YuNet model
# is intended for OpenCV 5.x ONNX Runtime when input resolution changes.
os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")

ROOT = Path(__file__).resolve().parent
# Prefer a user-writable cache, while still accepting models downloaded by older
# installers into the project's models/face directory.
PROFILE_DIR = Path.home() / ".spatial_desktop"
PROFILE_FILE = PROFILE_DIR / "face_profile.json"
FACE_MODELS = PROFILE_DIR / "models" / "face"
LEGACY_FACE_MODELS = ROOT / "models" / "face"
DETECTOR_FILENAME = "face_detection_yunet_2026may.onnx"
RECOGNIZER_FILENAME = "face_recognition_sface_2021dec_int8.onnx"
# Compatibility aliases for external tooling; the service resolves writable and
# legacy locations dynamically rather than assuming these files already exist.
DETECTOR_MODEL = LEGACY_FACE_MODELS / DETECTOR_FILENAME
RECOGNIZER_MODEL = LEGACY_FACE_MODELS / RECOGNIZER_FILENAME
WELCOME_ABSENCE_SECONDS = 30 * 60
COSINE_THRESHOLD = 0.40
ENROLL_REQUIRED_SAMPLES = 8
ENROLL_TIMEOUT_SECONDS = 35.0
MODEL_PREPARE_TIMEOUT_SECONDS = 180.0
MODEL_RETRY_SECONDS = 30.0

FACE_MODEL_SPECS = {
    DETECTOR_FILENAME: {
        "urls": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
        ),
        "min_size": 100_000,
    },
    RECOGNIZER_FILENAME: {
        "urls": (
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx?download=true",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
        ),
        "min_size": 5_000_000,
    },
}


def _valid_model_file(path: Path, minimum_size: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < minimum_size:
            return False
        # Git LFS can return a tiny pointer file when a raw file URL is wrong.
        with path.open("rb") as stream:
            head = stream.read(64)
        return b"git-lfs.github.com/spec/v1" not in head
    except OSError:
        return False


def find_face_model_pair():
    """Return (detector, recognizer) from user cache or legacy project location."""
    for directory in (FACE_MODELS, LEGACY_FACE_MODELS):
        detector = directory / DETECTOR_FILENAME
        recognizer = directory / RECOGNIZER_FILENAME
        if (_valid_model_file(detector, FACE_MODEL_SPECS[DETECTOR_FILENAME]["min_size"])
                and _valid_model_file(recognizer, FACE_MODEL_SPECS[RECOGNIZER_FILENAME]["min_size"])):
            return detector, recognizer
    return None


def download_face_models(target_dir: Optional[Path] = None,
                         status_callback: Optional[Callable[[str], None]] = None) -> tuple[Path, Path]:
    """Download the optional models to a user-writable folder, validating each file.

    It is safe to call repeatedly: already-valid files are kept and downloads use
    temporary files before replacing their destination.
    """
    import urllib.error

    target_dir = Path(target_dir or FACE_MODELS)
    status_callback = status_callback or (lambda _message: None)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, spec in FACE_MODEL_SPECS.items():
        target = target_dir / filename
        if _valid_model_file(target, spec["min_size"]):
            continue
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        last_error = None
        downloaded = False
        for url in spec["urls"]:
            temp = target.with_suffix(target.suffix + ".part")
            try:
                status_callback(f"Downloading face model: {filename}")
                request = urllib.request.Request(url, headers={"User-Agent": "SpatialDesktop/1.0"})
                with urllib.request.urlopen(request, timeout=25) as response, temp.open("wb") as output:
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    count = 0
                    last_progress_bucket = -1
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        count += len(block)
                        # Keep GUI status useful without flooding the event queue.
                        if total:
                            progress_bucket = min(3, (count * 4) // total)
                            if progress_bucket > last_progress_bucket and progress_bucket < 4:
                                status_callback(f"Downloading {filename}: {progress_bucket * 25}%")
                                last_progress_bucket = progress_bucket
                if not _valid_model_file(temp, spec["min_size"]):
                    raise RuntimeError("download was incomplete or returned a Git LFS pointer instead of the model")
                os.replace(temp, target)
                downloaded = True
                break
            except Exception as exc:
                last_error = exc
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
        if not downloaded:
            raise RuntimeError(f"Could not download {filename}: {last_error}")
    (target_dir / "README.txt").write_text(
        "Optional local Face Welcome models from OpenCV Zoo.\n"
        "Face recognition remains opt-in. No raw camera frames are stored.\n",
        encoding="utf-8",
    )
    return target_dir / DETECTOR_FILENAME, target_dir / RECOGNIZER_FILENAME


class FaceWelcomeService:
    """Low-frequency face matching. Inert until a profile exists or enrollment starts."""
    def __init__(self, frame_provider: Callable[[], object],
                 status_callback: Optional[Callable[[str], None]] = None,
                 welcome_callback: Optional[Callable[[str], None]] = None):
        self.frame_provider = frame_provider
        self.status_callback = status_callback or (lambda _message: None)
        self.welcome_callback = welcome_callback or (lambda _message: None)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._detector = None
        self._recognizer = None
        self._detector_model_path: Optional[Path] = None
        self._recognizer_model_path: Optional[Path] = None
        self._model_error_reported = False
        self._last_model_error_at = 0.0
        self._next_model_attempt_at = 0.0
        self._prepare_deadline = 0.0
        self._profile = self._read_profile()
        self._profile_name = str(self._profile.get("username", "")).strip() if self._profile else ""
        self._last_seen_epoch = self._profile_last_seen(self._profile)
        self._last_profile_write = 0.0
        self._startup_monotonic = time.monotonic()
        self._present = False
        self._match_streak = 0
        self._collect_name: Optional[str] = None
        self._collect_deadline = 0.0
        self._collected: list[np.ndarray] = []
        self._last_sample_at = 0.0
        self._last_status_at = 0.0

    def _read_profile(self):
        try:
            raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            name = str(raw.get("username", "")).strip()
            emb = np.asarray(raw.get("embedding", []), dtype=np.float32).reshape(-1)
            if not name or emb.size < 32 or not np.isfinite(emb).all():
                return None
            norm = float(np.linalg.norm(emb))
            if norm <= 1e-6:
                return None
            raw["username"] = name[:48]
            raw["embedding"] = (emb / norm).tolist()
            return raw
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _profile_last_seen(profile):
        try:
            value = float(profile.get("last_seen_epoch", 0.0)) if profile else 0.0
            return value if value > 0 else time.time()
        except (TypeError, ValueError):
            return time.time()

    def profile_name(self) -> str:
        with self._lock:
            return self._profile_name

    def has_profile(self) -> bool:
        with self._lock:
            return bool(self._profile and self._profile_name)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="LocalFaceWelcome", daemon=True)
        self._thread.start()

    def begin_enrollment(self, username: str) -> bool:
        name = " ".join(str(username).split()).strip()[:48]
        if not name:
            self.status_callback("Face enrollment cancelled: enter a display name.")
            return False
        with self._lock:
            self._collect_name = name
            # Start the sample timeout only after required model files are ready.
            # First-time downloads happen in the face worker, never in the GUI thread.
            self._collect_deadline = 0.0
            self._prepare_deadline = time.monotonic() + MODEL_PREPARE_TIMEOUT_SECONDS
            self._collected = []
            self._last_sample_at = 0.0
            self._match_streak = 0
            self._present = False
        self._wake.set()
        self.start()
        self.status_callback("Preparing local face recognition automatically. First setup needs internet; Spatial Desktop will stay running.")
        return True

    def remove_profile(self):
        with self._lock:
            self._profile = None
            self._profile_name = ""
            self._last_seen_epoch = time.time()
            self._present = False
            self._match_streak = 0
            self._collect_name = None
            self._collect_deadline = 0.0
            self._prepare_deadline = 0.0
            self._collected.clear()
        try:
            PROFILE_FILE.unlink(missing_ok=True)
        except OSError as exc:
            self.status_callback(f"Could not remove local face profile: {exc}")
            return False
        self.status_callback("Local face welcome profile removed.")
        return True

    def _ensure_models(self):
        if self._detector is not None and self._recognizer is not None:
            return True
        if time.monotonic() < self._next_model_attempt_at:
            return False
        pair = find_face_model_pair()
        if pair is None:
            now = time.monotonic()
            if now < self._next_model_attempt_at:
                return False
            self._next_model_attempt_at = now + MODEL_RETRY_SECONDS
            try:
                self.status_callback("Face models aren't installed yet. Downloading them automatically…")
                pair = download_face_models(FACE_MODELS, self.status_callback)
            except Exception as exc:
                # Enrollment and the whole desktop continue running. A later retry
                # is automatic; users are not asked to run a helper script.
                if now - self._last_model_error_at >= MODEL_RETRY_SECONDS:
                    self.status_callback(f"Automatic face-model setup failed; will retry. Check your internet connection. ({exc})")
                    self._last_model_error_at = now
                return False
        detector_path, recognizer_path = pair
        try:
            import cv2
            detector_factory = getattr(cv2, "FaceDetectorYN_create", None)
            if detector_factory is None:
                detector_factory = cv2.FaceDetectorYN.create
            recognizer_factory = getattr(cv2, "FaceRecognizerSF_create", None)
            if recognizer_factory is None:
                recognizer_factory = cv2.FaceRecognizerSF.create
            # OpenCV 5's current YuNet model supports changing the input size.
            self._detector = detector_factory(str(detector_path), "", (480, 270), 0.80, 0.30, 5000)
            self._recognizer = recognizer_factory(str(recognizer_path), "")
            self._detector_model_path = detector_path
            self._recognizer_model_path = recognizer_path
            self._model_error_reported = False
            self._next_model_attempt_at = 0.0
            self.status_callback("Face models ready. You can look at the camera to enrol.")
            return True
        except Exception as exc:
            self._detector = None
            self._recognizer = None
            self._next_model_attempt_at = time.monotonic() + MODEL_RETRY_SECONDS
            if not self._model_error_reported:
                self.status_callback(f"Local face recognition unavailable: {exc}. The desktop will keep running.")
                self._model_error_reported = True
            return False

    def _extract_embedding(self, frame):
        import cv2
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        height, width = frame.shape[:2]
        try:
            self._detector.setInputSize((int(width), int(height)))
        except Exception:
            # Some OpenCV builds want the detector to be recreated when the input changes.
            model_path = str(self._detector_model_path or DETECTOR_MODEL)
            self._detector = cv2.FaceDetectorYN.create(model_path, "", (int(width), int(height)), 0.80, 0.30, 5000)
        _retval, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        # Prefer the largest visible face; only one enrolled person is supported initially.
        face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
        if float(face[2]) < 30 or float(face[3]) < 30:
            return None
        aligned = self._recognizer.alignCrop(frame, face)
        embedding = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size < 32 or norm <= 1e-6 or not np.isfinite(embedding).all():
            return None
        return embedding / norm

    def _write_profile(self, feature: np.ndarray, username: str, last_seen_epoch: float):
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                PROFILE_DIR.chmod(0o700)
            except OSError:
                pass
        profile = {
            "schema_version": 1,
            "username": username,
            "embedding": [float(value) for value in feature.reshape(-1)],
            "last_seen_epoch": float(last_seen_epoch),
            "storage_note": "Local numeric face embedding only; no face images are stored.",
        }
        temp = PROFILE_FILE.with_suffix(".json.tmp")
        temp.write_text(json.dumps(profile, separators=(",", ":")), encoding="utf-8")
        if os.name != "nt":
            try:
                temp.chmod(0o600)
            except OSError:
                pass
        os.replace(temp, PROFILE_FILE)
        with self._lock:
            self._profile = profile
            self._profile_name = username
            self._last_seen_epoch = float(last_seen_epoch)
            self._last_profile_write = time.monotonic()

    def _finish_enrollment(self):
        with self._lock:
            name = self._collect_name
            features = list(self._collected)
            self._collect_name = None
            self._collected = []
        if not name or len(features) < ENROLL_REQUIRED_SAMPLES:
            self.status_callback("Face enrollment failed — not enough clear face samples. Try again with better lighting.")
            return
        mean = np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-6:
            self.status_callback("Face enrollment failed — samples were inconsistent. Please try again.")
            return
        mean /= norm
        try:
            self._write_profile(mean, name, time.time())
            with self._lock:
                self._present = True
                self._match_streak = 0
            self.status_callback(f"Face welcome enabled for {name}. A local face template was saved; no photos were retained.")
        except OSError as exc:
            self.status_callback(f"Could not save local face profile: {exc}")

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                has_profile = bool(self._profile)
                enrolling = self._collect_name is not None
                deadline = self._collect_deadline
                name = self._collect_name
            if not has_profile and not enrolling:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            now_mono = time.monotonic()
            if not self._ensure_models():
                with self._lock:
                    enrolling_now = self._collect_name is not None
                    prepare_deadline = self._prepare_deadline
                if enrolling_now and prepare_deadline and now_mono > prepare_deadline:
                    with self._lock:
                        self._collect_name = None
                        self._collected = []
                        self._collect_deadline = 0.0
                        self._prepare_deadline = 0.0
                    self.status_callback("Face setup did not complete. Check the status above, then press Enrol Face to retry; Spatial Desktop is still running.")
                self._stop.wait(1.0)
                continue
            # Begin the sample collection window only after downloads and model
            # initialisation have succeeded, avoiding the old 35-second race.
            with self._lock:
                if self._collect_name is not None and self._collect_deadline <= 0.0:
                    self._collect_deadline = time.monotonic() + ENROLL_TIMEOUT_SECONDS
                    self._prepare_deadline = 0.0
                    start_collecting = True
                else:
                    start_collecting = False
                deadline = self._collect_deadline
                enrolling = self._collect_name is not None
                name = self._collect_name
            if start_collecting:
                self.status_callback("Face models ready — enrollment started. Look toward the camera in even lighting.")
            frame = self.frame_provider()
            now_mono = time.monotonic()
            if enrolling and deadline > 0.0 and now_mono > deadline:
                self._finish_enrollment()
                continue
            if frame is None:
                self._stop.wait(0.25)
                continue
            if enrolling:
                if now_mono > deadline:
                    self._finish_enrollment()
                elif now_mono-self._last_sample_at >= 0.65:
                    try:
                        feature = self._extract_embedding(frame)
                    except Exception as exc:
                        self.status_callback(f"Face enrollment sample failed: {exc}")
                        feature = None
                    self._last_sample_at = now_mono
                    if feature is not None:
                        with self._lock:
                            if self._collect_name is not None:
                                self._collected.append(feature)
                                count = len(self._collected)
                                current_name = self._collect_name
                            else:
                                count = 0
                                current_name = name or "user"
                        self.status_callback(f"Enrolling {current_name}: {count}/{ENROLL_REQUIRED_SAMPLES} clear samples")
                        if count >= ENROLL_REQUIRED_SAMPLES:
                            self._finish_enrollment()
                self._stop.wait(0.08)
                continue

            # Recognition runs at roughly 1 Hz to keep CPU use low and does not store frames.
            try:
                feature = self._extract_embedding(frame)
            except Exception as exc:
                self.status_callback(f"Face check skipped: {exc}")
                feature = None
            now_epoch = time.time()
            if feature is None:
                with self._lock:
                    self._match_streak = 0
                    self._present = False
            else:
                with self._lock:
                    profile = self._profile
                    expected = np.asarray(profile.get("embedding", []), dtype=np.float32) if profile else np.empty(0)
                if expected.size == feature.size and expected.size:
                    similarity = float(np.dot(feature, expected) / max(1e-6, float(np.linalg.norm(expected))))
                else:
                    similarity = -1.0
                if similarity >= COSINE_THRESHOLD:
                    with self._lock:
                        self._match_streak += 1
                        streak = self._match_streak
                        was_present = self._present
                        elapsed = now_epoch - self._last_seen_epoch
                        username = self._profile_name
                        should_welcome = (not was_present and streak >= 2 and elapsed >= WELCOME_ABSENCE_SECONDS
                                          and now_mono-self._startup_monotonic >= 60.0)
                        if streak >= 2:
                            self._present = True
                            self._last_seen_epoch = now_epoch
                    if should_welcome and username:
                        self.welcome_callback(f"Welcome back, {username}.")
                        self.status_callback(f"Welcome back, {username}.")
                else:
                    with self._lock:
                        self._match_streak = 0
                        self._present = False

            with self._lock:
                should_write = self._profile is not None and self._present and now_mono-self._last_profile_write >= 30.0
                current_profile = dict(self._profile) if self._profile else None
                current_name = self._profile_name
                last_seen = self._last_seen_epoch
            if should_write and current_profile:
                try:
                    self._write_profile(np.asarray(current_profile["embedding"], dtype=np.float32), current_name, last_seen)
                except OSError:
                    pass
            self._stop.wait(1.0)

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)
        # Preserve a recent last-seen timestamp for the next launch, but don't require
        # recognition on startup when this optional feature isn't configured.
        with self._lock:
            profile = dict(self._profile) if self._profile else None
            name = self._profile_name
            last_seen = self._last_seen_epoch
        if profile and name:
            try:
                self._write_profile(np.asarray(profile["embedding"], dtype=np.float32), name, last_seen)
            except OSError:
                pass
