"""Local Liquid AI d1 decision layer using the official llama.cpp System One endpoint.

The application starts and owns llama-server.exe; no manually running server is required.
"""
from __future__ import annotations
import json
import hashlib
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models" / "d1"
SETTINGS_FILE = ROOT / "d1_settings.json"
SERVER_LOG = ROOT / "logs" / "llama-server.log"
SERVER_PORT = 18080
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
MODELS = {
    "600m": {
        "label": "d1-omni-600M Q8 · ~670 MB",
        "repo": "LiquidAI/d1-omni-600M-GGUF",
        "model_file": "d1-omni-600M-Q8_0.gguf",
        "projector_file": "mmproj-d1-omni-600M-Q8_0.gguf",
        "estimated": "about 670 MB",
    },
}
_lock = threading.RLock()
_server_process = None
_loading_server = False
_downloading = False
_error = ""
_ready = False
_server_lock_path = Path(tempfile.gettempdir()) / ("spatial-desktop-d1-" + hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:12] + ".lock")
_owns_server_lock = False


def selected_model() -> str:
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("model", "600m")
        return value if value in MODELS else "600m"
    except Exception:
        return "600m"


def model_paths(model_key: str | None = None) -> tuple[Path, Path]:
    spec = MODELS[model_key or selected_model()]
    folder = MODELS_DIR / (model_key or selected_model())
    return folder / spec["model_file"], folder / spec["projector_file"]


def server_path() -> Path:
    names = [ROOT / "tools" / "llama.cpp" / "llama-server.exe", ROOT / "llama-server.exe"]
    for path in names:
        if path.is_file():
            return path
    return Path("llama-server.exe")


def _download_worker(model_key: str) -> None:
    global _downloading, _error
    success = False
    try:
        import urllib.request
        spec = MODELS[model_key]
        folder = MODELS_DIR / model_key
        folder.mkdir(parents=True, exist_ok=True)
        for filename in (spec["model_file"], spec["projector_file"]):
            target = folder / filename
            if target.exists() and target.stat().st_size > 10_000_000:
                continue
            print(f"[d1] Downloading {filename} from {spec['repo']}...")
            temp = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(f"https://huggingface.co/{spec['repo']}/resolve/main/{filename}", headers={"User-Agent": "SpatialDesktop-d1"})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0") or 0)
                done = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block: break
                    output.write(block); done += len(block)
                    if total and done % (16 * 1024 * 1024) < 1024 * 1024:
                        print(f"[d1] {filename}: {done / total:.0%}")
            temp.replace(target)
        with _lock:
            _error = ""
        success = True
    except Exception as exc:
        with _lock:
            _error = f"Model download failed: {type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _downloading = False
    if success and model_key == selected_model():
        start_loading()


def download_model(model_key: str | None = None) -> bool:
    global _downloading, _error
    key = model_key or selected_model()
    if key not in MODELS:
        return False
    with _lock:
        if _downloading:
            return False
        model, projector = model_paths(key)
        if model.is_file() and projector.is_file() and model.stat().st_size > 10_000_000 and projector.stat().st_size > 10_000_000:
            return True
        _downloading = True
        _error = ""
        threading.Thread(target=_download_worker, args=(key,), name="d1-model-download", daemon=True).start()
        return False


def select_model(model_key: str) -> bool:
    if model_key not in MODELS:
        return False
    SETTINGS_FILE.write_text(json.dumps({"model": model_key}, indent=2) + "\n", encoding="utf-8")
    stop_server()
    model, projector = model_paths(model_key)
    if model.is_file() and projector.is_file():
        return True
    download_model(model_key)
    return False


def _server_ready() -> bool:
    try:
        req = urllib.request.Request(BASE_URL + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as response:
            return response.status == 200
    except Exception:
        return False


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _claim_server_lock() -> bool:
    """Cross-process guard so separate app instances cannot launch duplicate llama servers."""
    global _owns_server_lock
    for _ in range(2):
        try:
            fd = os.open(str(_server_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            _owns_server_lock = True
            return True
        except FileExistsError:
            try:
                owner_pid = int(_server_lock_path.read_text(encoding="ascii").strip())
            except Exception:
                owner_pid = -1
            if _pid_is_alive(owner_pid):
                return False
            try:
                _server_lock_path.unlink(missing_ok=True)
            except OSError:
                return False
    return False


def _release_server_lock() -> None:
    global _owns_server_lock
    if _owns_server_lock:
        try:
            _server_lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        _owns_server_lock = False


def _start_server_worker(model_key: str) -> None:
    global _server_process, _loading_server, _error, _ready
    try:
        # A previous app instance or a user-started server may already own the endpoint.
        if _server_ready():
            with _lock:
                _ready = True
            return
        if not _claim_server_lock():
            # Another Spatial Desktop process is starting the runtime. Wait for it rather
            # than spawning a second CPU-heavy llama-server process.
            deadline = time.time() + 45
            while time.time() < deadline:
                if _server_ready():
                    with _lock:
                        _ready = True
                        _error = ""
                    return
                time.sleep(0.5)
            raise RuntimeError("Another Spatial Desktop instance is starting the d1 runtime. Close duplicate app instances and retry if it remains unavailable.")
        binary = server_path()
        model, projector = model_paths(model_key)
        if not binary.is_file():
            raise FileNotFoundError("llama-server is missing. Run the installer to install the local d1 runtime.")
        if not model.is_file() or not projector.is_file():
            _release_server_lock()
            download_model(model_key)
            return
        # Conservative batch/context settings avoid the excessive CPU/RAM load caused by
        # the former 8192-token batch settings while retaining local inference.
        threads = max(2, min(6, (os.cpu_count() or 4) // 2))
        cmd = [str(binary), "--host", "127.0.0.1", "--port", str(SERVER_PORT),
               "-m", str(model), "--mmproj", str(projector), "--ctx-size", "2048",
               "-b", "512", "-ub", "128", "--threads", str(threads),
               "--threads-batch", str(threads), "--no-webui"]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Preserve llama.cpp startup diagnostics so model/runtime failures are visible
        # instead of silently disappearing into DEVNULL.
        log_handle = SERVER_LOG.open("a", encoding="utf-8", errors="replace")
        log_handle.write(f"\n\n--- llama-server start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_handle.write("Command: " + " ".join(cmd) + "\n")
        log_handle.flush()
        _server_process = subprocess.Popen(cmd, cwd=str(binary.parent), stdout=log_handle,
                                           stderr=subprocess.STDOUT, creationflags=creationflags)
        # Store the child PID once launched so a crashed/stopped server leaves a
        # reclaimable lock even if the owning Spatial Desktop UI is still open.
        try:
            _server_lock_path.write_text(str(_server_process.pid), encoding="ascii")
        except OSError:
            pass
        deadline = time.time() + 180
        while time.time() < deadline:
            if _server_process.poll() is not None:
                raise RuntimeError("llama-server exited before becoming ready. Check available RAM and the d1 model files.")
            if _server_ready():
                with _lock:
                    _ready = True
                    _error = ""
                return
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for the local d1 server to start.")
    except Exception as exc:
        with _lock:
            _error = f"{type(exc).__name__}: {exc}"
            _ready = False
        _release_server_lock()
    finally:
        with _lock:
            _loading_server = False


def _terminate_stale_locked_server() -> None:
    """Stop a leftover llama-server recorded in our lock file before a fresh app launch.

    Only terminate the recorded PID when its executable is demonstrably llama-server;
    never kill an arbitrary process merely because the local port responds.
    """
    try:
        pid = int(_server_lock_path.read_text(encoding="ascii").strip())
    except Exception:
        return
    if pid <= 0 or pid == os.getpid() or not _pid_is_alive(pid):
        try:
            _server_lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    is_llama = False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=3, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            is_llama = "llama-server.exe" in (result.stdout or "").lower()
        else:
            comm = Path(f"/proc/{pid}/comm")
            if comm.exists():
                is_llama = "llama-server" in comm.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        is_llama = False

    if not is_llama:
        # The lock may belong to another live Spatial Desktop instance that is
        # still acquiring the server. Do not break its lock or spawn a duplicate.
        return

    if is_llama:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, text=True, timeout=5, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                os.kill(pid, 15)
                deadline = time.time() + 2
                while time.time() < deadline and _pid_is_alive(pid):
                    time.sleep(0.1)
                if _pid_is_alive(pid):
                    os.kill(pid, 9)
        except Exception:
            pass
    try:
        _server_lock_path.unlink(missing_ok=True)
    except OSError:
        pass

def start_loading(fresh: bool = False) -> bool:
    global _loading_server, _error, _server_process, _ready
    key = selected_model()
    if fresh:
        _terminate_stale_locked_server()
        with _lock:
            _server_process = None
            _ready = False
            _loading_server = False
    # Recover cleanly if a previously started llama-server crashed while the UI stayed open.
    if _server_process is not None and _server_process.poll() is not None:
        _server_process = None
        _ready = False
        _release_server_lock()
    with _lock:
        if _ready and _server_ready():
            return True
        if _loading_server or _downloading:
            return False
        model, projector = model_paths(key)
        if not model.is_file() or not projector.is_file():
            _error = "Model files are missing; downloading " + MODELS[key]["estimated"] + "."
            download_model(key)
            return False
        _loading_server = True
        _error = ""
        threading.Thread(target=_start_server_worker, args=(key,), name="d1-server-start", daemon=True).start()
        return False


def stop_server() -> None:
    global _server_process, _ready, _loading_server
    with _lock:
        proc = _server_process
        _server_process = None
        _ready = False
        _loading_server = False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
    _release_server_lock()


def status() -> dict[str, Any]:
    key = selected_model()
    model, projector = model_paths(key)
    with _lock:
        alive = (_server_process is not None and _server_process.poll() is None) or _server_ready()
        return {"ready": bool(_ready and alive), "loading": _loading_server,
                "downloading": _downloading, "error": _error, "model": key,
                "label": MODELS[key]["label"], "model_exists": model.is_file() and projector.is_file(),
                "server": str(server_path())}


def _decide(state: str, questions: dict) -> dict | None:
    if not start_loading():
        return None
    try:
        payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(BASE_URL + "/v1/systemone", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=8.0) as response:
            result = json.loads(response.read().decode("utf-8"))
        answers = result.get("answers", {}) if isinstance(result, dict) else {}
        return answers if isinstance(answers, dict) else None
    except Exception as exc:
        global _error
        with _lock:
            _error = f"Decision request failed: {type(exc).__name__}: {exc}"
        return None


def classify_voice_command(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    clean = " ".join((text or "").split())
    if len(clean) < 3:
        return None
    criteria = {action: "Voice command meaning: " + ", ".join(phrases[:8]) for action, phrases in aliases.items()}
    if not criteria:
        return None
    answers = _decide("Interpret this recognised English speech as one Spatial Desktop voice command. Choose only if the intent clearly matches an available action; otherwise choose unknown.\nRecognised speech: " + clean,
        {"action": {"type": "choice", "instructions": "Which supported command, if any, is the speaker requesting?",
                     "criteria": {**criteria, "unknown": "No supported command is clearly requested; this may be dictation or a search query."}}})
    answer = (answers or {}).get("action")
    if not isinstance(answer, dict): return None
    action = answer.get("choice")
    try: confidence = float(answer.get("confidence", 0.0))
    except (TypeError, ValueError): confidence = 0.0
    return action if action in aliases and confidence >= 0.78 else None


def rerank_search_results(query: str, results: list) -> list | None:
    if not results: return None
    candidates = results[:8]
    criteria = {}
    for i, item in enumerate(candidates):
        criteria[f"result_{i}"] = f"File/folder name: {getattr(item, 'name', '')}. Location: {getattr(item, 'parent', '')}. Content excerpt: {getattr(item, 'snippet', '')[:180]}"
    answers = _decide("A user is searching their own local files. Choose the single result most relevant to their query. Use only filenames, paths and excerpts shown.\nQuery: " + query,
        {"best_result": {"type": "choice", "instructions": "Which candidate best matches the user's search intent?", "criteria": criteria}})
    answer = (answers or {}).get("best_result")
    if not isinstance(answer, dict): return None
    choice = answer.get("choice")
    try: confidence = float(answer.get("confidence", 0.0))
    except (TypeError, ValueError): confidence = 0.0
    if confidence < 0.48 or not isinstance(choice, str) or not choice.startswith("result_"): return None
    try: idx = int(choice.removeprefix("result_"))
    except ValueError: return None
    if idx < 0 or idx >= len(candidates): return None
    chosen = candidates[idx]
    return [chosen] + [item for item in results if item is not chosen]
