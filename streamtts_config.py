#!/usr/bin/env python3
"""StreamTTS settings GUI for Spatial Desktop, based on the full StreamTTS voice controls."""
from __future__ import annotations
import json
import os
import re
import shutil
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog, simpledialog

ROOT = Path(__file__).resolve().parent
from user_paths import user_file
SETTINGS = user_file("voice_settings.json", ROOT)
MODELS_DIR = ROOT / "models"
STREAMTTS_PY = ROOT / "streamtts" / "python"
if str(STREAMTTS_PY) not in sys.path:
    sys.path.insert(0, str(STREAMTTS_PY))

try:
    from voice_catalog import VOICE_CATALOG
except Exception:
    VOICE_CATALOG = {}

DEFAULTS = {
    "enabled": True,
    "voice": "vits-piper-en_GB-alan-medium",
    "speed": 1.2,
    "pitch": 0.0,
    "mode": "balanced",
    "language": "en-gb",
}
SYSTEM_PREFIX = "system:"
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$")


def installed_voice_dirs():
    found = {}
    if MODELS_DIR.exists():
        for token_file in MODELS_DIR.rglob("tokens.txt"):
            folder = token_file.parent
            if list(folder.glob("*.onnx")):
                found[folder.name] = folder
    return found


def find_model_dir(model_id):
    candidate = MODELS_DIR / model_id
    if candidate.is_dir() and (candidate / "tokens.txt").is_file() and list(candidate.glob("*.onnx")):
        return candidate
    for token_file in MODELS_DIR.rglob("tokens.txt") if MODELS_DIR.exists() else []:
        folder = token_file.parent
        if folder.name == model_id and list(folder.glob("*.onnx")):
            return folder
    return None


def read_settings():
    data = dict(DEFAULTS)
    try:
        raw = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # Support both Spatial Desktop's `voice` key and the original GUI's `voice_id`.
            if "voice" not in raw and "voice_id" in raw:
                raw["voice"] = raw["voice_id"]
            data.update({k: raw[k] for k in data if k in raw})
    except (OSError, ValueError, TypeError):
        pass
    try:
        data["speed"] = max(0.5, min(2.0, float(data["speed"])))
    except (TypeError, ValueError):
        data["speed"] = 1.2
    try:
        data["pitch"] = max(-6.0, min(6.0, float(data["pitch"])))
    except (TypeError, ValueError):
        data["pitch"] = 0.0
    if data["mode"] not in ("immediate", "balanced", "natural"):
        data["mode"] = "balanced"
    voice = str(data.get("voice", DEFAULTS["voice"]))
    # Preserve any voice ID shipped in the original StreamTTS catalogue, even
    # when its model has not been downloaded yet.
    if not (voice in VOICE_CATALOG or voice.startswith("vits-piper-") or
            voice.startswith(SYSTEM_PREFIX) or find_model_dir(voice)):
        data["voice"] = DEFAULTS["voice"]
    return data


class StreamTTSConfig(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("StreamTTS Config — Spatial Desktop")
        self.geometry("690x690")
        self.minsize(590, 620)
        self.settings = read_settings()
        self.system_voices = {}
        self.extra_model_ids = set()
        self._voice_map = {}
        self._discover_system_voices()
        saved = str(self.settings.get("voice", DEFAULTS["voice"]))
        if saved not in VOICE_CATALOG and not saved.startswith(SYSTEM_PREFIX) and MODEL_ID_RE.fullmatch(saved):
            self.extra_model_ids.add(saved)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", padding=2)
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Status.TLabel", foreground="#276749")
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="StreamTTS Config", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Voice models, system voices, speech rate, pitch and streaming behaviour.").pack(anchor="w", pady=(2, 12))

        self.enabled_var = tk.BooleanVar(value=bool(self.settings["enabled"]))
        ttk.Checkbutton(outer, text="Enable StreamTTS speech output", variable=self.enabled_var).pack(anchor="w", pady=(0, 8))

        voice_box = ttk.LabelFrame(outer, text="Voice", padding=10)
        voice_box.pack(fill="x", pady=(0, 8))
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(voice_box, textvariable=self.voice_var, state="readonly")
        self.voice_combo.pack(fill="x", pady=(0, 7))
        self.voice_combo.bind("<<ComboboxSelected>>", self._voice_selected)
        row = ttk.Frame(voice_box)
        row.pack(fill="x")
        ttk.Button(row, text="Apply selected voice", command=self.apply_selected_voice).pack(side="left")
        ttk.Button(row, text="Download other…", command=self.download_other_voice).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Import folder…", command=self.import_voice_folder).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Refresh", command=self.refresh_voices).pack(side="left", padx=(6, 0))

        controls = ttk.LabelFrame(outer, text="Speech settings", padding=10)
        controls.pack(fill="x", pady=(0, 8))
        self.speed_var = tk.DoubleVar(value=float(self.settings["speed"]))
        self.speed_label = tk.StringVar(value=f'{self.speed_var.get():.2f}×')
        ttk.Label(controls, text="Speech rate").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.speed_label).grid(row=0, column=1, sticky="e")
        ttk.Scale(controls, from_=0.5, to=2.0, variable=self.speed_var, command=self._speed_changed).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 9))
        self.pitch_var = tk.DoubleVar(value=float(self.settings["pitch"]))
        self.pitch_label = tk.StringVar(value=f'{self.pitch_var.get():+.1f} semitones')
        ttk.Label(controls, text="Pitch shift").grid(row=2, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.pitch_label).grid(row=2, column=1, sticky="e")
        self.pitch_scale = ttk.Scale(controls, from_=-6.0, to=6.0, variable=self.pitch_var, command=self._pitch_changed)
        self.pitch_scale.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        ttk.Label(controls, text="−6 lower · 0 original · +6 higher").grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 9))
        ttk.Label(controls, text="Streaming mode").grid(row=5, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=self.settings["mode"])
        ttk.Combobox(controls, textvariable=self.mode_var, state="readonly", values=("immediate", "balanced", "natural")).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(3, 4))
        ttk.Label(controls, text="Immediate: shortest chunks · Balanced: responsive + natural · Natural: longer chunks").grid(row=7, column=0, columnspan=2, sticky="w")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=0)

        test_box = ttk.LabelFrame(outer, text="Test voice", padding=10)
        test_box.pack(fill="x", pady=(0, 8))
        self.test_text = tk.StringVar(value="Hello. This is a test of the selected StreamTTS voice.")
        ttk.Entry(test_box, textvariable=self.test_text).pack(fill="x", pady=(0, 7))
        ttk.Button(test_box, text="Test selected voice", command=self.test_voice).pack(anchor="e")

        self.status_var = tk.StringVar(value="Settings are saved for Spatial Desktop. Restart Spatial Desktop to apply them.")
        ttk.Label(outer, textvariable=self.status_var, wraplength=630, style="Status.TLabel").pack(anchor="w", pady=(2, 8))
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", side="bottom")
        ttk.Button(bottom, text="Save settings", command=self.save_settings).pack(side="right")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right", padx=(0, 8))
        self.refresh_voices(select=saved)

    def _discover_system_voices(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            for voice in engine.getProperty("voices") or []:
                voice_id = str(getattr(voice, "id", "") or "")
                if not voice_id:
                    continue
                name = str(getattr(voice, "name", voice_id) or voice_id)
                langs = getattr(voice, "languages", []) or []
                lang = ""
                if langs:
                    first = langs[0]
                    if isinstance(first, bytes):
                        first = first.decode("utf-8", errors="ignore")
                    lang = str(first).strip("\\x00").replace("_", "-")
                self.system_voices[SYSTEM_PREFIX + voice_id] = {"id": voice_id, "name": name, "language": lang}
            try:
                engine.stop()
            except Exception:
                pass
        except Exception:
            self.system_voices = {}

    def refresh_voices(self, select=None):
        installed = installed_voice_dirs()
        ids = list(VOICE_CATALOG)
        for model_id in sorted(set(installed) | self.extra_model_ids):
            if model_id not in ids:
                ids.append(model_id)
        ids.extend(k for k in self.system_voices if k not in ids)
        self._voice_map = {}
        labels = []
        for voice_id in ids:
            if voice_id.startswith(SYSTEM_PREFIX):
                v = self.system_voices.get(voice_id, {})
                label = "System voice — " + v.get("name", voice_id[len(SYSTEM_PREFIX):])
                if v.get("language"):
                    label += f" ({v['language']})"
            else:
                label = VOICE_CATALOG.get(voice_id, {}).get("label", voice_id)
                if voice_id in installed:
                    label += "  [installed]"
                else:
                    label += "  [download on apply]"
            if label in self._voice_map:
                label += f" [{voice_id[-18:]}]"
            self._voice_map[label] = voice_id
        labels = list(self._voice_map)
        self.voice_combo.configure(values=labels)
        target = select or self.settings.get("voice", DEFAULTS["voice"])
        chosen = next((label for label, vid in self._voice_map.items() if vid == target), labels[0] if labels else "")
        self.voice_var.set(chosen)
        self._update_pitch_availability()

    def _voice_selected(self, _event=None):
        self._update_pitch_availability()

    def _selected_id(self):
        return self._voice_map.get(self.voice_var.get(), DEFAULTS["voice"])

    def _update_pitch_availability(self):
        is_system = self._selected_id().startswith(SYSTEM_PREFIX)
        self.pitch_scale.configure(state="disabled" if is_system else "normal")
        if is_system:
            self.pitch_label.set("not supported")
        else:
            self._pitch_changed()

    def _speed_changed(self, _value=None):
        self.speed_label.set(f"{float(self.speed_var.get()):.2f}×")

    def _pitch_changed(self, _value=None):
        self.pitch_label.set(f"{float(self.pitch_var.get()):+.1f} semitones")

    def download_other_voice(self):
        model_id = simpledialog.askstring(
            "Download another voice model",
            "Enter a compatible sherpa-onnx Piper/VITS model ID.\nExample: vits-piper-en_GB-alan-medium",
            parent=self,
        )
        if not model_id:
            return
        model_id = model_id.strip()
        if not MODEL_ID_RE.fullmatch(model_id) or not model_id.startswith("vits-piper-"):
            messagebox.showerror("Invalid model ID", "Use a valid vits-piper model ID from the sherpa-onnx TTS models release.", parent=self)
            return
        self.extra_model_ids.add(model_id)
        self.refresh_voices(select=model_id)
        self.status_var.set("Voice selected. Save settings; Spatial Desktop downloads missing models when speech is first used.")

    def import_voice_folder(self):
        selected = filedialog.askdirectory(title="Choose a TTS model folder (must contain .onnx and tokens.txt)", parent=self)
        if not selected:
            return
        source = Path(selected)
        if not list(source.glob("*.onnx")) or not (source / "tokens.txt").is_file():
            messagebox.showerror("Invalid voice folder", "The selected folder must contain at least one .onnx model and tokens.txt directly inside it.", parent=self)
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", source.name).strip(".-_") or "voice"
        dest = MODELS_DIR / slug
        suffix = 2
        while dest.exists() and dest.resolve() != source.resolve():
            dest = MODELS_DIR / f"{slug}-{suffix}"
            suffix += 1
        if dest.resolve() != source.resolve():
            try:
                shutil.copytree(source, dest)
            except Exception as exc:
                messagebox.showerror("Import voice", str(exc), parent=self)
                return
        self.extra_model_ids.add(dest.name)
        self.refresh_voices(select=dest.name)
        self.status_var.set(f"Imported voice model: {dest.name}. Save settings to use it.")

    def apply_selected_voice(self):
        self.save_settings()
        voice_id = self._selected_id()
        if voice_id.startswith(SYSTEM_PREFIX) or find_model_dir(voice_id):
            self.status_var.set("Voice settings saved. Restart Spatial Desktop to apply them.")
            return
        def worker():
            import tarfile
            import urllib.request
            temp = None
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                archive = MODELS_DIR / (voice_id + ".tar.bz2")
                temp = archive.with_suffix(archive.suffix + ".part")
                url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{voice_id}.tar.bz2"
                self.after(0, lambda: self.status_var.set(f"Downloading voice model {voice_id}…"))
                with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                temp.replace(archive)
                base = MODELS_DIR.resolve()
                with tarfile.open(archive, "r:bz2") as tf:
                    for member in tf.getmembers():
                        target = (MODELS_DIR / member.name).resolve()
                        if target != base and base not in target.parents:
                            raise RuntimeError("Voice archive contains an unsafe path")
                        if member.issym() or member.islnk():
                            raise RuntimeError("Voice archive contains a link")
                    tf.extractall(MODELS_DIR)
                if not find_model_dir(voice_id):
                    raise RuntimeError("Downloaded archive did not contain the expected ONNX model and tokens.txt")
                self.after(0, lambda: (self.refresh_voices(select=voice_id), self.status_var.set("Voice model downloaded. Restart Spatial Desktop to apply it.")))
            except Exception as exc:
                self.after(0, lambda msg=str(exc): self.status_var.set(f"Voice download failed: {msg}"))
            finally:
                try:
                    if temp is not None:
                        temp.unlink(missing_ok=True)
                except Exception:
                    pass
        threading.Thread(target=worker, daemon=True).start()

    def save_settings(self):
        voice_id = self._selected_id()
        settings = {
            "enabled": bool(self.enabled_var.get()),
            "voice": voice_id,
            "speed": round(max(0.5, min(2.0, float(self.speed_var.get()))), 3),
            "pitch": round(max(-6.0, min(6.0, float(self.pitch_var.get()))), 2),
            "mode": self.mode_var.get() if self.mode_var.get() in ("immediate", "balanced", "natural") else "balanced",
            "language": "en-us" if "en_US" in voice_id or "(en-us" in voice_id.lower() else "en-gb",
        }
        try:
            SETTINGS.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save settings", str(exc), parent=self)
            return
        self.settings = settings
        self.status_var.set(f"Saved {SETTINGS.name}. Spatial Desktop will apply these settings before its next spoken response. If speech is already playing, the current utterance finishes with the previous settings.")

    def test_voice(self):
        voice_id = self._selected_id()
        text = self.test_text.get().strip()
        if not text:
            return
        # Preview through the same StreamTTS engine implementation used by Spatial Desktop.
        def worker():
            try:
                if voice_id.startswith(SYSTEM_PREFIX):
                    from system_voice import SystemVoiceTTS
                    engine = SystemVoiceTTS(voice_id[len(SYSTEM_PREFIX):], mode=self.mode_var.get(), speed=float(self.speed_var.get()))
                else:
                    model_dir = find_model_dir(voice_id)
                    if model_dir is None:
                        self.after(0, lambda: self.status_var.set("Voice is not installed yet. Save settings and Spatial Desktop will download it on first use."))
                        return
                    from streamtts_python import StreamingTTS
                    engine = StreamingTTS(model_dir, mode=self.mode_var.get(), num_threads=2,
                                          speed=float(self.speed_var.get()), pitch_semitones=float(self.pitch_var.get()))
                try:
                    engine.push_text(text)
                    engine.flush()
                    engine.wait_until_idle(timeout=120)
                finally:
                    engine.close()
                self.after(0, lambda: self.status_var.set("Voice test completed."))
            except Exception as exc:
                msg = str(exc)
                if "needs librosa" in msg.lower() or "no module named 'librosa'" in msg.lower():
                    msg += " Install it with: python -m pip install librosa"
                self.after(0, lambda msg=msg: self.status_var.set(f"Voice test failed: {msg}"))
        self.status_var.set("Testing voice…")
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    try:
        StreamTTSConfig().mainloop()
    except tk.TclError as exc:
        print(f"Could not open StreamTTS Config GUI: {exc}", file=sys.stderr)
        raise SystemExit(1)
