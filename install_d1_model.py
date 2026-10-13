"""Download the default quantized d1 model and a compatible llama.cpp server on Windows."""
from __future__ import annotations
import json, os, shutil, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "600m": ("LiquidAI/d1-omni-600M-GGUF", "d1-omni-600M-Q8_0.gguf", "mmproj-d1-omni-600M-Q8_0.gguf"),
}

def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"Already downloaded: {target.name}")
        return
    temp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SpatialDesktop-d1-installer"})
    with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            block = response.read(1024 * 1024)
            if not block: break
            out.write(block); done += len(block)
            if total:
                print(f"\r  {target.name}: {done/total*100:5.1f}%", end="", flush=True)
    if total: print()
    temp.replace(target)


def _windows_cpu_features() -> tuple[str, bool]:
    """Return (architecture, avx2_available) using OS/CPU capability checks."""
    import platform
    machine = (platform.machine() or "").lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    avx2 = False
    if arch == "x64" and os.name == "nt":
        try:
            import ctypes
            # Windows PF_AVX2_INSTRUCTIONS_AVAILABLE. This reports OS-enabled
            # processor support, rather than inferring from the CPU marketing name.
            avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        except Exception:
            avx2 = False
        # Conservative fallback for older Windows/Python builds where the
        # feature constant/API is unavailable. The i7-7700 supports AVX2.
        if not avx2:
            try:
                import subprocess
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                cpu_name = (result.stdout or "").lower()
                if "i7-7700" in cpu_name or "i7-7700k" in cpu_name:
                    avx2 = True
            except Exception:
                pass
    return arch, avx2


def _asset_rank(name: str, arch: str, avx2: bool) -> int:
    """Rank release asset names from most suitable to least suitable."""
    n = name.lower()
    if not n.endswith(".zip") or "win" not in n:
        return -1
    if any(x in n for x in ("cuda", "cudart", "vulkan", "sycl", "opencl", "hip", "rocm", "metal", "kompute")):
        return -1
    if arch == "arm64":
        if not any(x in n for x in ("arm64", "aarch64")):
            return -1
        return 100 if "cpu" in n or "noavx" in n else 80
    if any(x in n for x in ("arm64", "aarch64", "win32", "x86-")):
        return -1
    if not any(x in n for x in ("x64", "amd64", "win64")):
        # Current official binaries are normally explicitly suffixed x64.
        return -1
    # Prefer the strongest CPU instruction set the machine is known to support.
    if avx2:
        if "avx2" in n:
            return 120
        if "avx" in n and "noavx" not in n:
            return 90
        if "cpu" in n or "noavx" in n:
            return 70
    else:
        if "noavx" in n or "avx1" in n or "avx-" in n:
            return 120
        if "cpu" in n:
            return 100
        if "avx2" in n or "avx512" in n:
            return -1
    # Generic Windows x64 archive is a final compatible candidate if naming
    # does not advertise unsupported instructions.
    if not any(x in n for x in ("avx512", "avx2")):
        return 50
    return -1


def _probe_llama_server(exe: Path) -> bool:
    """Reject HTML/error downloads, wrong-architecture executables and missing DLLs."""
    if not exe.is_file() or exe.stat().st_size < 100_000:
        return False
    try:
        import subprocess
        result = subprocess.run([str(exe), "--version"], capture_output=True,
                                text=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = ((result.stdout or "") + (result.stderr or "")).lower()
        # llama.cpp versions normally return 0, but accept version output from
        # builds that use a non-zero status for --version.
        return result.returncode == 0 and bool(output.strip()) and (
            "version" in output or "llama" in output or "build" in output
        )
    except (OSError, subprocess.TimeoutExpired, Exception):
        return False


def install_llama_server(force: bool = False) -> bool:
    """Install a Windows llama.cpp server build compatible with this machine.

    Selects an official release asset by OS architecture and CPU features,
    extracts the executable plus adjacent runtime DLLs, probes the executable,
    and retries with the next compatible asset if a candidate is unusable.
    """
    import platform
    import subprocess
    dest = ROOT / "tools" / "llama.cpp"
    dest.mkdir(parents=True, exist_ok=True)
    exe = dest / "llama-server.exe"
    arch, avx2 = _windows_cpu_features()

    if exe.exists() and not force:
        if _probe_llama_server(exe):
            print(f"Verified existing llama.cpp server: {exe}")
            return True
        print("Existing llama-server.exe failed verification; reinstalling a compatible build.")
        try:
            # Move the invalid install out of the way first, so Windows cannot
            # leave a half-overwritten executable after an interrupted download.
            bad = dest.with_name(dest.name + ".invalid")
            if bad.exists():
                shutil.rmtree(bad, ignore_errors=True)
            dest.rename(bad)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(bad, ignore_errors=True)
        except OSError:
            try: exe.unlink(missing_ok=True)
            except OSError: pass

    print(f"Detecting compatible llama.cpp build: Windows {arch}, AVX2={'yes' if avx2 else 'no/unknown'}")
    headers = {"User-Agent": "SpatialDesktop-installer", "Accept": "application/vnd.github+json"}
    api_urls = [
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5",
    ]
    releases = []
    last_error = None
    for api_url in api_urls:
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            releases = [data] if isinstance(data, dict) else data
            if releases:
                break
        except Exception as exc:
            last_error = exc
    if not releases:
        print(f"ERROR: Could not query official llama.cpp releases: {last_error}")
        return False

    candidates = []
    seen = set()
    for release in releases:
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            url = asset.get("browser_download_url", "")
            rank = _asset_rank(name, arch, avx2)
            if rank >= 0 and url and url not in seen:
                seen.add(url)
                candidates.append((rank, name, url))
        # Do not stop at the first release if it lacks a compatible CPU package.
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        print("ERROR: No compatible official Windows CPU archive was found in recent llama.cpp releases.")
        print("Check https://github.com/ggml-org/llama.cpp/releases and retry later.")
        return False

    archive = ROOT / "tools" / "llama-cpp-windows.zip"
    for rank, name, url in candidates:
        print(f"Trying compatible llama.cpp asset: {name}")
        try:
            # Never reuse an archive left by an interrupted or older attempt:
            # each candidate URL must be downloaded and validated independently.
            archive.unlink(missing_ok=True)
            download_file(url, archive)
            with zipfile.ZipFile(archive) as zf:
                infos = zf.infolist()
                matches = [i for i in infos if i.filename.lower().endswith("llama-server.exe")]
                if not matches:
                    raise RuntimeError("archive does not contain llama-server.exe")
                # Prefer the server and DLLs from the same directory as the exe.
                chosen = matches[0]
                prefix = chosen.filename.rsplit("/", 1)[0] if "/" in chosen.filename else ""
                stage = dest.with_name(dest.name + ".staging")
                shutil.rmtree(stage, ignore_errors=True)
                stage.mkdir(parents=True, exist_ok=True)
                for info in infos:
                    if info.is_dir():
                        continue
                    name_in_zip = info.filename
                    if prefix:
                        if not name_in_zip.startswith(prefix + "/"):
                            continue
                        relative = name_in_zip[len(prefix) + 1:]
                    else:
                        relative = name_in_zip
                    if not relative or "/" in relative:
                        continue
                    low = relative.lower()
                    # Preserve all DLLs and EXEs shipped alongside the server;
                    # ggml backends and runtime DLLs are required by some builds.
                    if low.endswith((".dll", ".exe", ".json", ".bin")):
                        target = stage / relative
                        with zf.open(info) as src, target.open("wb") as out:
                            shutil.copyfileobj(src, out)
                staged_exe = stage / "llama-server.exe"
                if not staged_exe.exists():
                    raise RuntimeError("server executable was not extracted")
                if not _probe_llama_server(staged_exe):
                    raise RuntimeError("downloaded server failed its --version compatibility probe")
                # Swap only after the staged executable passes verification.
                backup = dest.with_name(dest.name + ".old")
                shutil.rmtree(backup, ignore_errors=True)
                if dest.exists():
                    dest.rename(backup)
                try:
                    stage.rename(dest)
                except Exception:
                    if backup.exists() and not dest.exists():
                        backup.rename(dest)
                    raise
                shutil.rmtree(backup, ignore_errors=True)
                archive.unlink(missing_ok=True)
                print(f"Installed and verified llama.cpp server at {exe}")
                return True
        except Exception as exc:
            print(f"WARNING: {name} was not usable: {type(exc).__name__}: {exc}")
            try: archive.unlink(missing_ok=True)
            except OSError: pass
            try: shutil.rmtree(dest.with_name(dest.name + ".staging"), ignore_errors=True)
            except OSError: pass
            continue

    print("ERROR: All compatible llama.cpp downloads failed verification.")
    print("Spatial Desktop can still run without d1; rerun install_d1_model.py to retry.")
    return False

def main() -> int:
    # The 600M Q8 model is the only supported installer choice for now.
    key = "600m"
    repo, model_name, mmproj_name = MODELS[key]
    folder = ROOT / "models" / "d1" / key
    print(f"\nDownloading Liquid AI d1 {key} quantized model into {folder}")
    print("This includes the decision weights and the matching multimodal projector.")
    try:
        download_file(f"https://huggingface.co/{repo}/resolve/main/{model_name}", folder / model_name)
        download_file(f"https://huggingface.co/{repo}/resolve/main/{mmproj_name}", folder / mmproj_name)
        llama_ok = True
        if os.name == "nt":
            llama_ok = install_llama_server()
        settings = ROOT / "d1_settings.json"
        settings.write_text(json.dumps({"model": key}, indent=2) + "\n", encoding="utf-8")
        if os.name == "nt" and not llama_ok:
            print("\nWARNING: d1 model downloaded, but llama.cpp could not be installed.")
            print("Rerun: .venv\\Scripts\\python.exe install_d1_model.py 600m")
            return 2
        print("\nd1 model and compatible llama.cpp setup complete. Spatial Desktop starts llama-server automatically when needed.")
        return 0
    except KeyboardInterrupt:
        print("\nDownload cancelled. Remove any .part files or rerun this script to retry.")
        return 1
    except Exception as exc:
        print(f"\nERROR downloading/installing d1: {type(exc).__name__}: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
