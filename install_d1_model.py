"""Download the default quantized d1 model and a compatible llama.cpp server.

Works on Windows (x64 and ARM64) and Linux (x64, ARM64 and s390x prebuilt
releases; any other architecture, or a system the prebuilt binaries can't run on,
falls back to building llama-server from source when cmake and a C++ compiler are
available). macOS release builds are handled by the same code path.
"""
from __future__ import annotations
import json, os, platform, shutil, stat, subprocess, sys, tarfile, urllib.request, zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MODELS = {
    "600m": ("LiquidAI/d1-omni-600M-GGUF", "d1-omni-600M-Q8_0.gguf", "mmproj-d1-omni-600M-Q8_0.gguf"),
}
IS_WINDOWS = os.name == "nt"
SERVER_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"
SERVER_NAMES = ("llama-server.exe", "llama-server")
# GPU / vendor-specific release packages. The CPU package is the one that runs
# everywhere; these need extra drivers or runtimes and are never selected.
GPU_BACKEND_TOKENS = ("cuda", "cudart", "vulkan", "sycl", "opencl", "hip", "rocm", "metal",
                      "kompute", "openvino", "adreno", "snapdragon", "cann", "musa", "xcframework")
RELEASES_API = [
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20",
]


def download_file(url: str, target: Path, min_existing_size: int = 10_000_000) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if min_existing_size and target.exists() and target.stat().st_size > min_existing_size:
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


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _normalise_arch(machine: str) -> str:
    m = (machine or "").strip().lower()
    if m in {"x86_64", "amd64", "x64", "em64t"}:
        return "x64"
    if m in {"aarch64", "arm64", "armv8", "armv8l", "aarch64_be"}:
        return "arm64"
    if m in {"i386", "i486", "i586", "i686", "x86"}:
        return "x86"
    if m.startswith("armv7") or m.startswith("armv6") or m == "arm":
        return "arm32"
    return m or "unknown"


def _windows_native_arch() -> str | None:
    """Native Windows CPU, even when Python itself runs under x64 emulation on ARM64."""
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        process_machine = wintypes.USHORT(0)
        native_machine = wintypes.USHORT(0)
        fn = getattr(kernel32, "IsWow64Process2", None)
        if fn is not None and fn(kernel32.GetCurrentProcess(), ctypes.byref(process_machine),
                                 ctypes.byref(native_machine)):
            return {0xAA64: "arm64", 0x8664: "x64", 0x014C: "x86"}.get(native_machine.value)
    except Exception:
        pass
    for var in ("PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"):
        value = os.environ.get(var, "")
        if value:
            return _normalise_arch(value)
    return None


def _platform_target() -> tuple[str, str]:
    """Return (os_tag, arch) where os_tag is windows / linux / macos."""
    system = platform.system().lower()
    if IS_WINDOWS or system == "windows":
        return "windows", (_windows_native_arch() or _normalise_arch(platform.machine()))
    if system == "darwin":
        return "macos", _normalise_arch(platform.machine())
    return "linux", _normalise_arch(platform.machine())


def _windows_cpu_features() -> tuple[str, bool]:
    """Return (architecture, avx2_available) using OS/CPU capability checks."""
    arch = _windows_native_arch() or _normalise_arch(platform.machine())
    avx2 = False
    if arch == "x64" and IS_WINDOWS:
        try:
            import ctypes
            # Windows PF_AVX2_INSTRUCTIONS_AVAILABLE. This reports OS-enabled
            # processor support, rather than inferring from the CPU marketing name.
            avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        except Exception:
            avx2 = False
    return arch, avx2


def _linux_avx2() -> bool:
    try:
        return " avx2" in Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return False


def _arch_candidates(os_tag: str, arch: str) -> list[str]:
    """Release architectures to try, best first."""
    if os_tag == "windows" and arch == "arm64":
        return ["arm64", "x64"]  # Windows on ARM can also run x64 builds under emulation
    if os_tag == "macos" and arch == "arm64":
        return ["arm64"]
    return [arch]


# ---------------------------------------------------------------------------
# Release asset selection
# ---------------------------------------------------------------------------

def _asset_rank(name: str, os_tag: str, arch: str, avx2: bool = True) -> int:
    """Rank a llama.cpp release asset for this OS/arch; -1 means unusable.

    Current official CPU packages are named, for example:
      llama-b1234-bin-win-cpu-x64.zip       llama-b1234-bin-win-cpu-arm64.zip
      llama-b1234-bin-ubuntu-x64.tar.gz     llama-b1234-bin-ubuntu-arm64.tar.gz
      llama-b1234-bin-ubuntu-s390x.tar.gz   llama-b1234-bin-macos-arm64.tar.gz
    Current CPU packages contain every CPU variant (AVX/AVX2/AVX-512...) and pick
    the best one at runtime. Older avx/avx2/noavx-named packages are still ranked
    as a fallback for older release listings.
    """
    n = name.lower()
    if not (n.endswith(".zip") or n.endswith(".tar.gz") or n.endswith(".tgz")):
        return -1
    if any(token in n for token in GPU_BACKEND_TOKENS):
        return -1
    if not n.startswith("llama-"):
        return -1
    if os_tag == "windows":
        if not n.endswith(".zip") or not any(t in n for t in ("-win-", "-windows-", "-win64")):
            return -1
    elif os_tag == "linux":
        if not any(t in n for t in ("-ubuntu-", "-linux-")) or "android" in n:
            return -1
    elif os_tag == "macos":
        if "-macos-" not in n:
            return -1
    else:
        return -1

    stem = n.rsplit(".tar.gz", 1)[0] if n.endswith(".tar.gz") else n.rsplit(".", 1)[0]
    arch_tokens = {"x64": ("x64", "amd64", "x86_64", "win64"), "arm64": ("arm64", "aarch64"),
                   "x86": ("x86", "win32", "i686"), "s390x": ("s390x",)}.get(arch, (arch,))
    parts = stem.replace("_", "-").split("-")
    if not any(token in parts or stem.endswith(token) for token in arch_tokens):
        # Old Windows CPU packages (avx2/noavx) had no explicit arch and were x64.
        if not (os_tag == "windows" and arch == "x64" and
                any(p in parts for p in ("avx", "avx2", "avx512", "noavx"))):
            return -1
    if "cpu" in parts or (os_tag != "windows" and not any(p in parts for p in ("avx", "avx2", "avx512", "noavx"))):
        return 120
    if "avx512" in parts:
        # Can pass the --version probe yet crash during inference on CPUs without
        # AVX-512; the universal "cpu" and avx2/avx packages cover these machines.
        return -1
    if "avx2" in parts:
        return 100 if avx2 else -1
    if "avx" in parts:
        return 85 if avx2 else 90
    if "noavx" in parts:
        return 60 if avx2 else 95
    return 75


def _query_releases() -> tuple[list, Exception | None]:
    headers = {"User-Agent": "SpatialDesktop-installer", "Accept": "application/vnd.github+json"}
    releases, seen, last_error = [], set(), None
    for api_url in RELEASES_API:
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            for release in ([data] if isinstance(data, dict) else data):
                if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
                    continue
                identity = release.get("html_url") or release.get("tag_name") or release.get("name")
                if identity not in seen:
                    seen.add(identity)
                    releases.append(release)
        except Exception as exc:
            last_error = exc
    return releases, last_error


def select_release_assets(releases: list, os_tag: str, arch: str, avx2: bool = True) -> list[tuple[int, str, str]]:
    """Return (rank, name, url) candidates best first; newest release wins ties."""
    candidates, seen = [], set()
    arches = _arch_candidates(os_tag, arch)
    for release_index, release in enumerate(releases):
        for asset in release.get("assets", []) or []:
            name = asset.get("name", "")
            url = asset.get("browser_download_url", "")
            if not url or url in seen:
                continue
            for arch_index, candidate_arch in enumerate(arches):
                rank = _asset_rank(name, os_tag, candidate_arch, avx2)
                if rank >= 0:
                    seen.add(url)
                    # Native architecture first, then newest release, then package rank.
                    candidates.append(((-arch_index, -release_index, rank), name, url))
                    break
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(key[2], name, url) for key, name, url in candidates]


# ---------------------------------------------------------------------------
# Verification and staging
# ---------------------------------------------------------------------------

def _runtime_env(exe: Path) -> dict:
    """Make bundled shared libraries next to the server resolvable on Linux/macOS."""
    env = dict(os.environ)
    if not IS_WINDOWS:
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            existing = env.get(var, "")
            env[var] = str(exe.parent) + (os.pathsep + existing if existing else "")
    return env


def _probe_llama_server(exe: Path, verbose: bool = False) -> bool:
    """Check that a server can launch; optionally print the actual loader error."""
    if not exe.is_file():
        if verbose: print(f"    Probe failed: executable not found: {exe}")
        return False
    try:
        if exe.stat().st_size < 100_000 and not _has_server_impl(exe.parent):
            if verbose: print(f"    Probe failed: executable is unexpectedly small ({exe.stat().st_size} bytes)")
            return False
        if not IS_WINDOWS and not os.access(exe, os.X_OK):
            try:
                exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass
        result = subprocess.run(
            [str(exe), "--version"], cwd=str(exe.parent), capture_output=True,
            text=True, timeout=20, env=_runtime_env(exe),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        output = (stdout + "\n" + stderr).lower()
        okay = result.returncode == 0 and bool(output.strip()) and (
            "version" in output or "llama" in output or "build" in output
        )
        if verbose and not okay:
            print(f"    Probe failed: exit code {result.returncode}; stdout={stdout[:500]!r}; stderr={stderr[:1000]!r}")
        return okay
    except Exception as exc:
        if verbose:
            print(f"    Probe could not launch executable: {type(exc).__name__}: {exc}")
        return False


def _has_server_impl(folder: Path) -> bool:
    """Current llama.cpp builds split the server into a small launcher executable plus
    a llama-server-impl library (llama-server-impl.dll / libllama-server-impl.so)."""
    try:
        return any("llama-server-impl" in item.name.lower() for item in folder.iterdir())
    except OSError:
        return False


def _choose_server_executable(matches, launcher_dirs=()):
    """Choose the actual server binary, not a small stub/symlink-like entry.

    A large, self-contained binary is preferred. A small launcher is only accepted
    when its folder (in launcher_dirs) also holds the llama-server-impl library
    that contains the real server; the --version probe then verifies it runs.
    """
    plausible = [item for item in matches if int(getattr(item, "file_size", 0)) >= 100_000]
    if plausible:
        return max(plausible, key=lambda item: item.file_size)
    launcher_dirs = {str(Path(d)) for d in launcher_dirs}
    launchers = [item for item in matches
                 if int(getattr(item, "file_size", 0)) >= 1024
                 and str(Path(str(item.filename)).parent) in launcher_dirs]
    return max(launchers, key=lambda item: item.file_size) if launchers else None


def _swap_into_place(stage: Path, dest: Path) -> None:
    """Replace dest with a verified staging folder, restoring the old one on failure."""
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


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract a .zip or .tar.gz without allowing paths outside target."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                out = (target / info.filename.replace("\\", "/")).resolve()
                if out != root and root not in out.parents:
                    raise RuntimeError(f"unsafe path in archive: {info.filename}")
            zf.extractall(target)
            if not IS_WINDOWS:
                # zip does not reliably carry the executable bit.
                for info in zf.infolist():
                    mode = (info.external_attr >> 16) & 0o777
                    if mode:
                        try:
                            (target / info.filename).chmod(mode)
                        except OSError:
                            pass
        return
    with tarfile.open(archive, "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(target, filter="data")
        else:
            for member in tf.getmembers():
                out = (target / member.name).resolve()
                if out != root and root not in out.parents:
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
                if member.issym() or member.islnk():
                    link = (out.parent / member.linkname).resolve()
                    if link != root and root not in link.parents:
                        raise RuntimeError(f"unsafe link in archive: {member.name}")
            tf.extractall(target)


def _find_server_in_tree(tree: Path):
    """Locate the real llama-server binary inside an extracted release."""
    matches = []
    for path in tree.rglob("*"):
        if path.name.lower() in SERVER_NAMES and path.is_file() and not path.is_symlink():
            matches.append(SimpleNamespace(filename=str(path), file_size=path.stat().st_size, path=path))
    launcher_dirs = [str(m.path.parent) for m in matches if _has_server_impl(m.path.parent)]
    chosen = _choose_server_executable(matches, launcher_dirs)
    return (chosen.path if chosen is not None else None), matches


def _is_runtime_library(name: str) -> bool:
    low = name.lower()
    return low.endswith(".dll") or ".so" in low or low.endswith(".dylib")


def _stage_server_dir(server: Path, stage: Path, package_root: Path | None = None) -> Path:
    """Copy llama-server and the runtime libraries beside it into stage.

    Release archives also contain dozens of unrelated tools; only the server and
    its shared libraries (ggml CPU variants, llama, mtmd, ...) are kept. Library
    version symlinks are preserved on Linux/macOS.
    """
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    for item in server.parent.iterdir():
        if item.name == server.name or not _is_runtime_library(item.name):
            continue
        if item.is_symlink():
            os.symlink(os.readlink(item), stage / item.name)
        elif item.is_file():
            shutil.copy2(item, stage / item.name)
    staged = stage / SERVER_NAME
    shutil.copy2(server, staged)
    if IS_WINDOWS and package_root is not None:
        # Some packages keep runtime DLLs in subfolders; Windows resolves DLLs beside the exe.
        for dll in package_root.rglob("*.dll"):
            if not (stage / dll.name).exists():
                shutil.copy2(dll, stage / dll.name)
    return staged


# ---------------------------------------------------------------------------
# Reusing an existing install
# ---------------------------------------------------------------------------

def _find_existing_llama_server() -> Path | None:
    """Find a usable llama-server from PATH or common LM Studio/runtime folders."""
    candidates: list[Path] = []
    explicit = os.environ.get("SPATIAL_DESKTOP_LLAMA_SERVER", "").strip().strip('"')
    if explicit:
        explicit_path = Path(os.path.expandvars(explicit)).expanduser()
        if explicit_path.is_dir():
            explicit_path = explicit_path / SERVER_NAME
        if explicit_path.is_file():
            candidates.append(explicit_path)
        else:
            print(f"WARNING: SPATIAL_DESKTOP_LLAMA_SERVER does not point to a file: {explicit_path}")
    for command in (SERVER_NAME,):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))

    home = Path.home()
    if IS_WINDOWS:
        roots = [
            home / ".lmstudio" / "extensions" / "backends",
            home / ".lmstudio" / "bin",
            Path(os.environ.get("LOCALAPPDATA", "")) / "LM-Studio",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio",
            Path(os.environ.get("PROGRAMFILES", r"C:\\Program Files")) / "LM Studio",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")) / "LM Studio",
        ]
    else:
        roots = [
            home / ".lmstudio" / "extensions" / "backends",
            home / ".local" / "bin",
            home / "llama.cpp" / "build" / "bin",
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        ]
    # Also search common development/project folders. Users often have a working
    # llama.cpp executable inside a different project's venv/tools directory.
    for name in ("Documents", "Desktop", "source", "Projects", "Development", "Code", "GitHub", "repos", "src"):
        candidate_root = home / name
        if candidate_root.is_dir():
            roots.append(candidate_root)
    # Search likely runtime/project locations and cap total traversal cost.
    visited_dirs = 0
    for root in roots:
        try:
            if not root.is_dir():
                continue
            direct = root / SERVER_NAME
            if direct.is_file():
                candidates.append(direct)
            count = 0
            for base, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d.lower() not in {"cache", "node_modules", ".git"}]
                if SERVER_NAME in files:
                    candidates.append(Path(base) / SERVER_NAME)
                count += 1
                visited_dirs += 1
                if count >= 2500 or visited_dirs >= 6000:
                    break
            if visited_dirs >= 6000:
                break
        except (OSError, PermissionError):
            continue

    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            if _probe_llama_server(candidate):
                return candidate
        except OSError:
            continue
    return None


def _adopt_existing_llama_server(source_exe: Path, dest: Path) -> bool:
    """Copy an existing, verified server and adjacent runtime files into the app."""
    stage = dest.with_name(dest.name + ".staging")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        source_exe = source_exe.resolve()
        source_dir = source_exe.parent
        for item in source_dir.iterdir():
            name = item.name.lower()
            if IS_WINDOWS:
                wanted = item.is_file() and item.suffix.lower() in {".dll", ".exe", ".json", ".bin"}
            else:
                # Shared libraries (with their version symlinks) that sit beside the server.
                wanted = ".so" in name or name.endswith(".dylib")
            if wanted:
                try:
                    if item.is_symlink():
                        os.symlink(os.readlink(item), stage / item.name)
                    elif item.is_file():
                        shutil.copy2(item, stage / item.name)
                except OSError:
                    pass
        target = stage / SERVER_NAME
        if not target.exists():
            shutil.copy2(source_exe, target)
        if not _probe_llama_server(target):
            raise RuntimeError("copied existing server failed verification")
        _swap_into_place(stage, dest)
        return True
    except Exception as exc:
        print(f"WARNING: Could not reuse existing llama.cpp server: {exc}")
        shutil.rmtree(stage, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# Building from source (any architecture with cmake + a C++ compiler)
# ---------------------------------------------------------------------------

def _build_tools_available() -> tuple[bool, str]:
    if not shutil.which("cmake"):
        return False, "cmake was not found"
    if IS_WINDOWS:
        if not (shutil.which("cl") or shutil.which("clang-cl") or shutil.which("g++") or shutil.which("clang++")):
            return False, "no C++ compiler (Visual Studio Build Tools, clang or MinGW) was found on PATH"
    elif not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        return False, "no C++ compiler (g++ or clang++) was found"
    return True, ""


def build_llama_server_from_source(dest: Path, releases: list) -> bool:
    """Compile llama-server for this exact machine (native CPU flags, static libs)."""
    ok, reason = _build_tools_available()
    if not ok:
        print(f"Cannot build llama.cpp from source: {reason}.")
        if not IS_WINDOWS:
            print("Install build tools (for example: sudo apt install cmake g++ / sudo dnf install cmake gcc-c++) and rerun.")
        return False
    tag = next((r.get("tag_name") for r in releases if r.get("tag_name")), None)
    if not tag:
        print("Cannot build llama.cpp from source: no release tag available.")
        return False
    tools = ROOT / "tools"
    src_archive = tools / f"llama.cpp-{tag}-src.tar.gz"
    src_root = tools / "llama.cpp-src"
    build_dir = src_root / "build"
    print(f"Building llama.cpp {tag} from source for this machine. This can take several minutes.")
    try:
        src_archive.unlink(missing_ok=True)
        download_file(f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz",
                      src_archive, min_existing_size=0)
        shutil.rmtree(src_root, ignore_errors=True)
        extract_dir = tools / "llama.cpp-src.extract"
        shutil.rmtree(extract_dir, ignore_errors=True)
        _safe_extract(src_archive, extract_dir)
        top = next((p for p in extract_dir.iterdir() if (p / "CMakeLists.txt").is_file()), None)
        if top is None:
            raise RuntimeError("source archive did not contain CMakeLists.txt")
        top.rename(src_root)
        shutil.rmtree(extract_dir, ignore_errors=True)
        configure = [
            "cmake", "-S", str(src_root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON",
            # The app talks to the server locally over HTTP and starts it with
            # --no-webui, so skip HTTPS/curl and the web UI download/build.
            "-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
        ]
        subprocess.run(configure, check=True)
        jobs = str(max(1, (os.cpu_count() or 2)))
        subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release",
                        "--target", "llama-server", "-j", jobs], check=True)
        built, _ = _find_server_in_tree(build_dir)
        if built is None:
            raise RuntimeError("build finished but llama-server was not produced")
        stage = dest.with_name(dest.name + ".staging")
        staged = _stage_server_dir(built, stage)
        if not _probe_llama_server(staged, verbose=True):
            raise RuntimeError("built server failed its --version probe")
        _swap_into_place(stage, dest)
        print(f"Built and verified llama.cpp server at {dest / SERVER_NAME}")
        return True
    except Exception as exc:
        print(f"ERROR: building llama.cpp from source failed: {type(exc).__name__}: {exc}")
        shutil.rmtree(dest.with_name(dest.name + ".staging"), ignore_errors=True)
        return False
    finally:
        src_archive.unlink(missing_ok=True)
        shutil.rmtree(src_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main installer
# ---------------------------------------------------------------------------

def install_llama_server(force: bool = False) -> bool:
    """Install a llama.cpp server build compatible with this OS and CPU.

    Selects an official release package by OS and architecture, extracts the
    server with its runtime libraries, probes it, and tries the next candidate if
    one is unusable. If no prebuilt package fits or works, builds from source.
    """
    dest = ROOT / "tools" / "llama.cpp"
    dest.mkdir(parents=True, exist_ok=True)
    exe = dest / SERVER_NAME
    os_tag, arch = _platform_target()
    if os_tag == "windows":
        _arch, avx2 = _windows_cpu_features()
    else:
        avx2 = _linux_avx2() if arch == "x64" else True

    if exe.exists() and not force:
        if _probe_llama_server(exe):
            print(f"Verified existing llama.cpp server: {exe}")
            return True
        print(f"Existing {SERVER_NAME} failed verification; reinstalling a compatible build.")
        try:
            # Move the invalid install out of the way first, so an interrupted
            # download cannot leave a half-overwritten executable behind.
            bad = dest.with_name(dest.name + ".invalid")
            if bad.exists():
                shutil.rmtree(bad, ignore_errors=True)
            dest.rename(bad)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(bad, ignore_errors=True)
        except OSError:
            try: exe.unlink(missing_ok=True)
            except OSError: pass

    if not force:
        existing = _find_existing_llama_server()
        if existing:
            print(f"Found an existing compatible llama.cpp server: {existing}")
            if _adopt_existing_llama_server(existing, dest):
                print(f"Using existing llama.cpp server at {exe}; no separate download required.")
                return True
        else:
            print("No reusable llama-server found on PATH or in common runtime folders.")

    print(f"Detecting compatible llama.cpp build: {os_tag} {arch}"
          + (f", AVX2={'yes' if avx2 else 'no/unknown'}" if arch == "x64" else ""))
    releases, last_error = _query_releases()
    if not releases:
        print(f"ERROR: Could not query official llama.cpp releases: {last_error}")
        return False

    candidates = select_release_assets(releases, os_tag, arch, avx2)
    if not candidates:
        print(f"No prebuilt llama.cpp CPU package is published for {os_tag} {arch}.")
    archive_dir = ROOT / "tools"
    for rank, name, url in candidates:
        print(f"Trying compatible llama.cpp asset: {name}")
        suffix = ".zip" if name.lower().endswith(".zip") else ".tar.gz"
        archive = archive_dir / ("llama-cpp-download" + suffix)
        extract_dir = dest.with_name(dest.name + ".extract")
        stage = dest.with_name(dest.name + ".staging")
        try:
            # Never reuse an archive left by an interrupted or older attempt:
            # each candidate URL must be downloaded and validated independently.
            archive.unlink(missing_ok=True)
            download_file(url, archive, min_existing_size=0)
            shutil.rmtree(extract_dir, ignore_errors=True)
            _safe_extract(archive, extract_dir)
            # Some release archives contain a tiny launcher/stub alongside the
            # actual server binary. Never blindly use the first filename match.
            server, matches = _find_server_in_tree(extract_dir)
            if server is None:
                details = ", ".join(f"{Path(m.filename).name} ({m.file_size} bytes)" for m in matches[:12])
                raise RuntimeError("archive contains no plausible llama-server binary"
                                   + (f"; server entries: {details}" if details else ""))
            print(f"    Selected archive executable: {server.relative_to(extract_dir)} ({server.stat().st_size:,} bytes)")
            staged_exe = _stage_server_dir(server, stage, package_root=extract_dir)
            if not _probe_llama_server(staged_exe, verbose=True):
                raise RuntimeError("downloaded server failed its --version compatibility probe (see launch diagnostics above)")
            # Swap only after the staged executable passes verification.
            _swap_into_place(stage, dest)
            print(f"Installed and verified llama.cpp server at {exe}")
            return True
        except Exception as exc:
            print(f"WARNING: {name} was not usable: {type(exc).__name__}: {exc}")
            shutil.rmtree(stage, ignore_errors=True)
            continue
        finally:
            try: archive.unlink(missing_ok=True)
            except OSError: pass
            shutil.rmtree(extract_dir, ignore_errors=True)

    # No prebuilt package for this platform, or none of them run here (for example
    # 32-bit ARM, RISC-V, ppc64le, musl-based distributions, or an older glibc).
    if build_llama_server_from_source(dest, releases):
        return True
    print("ERROR: Could not install a working llama.cpp server for this machine.")
    print(f"If {SERVER_NAME} is already installed elsewhere, set SPATIAL_DESKTOP_LLAMA_SERVER to its full path and rerun this installer.")
    print("Official releases: https://github.com/ggml-org/llama.cpp/releases")
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
        llama_ok = install_llama_server(force="--force" in sys.argv[1:])
        settings = ROOT / "d1_settings.json"
        settings.write_text(json.dumps({"model": key}, indent=2) + "\n", encoding="utf-8")
        if not llama_ok:
            retry = (r".venv\Scripts\python.exe install_d1_model.py 600m" if IS_WINDOWS
                     else ".venv/bin/python install_d1_model.py 600m")
            print("\nWARNING: d1 model downloaded, but llama.cpp could not be installed.")
            print("Rerun: " + retry)
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
