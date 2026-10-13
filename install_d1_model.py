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

def install_llama_server() -> bool:
    dest = ROOT / "tools" / "llama.cpp"
    dest.mkdir(parents=True, exist_ok=True)
    exe = dest / "llama-server.exe"
    if exe.exists():
        print("llama-server.exe is already installed.")
        return True
    print("Looking up the latest llama.cpp Windows CPU release...")
    req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", headers={"User-Agent": "SpatialDesktop-installer"})
    with urllib.request.urlopen(req, timeout=30) as response:
        release = json.loads(response.read().decode("utf-8"))
    assets = release.get("assets", [])
    candidates = [a for a in assets if a.get("name", "").lower().endswith(".zip") and "win-cpu-x64" in a.get("name", "").lower()]
    if not candidates:
        candidates = [a for a in assets if a.get("name", "").lower().endswith(".zip") and "win-cpu" in a.get("name", "").lower()]
    if not candidates:
        print("WARNING: couldn't find a CPU Windows llama.cpp ZIP in the latest release.")
        print("Download a recent Windows llama.cpp release and place llama-server.exe plus its DLLs in tools\\llama.cpp.")
        return False
    asset = candidates[0]
    archive = ROOT / "tools" / "llama-cpp-windows.zip"
    download_file(asset["browser_download_url"], archive)
    with zipfile.ZipFile(archive) as zf:
        matches = [n for n in zf.namelist() if n.lower().endswith("llama-server.exe")]
        if not matches:
            raise RuntimeError("llama-server.exe not found inside downloaded llama.cpp archive")
        prefix = matches[0].rsplit("/", 1)[0] if "/" in matches[0] else ""
        for name in zf.namelist():
            if prefix and not name.startswith(prefix + "/"):
                continue
            if name.endswith("/"):
                continue
            relative = name[len(prefix)+1:] if prefix else name
            # Preserve sibling DLLs required by llama-server.exe.
            if relative and (relative.lower().endswith((".dll", ".exe")) or relative.lower().endswith(".json")):
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as out: shutil.copyfileobj(src, out)
    archive.unlink(missing_ok=True)
    return exe.exists()

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
        if os.name == "nt":
            install_llama_server()
        settings = ROOT / "d1_settings.json"
        settings.write_text(json.dumps({"model": key}, indent=2) + "\n", encoding="utf-8")
        print("\nd1 model download complete. Spatial Desktop starts llama-server automatically when needed.")
        return 0
    except KeyboardInterrupt:
        print("\nDownload cancelled. Remove any .part files or rerun this script to retry.")
        return 1
    except Exception as exc:
        print(f"\nERROR downloading/installing d1: {type(exc).__name__}: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
