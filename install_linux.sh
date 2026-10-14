#!/bin/sh
# POSIX shell compatible: works whether launched directly or via `sh install_linux.sh`.
set -eu
set -x

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$SCRIPT_DIR
cd "$APP_DIR"
trap 'status=$?; if [ "$status" -ne 0 ]; then echo "ERROR: Installer failed (exit $status). See command output above." >&2; fi' 0

echo '============================================================'
echo 'Spatial Desktop Prototype - Linux installer'
echo "Project folder: $APP_DIR"
echo '============================================================'

echo '[1/9] Checking Python 3.10, 3.11, 3.12, or 3.13...'
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        case "$version" in
            3.10|3.11|3.12|3.13) PYTHON="$(command -v "$candidate")"; echo "Selected Python: $PYTHON ($version)"; break ;;
        esac
    fi
done
if [ -z "$PYTHON" ]; then
    echo 'ERROR: Python 3.10, 3.11, 3.12, or 3.13 is required.' >&2
    exit 1
fi

echo '[2/9] Checking virtual environment support...'
if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    echo 'ERROR: venv support is missing. Install your distribution package (for example, python3-venv).' >&2
    exit 1
fi

echo '[3/9] Creating/reusing virtual environment...'
if [ -x "$APP_DIR/.venv/bin/python" ]; then
    echo 'Existing virtual environment found; reusing it.'
else
    "$PYTHON" -m venv "$APP_DIR/.venv"
fi
VPY="$APP_DIR/.venv/bin/python"

echo '[4/9] Installing Spatial Desktop and local document-search dependencies...'
if ! "$VPY" -c 'import pip; print("Using pip", pip.__version__)'; then
    "$VPY" -m ensurepip
fi
"$VPY" -u -m pip install --verbose -r "$APP_DIR/requirements.txt"
echo '[4b] Installing the optional 3D hand renderer (moderngl / OpenGL)...'
if ! "$VPY" -u -m pip install -r "$APP_DIR/requirements-3d.txt"; then
    echo 'WARNING: 3D hand rendering could not be installed. The Skeleton and Flat hand styles still work.' >&2
    echo "Retry later with: $VPY -m pip install -r $APP_DIR/requirements-3d.txt" >&2
fi

echo '[5/9] Installing optional offline Kokoro TTS runtime...'
if "$VPY" -u -m pip install --verbose -r "$APP_DIR/requirements-voice.txt"; then
    echo '[6/9] Downloading Kokoro voice files for offline speech...'
    echo 'This download is only needed once and is skipped if files already exist.'
    if ! "$VPY" -u "$APP_DIR/install_voice_models.py"; then
        echo 'WARNING: Kokoro model download failed. The app will still install.' >&2
        echo "Retry later with: $VPY $APP_DIR/install_voice_models.py" >&2
    fi
else
    echo 'WARNING: Kokoro TTS dependencies could not be installed. Hand tracking and file search will still work.' >&2
    echo "Retry later with: $VPY -m pip install -r $APP_DIR/requirements-voice.txt" >&2
fi

echo '[7/9] Downloading optional local face welcome models...'
echo 'YuNet/SFace run locally. Face recognition stays disabled until a user enrolls in Settings.'
if ! "$VPY" -u "$APP_DIR/install_face_models.py"; then
    echo 'WARNING: Face-recognition model download failed. The app will still work without Face Welcome.' >&2
    echo "Retry later with: $VPY $APP_DIR/install_face_models.py" >&2
fi

echo '[8/9] Downloading the default quantized Liquid AI d1-omni-600M model and llama.cpp...'
echo 'The Q8 model plus projector is about 670 MB. A llama.cpp server for this CPU architecture is installed too'
echo '(prebuilt for x64/arm64/s390x; other architectures are compiled locally if cmake and a C++ compiler exist).'
if ! "$VPY" -u "$APP_DIR/install_d1_model.py" 600m; then
    echo 'WARNING: d1 setup did not complete. Spatial Desktop remains usable without d1.' >&2
    echo "Retry later with: $VPY $APP_DIR/install_d1_model.py 600m" >&2
fi

echo '[9/9] Creating desktop application launcher...'
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS_DIR"
cat > "$APPS_DIR/spatial-desktop-prototype.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Spatial Desktop Prototype
Comment=Camera-controlled spatial desktop and local search
Exec="$APP_DIR/run_linux.sh"
Path=$APP_DIR
Terminal=true
Categories=Utility;
DESKTOP
chmod +x "$APPS_DIR/spatial-desktop-prototype.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" || true
fi

echo '============================================================'
echo 'INSTALLATION COMPLETE'
echo "Launcher: $APPS_DIR/spatial-desktop-prototype.desktop"
echo 'Find Spatial Desktop Prototype in your app launcher.'
echo 'Local search uses SQLite and remains available without d1. d1 is an optional local decision model.'
echo 'Kokoro runs offline when its runtime and voice files are installed.'
echo 'Face Welcome is optional and local-only; enroll a user through Settings to enable it.'
echo '============================================================'
