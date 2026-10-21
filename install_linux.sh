#!/bin/sh
# POSIX shell compatible: works whether launched directly or via `sh install_linux.sh`.
set -eu
set -x

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$SCRIPT_DIR
cd "$APP_DIR"

# Internal elevated phase of this SAME installer: no separate touch setup file.
# Restrict virtual system-wide input injection to explicitly enrolled users.
if [ "${1:-}" = "--configure-uinput" ]; then
    account="${2:-}"
    if [ "$(id -u)" -ne 0 ] || [ -z "$account" ] || [ "$account" = root ] || ! id "$account" >/dev/null 2>&1; then
        echo 'ERROR: uinput setup requires root and a valid desktop account.' >&2
        exit 1
    fi
    if ! getent group freehand-uinput >/dev/null 2>&1; then
        groupadd --system freehand-uinput
    fi
    mkdir -p /etc/udev/rules.d
    # Use group ownership on desktops without logind; where available,
    # uaccess also grants the active graphical user direct access.
    printf '%s\n' 'KERNEL=="uinput", SUBSYSTEM=="misc", GROUP="freehand-uinput", MODE="0660", TAG+="uaccess"' > /etc/udev/rules.d/70-freehand-uinput.rules
    if command -v modprobe >/dev/null 2>&1; then
        modprobe uinput || echo 'Note: uinput may be built into the kernel.'
    fi
    if command -v udevadm >/dev/null 2>&1; then
        udevadm control --reload-rules || true
        udevadm trigger --subsystem-match=misc --sysname-match=uinput || true
    fi
    usermod -a -G freehand-uinput "$account"
    for device in /dev/uinput /dev/input/uinput; do
        if [ -e "$device" ]; then
            chgrp freehand-uinput "$device"
            chmod 0660 "$device"
        fi
    done
    # The new group membership applies at the next login. Where POSIX ACLs
    # are available, make the existing /dev/uinput usable immediately, without
    # temporarily opening the device to every local user or adding a repair step.
    IMMEDIATE_TOUCH=0
    if command -v setfacl >/dev/null 2>&1; then
        for device in /dev/uinput /dev/input/uinput; do
            if [ -e "$device" ] && setfacl -m "u:$account:rw" "$device"; then
                IMMEDIATE_TOUCH=1
            fi
        done
    fi
    echo "Freehand multitouch configured for account: $account"
    if [ "$IMMEDIATE_TOUCH" -eq 1 ]; then
        echo 'The current account can access /dev/uinput immediately (POSIX ACL).'
    elif ! id -nG "$account" | tr ' ' '\n' | grep -qx freehand-uinput; then
        echo 'IMPORTANT: Log out and sign back in for the new multitouch group membership.'
    fi
    exit 0
fi

trap 'status=$?; if [ "$status" -ne 0 ]; then echo "ERROR: Installer failed (exit $status). See command output above." >&2; fi' 0

echo '============================================================'
echo 'Freehand: Spatial Computing - Linux installer'
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

echo '[5/9] Installing optional offline StreamTTS runtime...'
if "$VPY" -u -m pip install --verbose -r "$APP_DIR/requirements-voice.txt"; then
    echo '[6/9] Downloading StreamTTS voice files for offline speech...'
    echo 'This download is only needed once and is skipped if files already exist.'
    if ! "$VPY" -u "$APP_DIR/install_voice_models.py"; then
        echo 'WARNING: StreamTTS model download failed. The app will still install.' >&2
        echo "Retry later with: $VPY $APP_DIR/install_voice_models.py" >&2
    fi
else
    echo 'WARNING: StreamTTS dependencies could not be installed. Hand tracking and file search will still work.' >&2
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

echo '[8b/9] Configuring Linux two-hand multitouch (/dev/uinput)...'
# Ask for privilege escalation HERE during regular installation, rather than
# asking users to download/run another setup script after installation.
DESKTOP_USER="$(id -un)"
if [ "$DESKTOP_USER" = root ]; then
    DESKTOP_USER="${SUDO_USER:-}"
fi
if [ -z "$DESKTOP_USER" ] || [ "$DESKTOP_USER" = root ]; then
    echo 'WARNING: Could not identify desktop user; uinput setup requires a non-root account.' >&2
    TOUCH_SETUP_RESULT=failed
else
    TOUCH_SETUP_RESULT=failed
    if [ "$(id -u)" -eq 0 ]; then
        if sh "$APP_DIR/install_linux.sh" --configure-uinput "$DESKTOP_USER"; then
            TOUCH_SETUP_RESULT=ok
        fi
    else
        # sudo works in an interactive terminal; pkexec supports graphical
        # authentication on desktops that provide a polkit agent.
        if command -v sudo >/dev/null 2>&1 && [ -t 0 ]; then
            if sudo sh "$APP_DIR/install_linux.sh" --configure-uinput "$DESKTOP_USER"; then
                TOUCH_SETUP_RESULT=ok
            fi
        fi
        if [ "$TOUCH_SETUP_RESULT" != ok ] && command -v pkexec >/dev/null 2>&1; then
            if pkexec sh "$APP_DIR/install_linux.sh" --configure-uinput "$DESKTOP_USER"; then
                TOUCH_SETUP_RESULT=ok
            fi
        fi
    fi
fi
if [ "$TOUCH_SETUP_RESULT" != ok ]; then
    echo 'WARNING: Could not configure /dev/uinput access; Freehand installs, but native multitouch requires permission.' >&2
    echo 'Re-run this main installer from a terminal to complete the permission step.' >&2
else
    echo 'Linux multitouch permission configured. New group membership may require sign-out/sign-in.'
fi

echo '[9/9] Creating/updating Freehand application-menu shortcut automatically...'
# Use the same entry name as Freehand's built-in updater. Do not create a
# separate repair file, or another application-menu entry on each update.
ICON_PATH="$APP_DIR/Freehand_logo_small.png"
ICON_DESKTOP_ENTRY=""
if [ -s "$ICON_PATH" ]; then
    ICON_DESKTOP_ENTRY="Icon=$ICON_PATH"
else
    echo "WARNING: $ICON_PATH is missing; the launcher will use the desktop default icon." >&2
fi

# ZIP extraction can lose executable bits. Bash is specified explicitly by
# the desktop entry, so a non-executable run_linux.sh is never a launch blocker.
chmod +x "$APP_DIR/run_linux.sh"

APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS_DIR"
# Directly execute the known-working launcher through Bash (not a login shell).
# TryExec checks a real executable on all target Linux machines, not a path
# that may have been extracted without its executable bit.
cat > "$APPS_DIR/spatial-desktop-prototype.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Freehand: Spatial Computing
Comment=Hands-free spatial computing and gesture control
Exec=/bin/bash "$APP_DIR/run_linux.sh"
TryExec=/bin/bash
Path=$APP_DIR
$ICON_DESKTOP_ENTRY
Terminal=false
StartupNotify=false
Categories=Utility;Accessibility;
DESKTOP
chmod 0644 "$APPS_DIR/spatial-desktop-prototype.desktop"
# Tidy superseded Freehand shortcuts from older releases without touching
# other applications or user-customized launchers.
for legacy in freehand.desktop freehand-spatial-computing.desktop spatial-desktop.desktop spatial_desktop.desktop; do
    stale="$APPS_DIR/$legacy"
    if [ -f "$stale" ] && grep -Fq 'Name=Freehand: Spatial Computing' "$stale" && grep -Fq 'run_linux.sh' "$stale"; then
        rm -f "$stale"
    fi
done
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi
printf 'Spatial Desktop launcher installed: %s\n' "$APPS_DIR/spatial-desktop-prototype.desktop"

echo '============================================================'
echo 'INSTALLATION COMPLETE'
echo "Launcher: $APPS_DIR/spatial-desktop-prototype.desktop"
echo 'Find Freehand: Spatial Computing in your app launcher.'
echo 'Local search uses SQLite and remains available without d1. d1 is an optional local decision model.'
echo 'D1 remains off at runtime by default; turn it on in Settings -> Engine when required.'
echo 'StreamTTS runs offline when its runtime and voice files are installed.'
echo 'Face Welcome is optional and local-only; enroll a user through Settings to enable it.'
echo '============================================================'
