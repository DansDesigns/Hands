#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer the environment created by install_linux.sh. This avoids rejecting a
# valid Python 3.13 venv just because the system Python check is stale.
VPY="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$VPY" ]]; then
    PYTHON=""
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
            case "$version" in
                3.10|3.11|3.12|3.13) PYTHON="$(command -v "$candidate")"; break ;;
            esac
        fi
    done
    if [[ -z "$PYTHON" ]]; then
        echo 'ERROR: Python 3.10, 3.11, 3.12, or 3.13 is required.' >&2
        echo 'Run ./install_linux.sh first, then retry.' >&2
        read -r -p 'Press Enter to close...' _ || true
        exit 1
    fi
    echo "Creating virtual environment with $PYTHON..."
    "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
    VPY="$SCRIPT_DIR/.venv/bin/python"
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Give users a persistent log when launched from a desktop menu, while also
# keeping errors visible in a terminal when one is available.
LOG_FILE="$SCRIPT_DIR/spatial_desktop.log"
echo "Starting Spatial Desktop at $(date -Is)" >> "$LOG_FILE"
set +e
"$VPY" -u "$SCRIPT_DIR/main.py" >> "$LOG_FILE" 2>&1
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
    echo "Spatial Desktop failed to start. Log: $LOG_FILE" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    read -r -p 'Press Enter to close...' _ || true
    exit "$status"
fi
