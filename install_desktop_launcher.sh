#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# ZIP extraction on some systems drops Unix executable bits, so repair the scripts
# before the desktop environment tries to execute them directly.
chmod +x "$APP_DIR/run_linux.sh" "$APP_DIR/spatial_desktop_launcher.sh"

APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS_DIR"
cat > "$APPS_DIR/spatial-desktop-prototype.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Freehand: Spatial Computing
Comment=Hand tracking and gesture control
Exec="$APP_DIR/spatial_desktop_launcher.sh"
TryExec="$APP_DIR/spatial_desktop_launcher.sh"
Terminal=false
Categories=Utility;
DESKTOP
chmod +x "$APPS_DIR/spatial-desktop-prototype.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi
printf 'Spatial Desktop launcher installed: %s
' "$APPS_DIR/spatial-desktop-prototype.desktop"
