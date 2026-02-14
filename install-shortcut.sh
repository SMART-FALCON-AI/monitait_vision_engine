#!/bin/bash
# Install MonitaQC desktop shortcut for the current user
# Run on target Linux machine: bash install-shortcut.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"
SHORTCUT_FILE="$DESKTOP_DIR/MonitaQC.desktop"

mkdir -p "$DESKTOP_DIR"

cat > "$SHORTCUT_FILE" << EOF
[Desktop Entry]
Name=MonitaQC
Comment=MonitaQC Vision Quality Control
Exec=xdg-open http://localhost
Icon=applications-science
Terminal=false
Type=Application
Categories=Utility;
StartupNotify=true
EOF

chmod +x "$SHORTCUT_FILE"

# Trust the shortcut (GNOME/Ubuntu)
if command -v gio &> /dev/null; then
    gio set "$SHORTCUT_FILE" metadata::trusted true 2>/dev/null
fi

echo "MonitaQC shortcut installed at: $SHORTCUT_FILE"
echo "Double-click it on the desktop to open MonitaQC in the browser."
