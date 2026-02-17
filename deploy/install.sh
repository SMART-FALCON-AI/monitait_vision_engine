#!/usr/bin/env bash
# ============================================================
# MonitaQC Offline Installer
# Run this on the REMOTE machine (no internet required).
# Prerequisites: Docker & Docker Compose installed.
#
# Usage:  sudo bash install.sh [install_dir]
# Default install dir: /opt/monitaqc
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${1:-/opt/monitaqc}"

echo "==========================================="
echo " MonitaQC Offline Installer"
echo "==========================================="

# ---- Pre-flight checks ----
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed. Install Docker first."
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "ERROR: Docker Compose v2 is not installed."
    exit 1
fi

# ---- 1. Load Docker images ----
echo ""
echo "[1/3] Loading Docker images (this may take a few minutes)..."
if [ -f "$SCRIPT_DIR/images/all-images.tar" ]; then
    docker load -i "$SCRIPT_DIR/images/all-images.tar"
    echo "  Images loaded successfully."
else
    echo "ERROR: images/all-images.tar not found!"
    exit 1
fi

# ---- 2. Copy project files ----
echo ""
echo "[2/3] Installing project to ${INSTALL_DIR}..."
mkdir -p "$INSTALL_DIR"

if [ -d "$SCRIPT_DIR/project" ]; then
    cp -a "$SCRIPT_DIR/project/." "$INSTALL_DIR/"
else
    echo "ERROR: project/ directory not found!"
    exit 1
fi

# Create volume directories
for d in volumes/redis volumes/timescaledb volumes/grafana \
         volumes/pigallery2_config volumes/pigallery2_db \
         volumes/weights raw_images; do
    mkdir -p "$INSTALL_DIR/$d"
done

# ---- 3. Start ----
echo ""
echo "[3/3] Starting MonitaQC..."
cd "$INSTALL_DIR"
python3 start.py up -d

echo ""
echo "==========================================="
echo " MonitaQC installed at: ${INSTALL_DIR}"
echo " Dashboard: http://$(hostname -I | awk '{print $1}'):80"
echo "==========================================="
echo ""
echo " Useful commands:"
echo "   cd ${INSTALL_DIR}"
echo "   python3 start.py          # Start (auto-detects hardware)"
echo "   docker compose logs -f    # View logs"
echo "   docker compose down       # Stop"
echo "   docker compose restart    # Restart"
echo ""
