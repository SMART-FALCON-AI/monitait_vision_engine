#!/usr/bin/env bash
# ============================================================
# MonitaQC Offline Deployment Packer
# Run this on a machine WITH internet/docker to create a
# self-contained archive that can be deployed offline.
#
# Usage:  ./deploy/pack.sh [output_dir]
# Output: monitaqc-v<VERSION>-offline.tar.gz
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION=$(grep -oP 'version="\K[^"]+' "$PROJECT_DIR/vision_engine/main.py" || echo "unknown")
OUTPUT_DIR="${1:-$PROJECT_DIR}"
PACK_NAME="monitaqc-v${VERSION}-offline"
STAGE_DIR="/tmp/${PACK_NAME}"

echo "==========================================="
echo " MonitaQC Offline Packer v${VERSION}"
echo "==========================================="

# ---- 1. Build all images ----
echo ""
echo "[1/4] Building Docker images..."
cd "$PROJECT_DIR"
docker compose build

# ---- 2. Collect image names ----
echo ""
echo "[2/4] Saving Docker images to tar..."

# Get all image names from compose
IMAGES=()

# Built images (vision_engine, yolo_inference)
for svc in $(docker compose config --services); do
    img=$(docker compose images "$svc" --format json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if isinstance(data, list):
    for d in data:
        print(d.get('Repository', '') + ':' + d.get('Tag', 'latest'))
        break
" 2>/dev/null || true)
    if [ -n "$img" ] && [ "$img" != ":" ]; then
        IMAGES+=("$img")
    fi
done

# Also pull and save the external images
EXTERNAL_IMAGES=(
    "redis:7-alpine"
    "timescale/timescaledb:latest-pg15"
    "grafana/grafana:latest"
    "bpatrik/pigallery2:latest"
)

for img in "${EXTERNAL_IMAGES[@]}"; do
    echo "  Pulling $img ..."
    docker pull "$img" 2>/dev/null || echo "  Warning: could not pull $img, using local"
    IMAGES+=("$img")
done

# ---- 3. Stage files ----
echo ""
echo "[3/4] Staging deployment files..."
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/images"

# Save all docker images into a single tar
echo "  Saving ${#IMAGES[@]} images (this may take a few minutes)..."
docker save "${IMAGES[@]}" -o "$STAGE_DIR/images/all-images.tar"

# Copy project files (excluding .git, raw_images, volumes data, __pycache__)
rsync -a --progress \
    --exclude '.git' \
    --exclude 'raw_images' \
    --exclude 'volumes/redis' \
    --exclude 'volumes/timescaledb' \
    --exclude 'volumes/grafana' \
    --exclude 'volumes/pigallery2_config' \
    --exclude 'volumes/pigallery2_db' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'output.jpg' \
    --exclude '.env' \
    --exclude '*.tar.gz' \
    "$PROJECT_DIR/" "$STAGE_DIR/project/"

# Copy the install script to root of archive
cp "$SCRIPT_DIR/install.sh" "$STAGE_DIR/install.sh"
chmod +x "$STAGE_DIR/install.sh"

# ---- 4. Create archive ----
echo ""
echo "[4/4] Creating archive..."
cd /tmp
tar czf "${OUTPUT_DIR}/${PACK_NAME}.tar.gz" "$PACK_NAME"
rm -rf "$STAGE_DIR"

SIZE=$(du -h "${OUTPUT_DIR}/${PACK_NAME}.tar.gz" | cut -f1)
echo ""
echo "==========================================="
echo " Done! Archive: ${PACK_NAME}.tar.gz (${SIZE})"
echo "==========================================="
echo ""
echo " Transfer this file to the remote machine, then run:"
echo "   tar xzf ${PACK_NAME}.tar.gz"
echo "   cd ${PACK_NAME}"
echo "   sudo bash install.sh"
echo ""
