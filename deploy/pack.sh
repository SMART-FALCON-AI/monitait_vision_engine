#!/usr/bin/env bash
# ============================================================
# MVE Offline Deployment Packer
# Run this on a machine WITH internet/docker to create a
# self-contained archive that can be deployed offline.
#
# Usage:  ./deploy/pack.sh [output_dir]
# Output: mve-v<VERSION>-offline.tar.gz
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# 4.0.436 — the VERSION file (main.py has no version="..." any more, so this was always "unknown")
VERSION=$(tr -d '[:space:]' < "$PROJECT_DIR/vision_engine/VERSION" 2>/dev/null || echo "unknown")
OUTPUT_DIR="${1:-$PROJECT_DIR}"
PACK_NAME="mve-v${VERSION}-offline"
STAGE_DIR="/tmp/${PACK_NAME}"

echo "==========================================="
echo " MVE Offline Packer v${VERSION}"
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

# Built images (vision_engine, yolo / math / anomaly inference).
# 4.0.436 — `docker compose config --images` lists every service's image. `docker compose
# images` only lists images of containers that already EXIST, so on a clean build host the
# inference images were silently left out of the offline archive.
while read -r img; do
    [ -n "$img" ] && IMAGES+=("$img")
done < <(docker compose config --images 2>/dev/null | sort -u)

# Also pull and save the external images
EXTERNAL_IMAGES=(
    "redis:7-alpine"
    "timescale/timescaledb:latest-pg15"
    "bpatrik/pigallery2:latest"
)   # 4.0.436 — grafana removed (gone from compose since 4.0.47; `docker save` failed on it)

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
    --exclude '.env.prepared_query_data*' \
    --exclude 'cold_queue' \
    --exclude '*.tar.gz' \
    "$PROJECT_DIR/" "$STAGE_DIR/project/"

# setup.sh is copied as part of the project tree above (it lives at project root).
# Make sure it's executable inside the staged archive.
chmod +x "$STAGE_DIR/project/setup.sh" 2>/dev/null || true

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
echo "   cd ${PACK_NAME}/project"
echo "   sudo bash setup.sh"
echo ""
