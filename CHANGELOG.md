# Changelog

All notable changes to MonitaQC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.10.4] - 2026-02-22

### Added
- **Ejector delay parameter**: New `EJECTOR_DELAY` setting (seconds) — time-based delay after encoder target is reached before sending the ejector command. Configurable via UI, API, and data file. Available in all 7 languages

## [3.10.3] - 2026-02-22

### Fixed
- **Camera config lost on reconnect**: Camera properties (exposure, gain, brightness, contrast, saturation) are now stored on the CameraBuffer and automatically re-applied when a camera disconnects and reconnects
- **Camera config not persisted to disk**: Changing camera settings via API now auto-saves to the data file, so settings survive container restarts
- **Timeline prev/next buttons not working**: Pagination buttons now use HTTP fallback when auto-update is paused, fixing the issue where WebSocket handler dropped incoming images during manual navigation
- **Ejector queue monitoring**: Added detailed `[EJ_QUEUE]` and `[EJ_FIRE]` logging to trace ejection queue lifecycle and serial command dispatch

## [3.10.2] - 2026-02-22

### Fixed
- **Ejector dashboard icon always red**: SSE stream was missing `ejector_enabled` field, causing dashboard to always show ejector as disabled. Now reads runtime value via config module reference
- **SSE ejector_offset stale value**: Both SSE and REST status endpoints now read ejector config from live module reference instead of import-time snapshot
- **Click-to-view wrong image**: Timeline popup now uses `d_path` for stable frame lookup instead of column index, which could shift as new frames arrived
- **Ejection evaluation filtered enabled procs**: Only enabled procedures are now passed to the evaluation function
- **Serial send logging**: `_send_message` now logs warnings when serial is unavailable, aiding hardware debugging

## [3.10.1] - 2026-02-22

### Fixed
- **Bounding box scaling**: Timeline bboxes now correctly scale from original image resolution to thumbnail dimensions — previously coordinates were drawn unscaled, causing boxes to appear out of bounds or mispositioned
- **Full-res popup scaling**: Click-to-zoom popup correctly uses original coordinates for disk images and scaled coordinates for thumbnail fallback
- **Frame metadata**: Original image dimensions (`orig_h`, `orig_w`) now stored per-frame for accurate bbox scaling

## [3.10.0] - 2026-02-20

### Fixed
- **Negative bbox area guard**: Area conditions now return False for malformed bounding boxes (negative width/height)
- **LAB color validation**: Color ΔE condition validates L\*a\*b\* array integrity before comparison
- **Empty class name filtering**: `update_color_references()` skips detections with empty class names
- **Procedure UI re-render**: Toggling "Enabled" or changing cameras now immediately updates the UI
- **New rule defaults**: Adding a rule now initializes all fields (area, max_delta_e, reference_mode) to prevent undefined values
- **States API docstring**: Fixed outdated `"enabled": true` → `"light_status_check": false`

### Changed
- **Tooltips added**: All ejection procedure UI elements now have descriptive hover tooltips
- **Documentation overhaul**: Updated README (v3.10.0, ejection procedures, new API endpoints, 7 languages), CHANGELOG (fixed version ordering, pre-release labels), USER_MANUAL (v3.10.0, correct tabs, ejection procedures section, multi-camera FAQ, pipelines FAQ)

## [3.9.0] - 2026-02-19

### Added
- **Area/Size Conditions**: New ejection conditions `Area >`, `Area <`, `Area =` for bbox size-based ejection (threshold in pixels)
- **Per-Camera Procedure Filtering**: Each procedure can now specify which cameras it applies to (e.g., "1,2") or leave empty for all cameras
- **Camera ID on Detections**: Each detection now carries `_cam` field identifying which camera captured it

### Changed
- **Conditional LAB Extraction**: L*a*b* color extraction only runs when an active `color_delta` procedure exists, saving CPU when not needed

## [3.8.0] - 2026-02-19

### Added
- **Color Delta E Condition**: New ejection procedure condition `Color ΔE >` compares detected object color (CIE L*a*b*) against a reference. Three reference modes: vs Previous, vs Running Average (last 20), vs Fixed (user-captured)
- **Color Reference API**: `POST/GET /api/color-reference/{class_name}` for setting and querying fixed color references
- **Per-State Light Status Check**: Light status check (closed-loop serial verification) is now configured per camera state instead of a global toggle

### Changed
- **Ejection Conditions Simplified**: Removed `Present`/`Not Present` conditions — replaced by `Count > 0` and `Count = 0`. Count-based conditions: Count =, Count >, Count <
- **Class Count Check Merged**: Standalone class count check (enable/disable, classes, confidence) removed from Process tab — functionality merged into procedure count conditions
- **State `enabled` Field Removed**: Replaced with `light_status_check` boolean on each State

## [3.7.0] - 2026-02-19

### Added
- **Process Tab**: New dedicated tab consolidating all detection, ejection, and image processing configuration from Hardware, Cameras, and Advanced tabs into one place
- **Dynamic GPU Auto-Tuning**: `start.py` now detects GPU VRAM via `nvidia-smi` and computes optimal YOLO replicas (1 per GPU) and workers (80% VRAM / 500MB per worker)
- **Autoscaler Scale-Down**: Workers now scale down after 3 consecutive idle checks (~90s), freeing CPU/RAM/disk resources when demand drops
- **Cold Queue Stale Frame Flushing**: Stale frames (>10 min old) are discarded at startup and during processing instead of wasting GPU on outdated images
- **HTTP Connection Pooling**: YOLO inference requests now use `requests.Session` with pooled connections instead of creating new TCP connections per call
- **Process Tab Translations**: Added i18n translations for Process tab in all 7 languages (EN, FA, AR, DE, TR, JA, ES)

### Changed
- **ColdDiskQueue Rewrite**: Replaced O(n) `glob + sort` per get() with O(1) in-memory deque index. With 121k queued frames, this fixed inference throughput from 0.7 FPS to 93 FPS
- **Autoscaler Queue Sensitivity**: Now considers cold queue depth (>1000=CRITICAL, >100=WARNING) in addition to hot queue depth
- **Max Inference Workers**: Capped at `min(cpu*2, 32)` instead of unbounded `cpu*4`
- **FPS Measurement Buffer**: Increased from 200 to 2000 entries, removing the false 20.0 FPS ceiling
- **Docker Defaults**: Default YOLO replicas lowered to 1, workers to 2 (start.py auto-overrides based on hardware)

### Fixed
- Inference bottleneck: 96 threads all globbing 121k files simultaneously caused filesystem thrashing
- Race condition in ColdDiskQueue where multiple threads could claim the same file
- Cold queue growing unbounded because autoscaler only checked hot queue depth
- Model form layout in Inference tab: all settings on one row, YOLO weight dropdown properly sized

## [1.1.0] - 2026-01-06

### Added
- **Comprehensive Documentation**:
  - 📖 **USER_MANUAL.md**: 72-page complete user guide covering all features
  - 📊 **AUDIT_REPORT.md**: System audit with optimization recommendations
  - Installation & quick start guides
  - Camera setup (USB, IP, multi-camera)
  - YOLO AI detection configuration
  - Audio alerts system documentation
  - Image processing & capture modes
  - Timeline & review features
  - Database integration (TimescaleDB + Grafana)
  - Troubleshooting guide with solutions
  - Best practices & FAQ
  - API reference & Docker commands

- **Real-time Audio Alerts System**:
  - SSE-based detection events via Redis for cross-process communication
  - Voice narration of detected object names (Text-to-Speech)
  - Per-object audio control (enable/disable individual objects)
  - 4 customizable beep sounds per object (sine, square, sawtooth, triangle)
  - Non-blocking audio processing
  - Volume control and test functionality
  - Clean UI in Advanced tab

- **Advanced UI Organization**:
  - Consolidated Audio Alerts configuration in Advanced tab
  - Database Storage configuration moved to Advanced tab
  - Data File Editor relocated to bottom of Advanced tab
  - Improved responsive layout with proper sizing
  - Fixed dropdown/button width issues

### Changed
- **Configuration Management**:
  - All settings now configurable via web interface at http://localhost:5050/status
  - Settings persisted to .env.prepared_query_data
  - Export/Import service configuration functionality

- **Detection Event Architecture**:
  - Migrated from in-memory deque to Redis for cross-process event sharing
  - ProcessPoolExecutor workers can now communicate events to main FastAPI process
  - SSE stream delivers real-time detection events
  - Polling endpoint available as fallback

### Fixed
- Audio playing for disabled objects (now respects per-object enable/disable)
- Detection events not reaching frontend (Redis cross-process solution)
- SSE stream not showing detection events (proper Redis integration)
- Dropdown too narrow and button too wide in Database Storage section

### System Audit Findings
- ✅ All 6 Docker services properly configured and necessary
- ✅ Configuration files properly synchronized
- ✅ Clean codebase with no duplicate code
- ⚠️ Identified 11 unused Python dependencies (~50MB bloat) - removal recommended
- ⚠️ Timeline buffer capped at 100 frames when quality=100% - documented for future fix

### Performance
- Low CPU usage during idle operation
- Efficient Redis LRU eviction for cache management
- Optimized disk cleanup service with automated space management
- SSE provides real-time event delivery with minimal overhead

## [1.0.0] - 2026-01-05

### Added
- **Unlimited Camera Support**: Updated from fixed 4-camera limit to dynamic camera detection
  - Merged improvements from `zarrin-error-fix` branch
  - Auto-detects all available cameras on startup
  - Backward compatible with legacy 4-camera configuration
- **IP Camera Support**: Native support for RTSP and HTTP/MJPEG cameras
  - Configure via `IP_CAMERAS` environment variable (comma-separated URLs)
  - Mix USB and IP cameras seamlessly
  - Automatic backend selection (V4L2 for USB, FFMPEG for IP)
  - Support for authentication (username/password in URL)
  - Comprehensive setup guide in [docs/IP_CAMERA_SETUP.md](docs/IP_CAMERA_SETUP.md)
- **Lightweight MJPEG Streaming**: Integrated live camera feed into status page
  - `/video_feed` endpoint with 10 FPS, JPEG quality 40 for minimal bandwidth
  - Embedded directly in status monitoring interface at port 5050
  - No additional services or dependencies required
  - Uses existing FastAPI StreamingResponse capability
- **Full Deployment Mode**: Created `docker-compose.full.yml` with all optional services
  - Shipment fulfillment web interface (Django)
  - PostgreSQL database
  - Celery workers (high/low priority queues)
  - Image gallery (Pigallery2)
  - Legacy streaming service

### Changed
- **Container Naming**: Renamed all containers from `monitaqc_*` to `monitait_*` prefix
  - `monitait_vision_engine` - Core CV processing
  - `monitait_redis` - Message broker
  - `monitait_yolo` - AI inference
  - `monitait_cleanup` - Disk management
- **Branding Update**: Changed from VirasAd/Monitait to Smart Falcon AI
  - Support: admin@smartfalcon-ai.com
  - Removed client-specific references
- **Base Image**: Updated Dockerfile from Debian Buster to Bookworm (Python 3.10-slim)

### Planned
- Merge with fabric inspection capabilities (from FabriQC)
- Merge with signal counting capabilities (from PartQC Signal Counter)
- Unified admin interface for all QC modes
- Multi-application mode support
- Enhanced API with OpenAPI documentation
- Advanced analytics and reporting
- Cloud synchronization improvements

## Pre-release (2025-12-29)

### 0.3.0 — Project Reorganization
- Moved core service files to dedicated `vision_engine/` directory
- Service renamed: `monitaqc_counter` → `monitaqc_vision`
- Updated docker-compose.yml build context to `./vision_engine`

### 0.2.0 — Lightweight Architecture
- Reduced from 11+ containers to 4 core services
- Optimized Redis with 256MB memory limit and LRU eviction
- Changed Redis from `redis:latest` to `redis:7-alpine`
- Created `docker-compose.full.yml` with all optional services

### 0.1.0 — Initial Fork
- Fork from PartQC Box Counter
- MonitaQC branding and naming
- Standardized docker-compose container names
