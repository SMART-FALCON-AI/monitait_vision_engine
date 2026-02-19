# MonitaQC - How-To Guide

**Version:** 3.6.0 | **Zero to Hero Configuration Guide**

This guide walks you through every tab and configuration option in MonitaQC, from first boot to full production use.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [First Boot](#2-first-boot)
3. [Dashboard Tab](#3-dashboard-tab-)
4. [AI Assistant Tab](#4-ai-assistant-tab-)
5. [Gallery Tab](#5-gallery-tab-)
6. [Charts Tab](#6-charts-tab-)
7. [Hardware Tab](#7-hardware-tab-)
8. [Cameras Tab](#8-cameras-tab-)
9. [Inference Tab](#9-inference-tab-)
10. [Advanced Tab](#10-advanced-tab-)
11. [Driver & GPU Setup](#11-driver--gpu-setup)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites & Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 LTS |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | - | NVIDIA with CUDA (RTX 2060+) |
| Disk | 50 GB | 256 GB SSD |
| Docker | 20.10+ | Latest stable |

### Software Prerequisites

1. **Docker & Docker Compose** (required)
2. **Python 3** (for the startup script)
3. **NVIDIA drivers + NVIDIA Container Toolkit** (for GPU inference)
4. **Git** (to clone the repository)

### Installing NVIDIA Drivers & Container Toolkit (Linux)

```bash
# 1. Install NVIDIA driver
sudo apt update
sudo apt install -y nvidia-driver-535   # or latest available

# Reboot after driver install
sudo reboot

# 2. Verify driver
nvidia-smi

# 3. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Updating NVIDIA Drivers

```bash
# Check current driver version
nvidia-smi

# Update driver
sudo apt update
sudo apt install -y nvidia-driver-550   # replace with desired version

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone http://gitlab.virasad.ir/monitait/monitaqc.git
cd monitaqc

# Place your YOLO model weights
# Train at https://ai-trainer.monitait.com, then:
cp /path/to/best.pt volumes/weights/best.pt

# Start the application
./start.sh          # Linux
start.bat           # Windows
```

The `start.py` script automatically:
- Detects your OS (Linux = production, Windows = development)
- Detects hardware (CPU cores, RAM, GPU count, GPU VRAM)
- Auto-tunes YOLO replicas (1 per GPU) and workers (80% VRAM / 500MB per worker)
- Auto-tunes shared memory and Redis memory
- Generates a `.env` file
- Runs `docker compose up -d`

Example output for RTX 3050 (8GB VRAM):
```
[MonitaQC] Hardware: 24 CPU cores, 31.1 GB RAM, 1 GPU(s), 8192MB VRAM
[MonitaQC] Auto-tuned: 1 YOLO replicas x 13 workers, SHM=4g, Redis=1554MB
```

### Verifying Installation

Open your browser to `http://<server-ip>` and verify:
- The Dashboard loads with all sidebar metrics
- Health status shows all services as connected (check `http://<server-ip>/health`)

---

## 2. First Boot

After starting MonitaQC for the first time, configure the system in this order:

1. **Cameras** - Connect and configure cameras
2. **Hardware** - Set up serial communication and lighting
3. **Inference** - Choose AI model and set confidence thresholds
4. **Process** - Configure ejector, OK/NG timing, image processing, and detection alerts
5. **Advanced** - Fine-tune timeline display and global settings
6. **Dashboard** - Monitor production in real-time

> **Important:** After configuring each section, click **"Save All Configuration"** (top-right corner) to persist settings across restarts.

---

## 3. Dashboard Tab 📊

The Dashboard is the main monitoring view during production.

### Layout

```
┌──────────────────────────────────────────────────────┐
│  ┌─────────┐  ┌──────────────────────────────────┐   │
│  │ Sidebar  │  │  Timeline (captured images)      │   │
│  │ 160px    │  │                                  │   │
│  │          │  │  cam1 │ cam1 │ cam1 │ cam1 │ ... │   │
│  │ Encoder  │  │  cam2 │ cam2 │ cam2 │ cam2 │ ... │   │
│  │ Speed    │  │                                  │   │
│  │ OK / NG  │  │  Each column = one capture event │   │
│  │ Ejector  │  │  Header shows: Encoder | Time    │   │
│  │ CPU/RAM  │  │                                  │   │
│  │ Disk     │  │  ◀ First | ◀ Prev | Next ▶ | ▶▶ │   │
│  │ FPS      │  │  🔍+ | Reset | 🔍- | Auto ☑    │   │
│  └─────────┘  └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### Sidebar Metrics (left column)

| Metric | Description |
|--------|-------------|
| **Shipment** | Current batch/shipment ID (click to edit) |
| **Encoder** | Current encoder position (pulses) |
| **Speed** | Conveyor speed in PPM (pulses per minute) |
| **Pulses/sec** | Raw pulse rate |
| **Movement** | Green circle = moving, Red circle = stopped |
| **Ejector Queue** | Items waiting to be ejected |
| **Ejector Active** | Whether ejector is enabled |
| **Ejector Offset** | Encoder delay from camera to ejector |
| **OK Counter** | Items that passed inspection |
| **NG Counter** | Items that failed inspection |
| **Ej OK / Ej NG** | Software-counted ejector confirmations |
| **Downtime** | Seconds since last movement |
| **Analog** | Analog sensor reading |
| **Capture Mode** | Current capture state name |
| **Inference** | Active AI service (YOLO/Gradio) |
| **Latency** | Inference time per frame (ms) |
| **Inf FPS** | Inference frames per second |
| **Cap FPS** | Capture frames per second |
| **CPU / RAM / Disk** | System resource usage |

### Timeline (right column)

The timeline shows a grid of captured images from all cameras:
- **Columns** = capture events (newest on the right)
- **Rows** = cameras (one row per camera)
- **Header strip** = encoder value and timestamp for each column
- **Green/Red bounding boxes** = detected objects overlaid on thumbnails

### Timeline Controls

| Control | Action |
|---------|--------|
| **◀◀ First** | Jump to the oldest page |
| **◀ Prev** | Go to the previous page |
| **Next ▶** | Go to the next page |
| **Last ▶▶** | Jump to the newest page |
| **Zoom In 🔍+** | Enlarge timeline images |
| **Reset** | Reset zoom to default |
| **Zoom Out 🔍-** | Shrink timeline images |
| **Auto-update ☑** | Automatically refresh (resumes after 30s of inactivity) |

### Clicking a Timeline Image

Click any image in the timeline to open a **full-resolution popup** showing:
- The raw camera frame with bounding boxes drawn from that specific capture
- Encoder value, timestamp, and camera ID
- Links to: **Gallery** (browse folder), **Download Raw**, **Download Annotated**

---

## 4. AI Assistant Tab 🤖

Chat with an AI model about your production data and quality metrics.

### Setup

1. Go to **Advanced** tab → **AI Configuration**
2. Add an AI model:
   - **Provider**: Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google), or Local
   - **API Key**: Your provider's API key
3. Click **Save Model**, then activate it

### Usage

- Type natural language questions in the chat box
- Example queries:
  - "What's the defect rate for the last hour?"
  - "Show me the most common defect types today"
  - "Compare production speed vs quality metrics"
  - "Analyze trends in the last 24 hours"
- Click **Clear Chat History** to reset the conversation

---

## 5. Gallery Tab 🖼️

Browse all captured images using PiGallery2 (port 5000).

- Images are organized by: `shipment / date_hour / filename`
- Supports search, filtering, and thumbnail browsing
- View raw captures and annotated (detected) images side by side
- Access directly at `http://<server-ip>:5000`

---

## 6. Charts Tab 📈

View production metrics and analytics using Grafana (port 3000).

### First-Time Setup

1. Open Charts tab (or go to `http://<server-ip>:3000`)
2. Login with default credentials: **admin / admin**
3. The TimescaleDB datasource is pre-configured
4. Create dashboards using data from:
   - `production_metrics` table: encoder, counters, speed, movement, downtime
   - `inference_results` table: detections, inference time, model used, image paths

### Key Metrics Available

| Table | Column | Description |
|-------|--------|-------------|
| `production_metrics` | `encoder_value` | Encoder position over time |
| `production_metrics` | `ok_counter` / `ng_counter` | Pass/fail counts |
| `production_metrics` | `is_moving` | Conveyor belt status |
| `production_metrics` | `downtime_seconds` | Idle time tracking |
| `inference_results` | `detection_count` | Objects detected per frame |
| `inference_results` | `inference_time_ms` | AI processing latency |
| `inference_results` | `model_used` | Which model processed the frame |
| `inference_results` | `detections` | Full detection JSON (classes, bboxes, confidence) |

---

## 7. Hardware Tab 🔧

Control physical hardware connected via serial (Arduino/PLC).

### Light Controls

| Button | Effect |
|--------|--------|
| **Both On** | Upper + Bottom lights on |
| **U On, B Off** | Upper light only |
| **B On, U Off** | Bottom light only |
| **Both Off** | All lights off |
| **Upper PWM** | Set upper light brightness (0-255) |
| **Bottom PWM** | Set bottom light brightness (0-255) |

### Encoder

- **Reset Encoder** - Resets the encoder counter to 0
- The encoder value tracks conveyor belt position in pulses

### Warning LED

- **Warning On / Off** - Control the external warning indicator

---

## 8. Cameras Tab 📷

### USB Cameras

USB cameras are auto-detected on startup via V4L2. Each camera shows:
- Live preview stream
- Camera ID and device path

### IP Camera Discovery

1. Enter your network **subnet** (e.g., `192.168.0`)
2. Click **Scan Network**
3. The system scans common RTSP/HTTP ports (554, 8554, 80, 8080) and tests known URL patterns for:
   - Hikvision, Dahua, Axis, Reolink, Foscam, and generic ONVIF cameras
4. Discovered cameras appear with a **preview** - click **Save** to add them

### Per-Camera Settings

| Setting | Description |
|---------|-------------|
| **FPS** | Capture frame rate |
| **Resolution** | Width x Height |
| **Exposure** | Camera exposure value |
| **Gain** | Camera gain/sensitivity |
| **Brightness** | Image brightness (0-255) |
| **Contrast** | Image contrast (0-255) |
| **Saturation** | Color saturation (0-255) |
| **ROI Enabled** | Crop to a region of interest |
| **ROI xmin/ymin/xmax/ymax** | Crop coordinates |

### IP Camera URL Formats

**Hikvision:**
```
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101    # Main stream
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102    # Sub stream
```

**Dahua:**
```
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0   # Main
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1   # Sub
```

**Axis:**
```
rtsp://root:password@192.168.1.100/axis-media/media.amp
http://192.168.1.100/mjpg/video.mjpg
```

**Generic RTSP:**
```
rtsp://username:password@192.168.1.100:554/stream1
```

### Capture States

States define **how and when** images are captured. Each state has one or more **phases**:

| Phase Setting | Description |
|---------------|-------------|
| **Light Mode** | Which lights to use: `U_ON_B_OFF`, `B_ON_U_OFF`, `U_ON_B_ON`, `U_OFF_B_OFF` |
| **Delay** | Seconds to wait after setting lights before capturing (e.g., `0.13`) |
| **Cameras** | Which camera IDs to capture (e.g., `1, 2`) |
| **Steps** | Capture every N encoder pulses. `-1` = continuous loop |
| **Analog** | Analog sensor trigger threshold. `-1` = disabled |

#### Example States

**Encoder-triggered (every 100 pulses, uplight):**
```
Name: encoder-100
Phases:
  1. Light: U_ON_B_OFF | Delay: 0.13s | Cameras: 1,2 | Steps: 100
```

**Continuous capture (as fast as possible):**
```
Name: infinite-max
Phases:
  1. Light: U_ON_B_OFF | Delay: 0.0s | Cameras: 1,2 | Steps: -1
```

**Multi-phase (uplight then backlight):**
```
Name: dual-light
Phases:
  1. Light: U_ON_B_OFF | Delay: 0.13s | Cameras: 1,2 | Steps: 50
  2. Light: B_ON_U_OFF | Delay: 0.13s | Cameras: 1,2 | Steps: 50
```

---

## 9. Inference Tab 🔮

Manage AI models and detection pipelines.

### Pipeline Status

The colored dot at the top shows the current pipeline status:
- **Green** = active and processing
- **Red** = stopped or error

### Managing Models

1. Click **Create/Edit Model**
2. Fill in:
   - **Model Name**: human-readable name (e.g., "Defect Detector")
   - **Type**: `YOLO` (local GPU) or `Gradio` (remote HuggingFace server)
   - **Inference URL**: API endpoint
     - YOLO: `http://yolo_inference:4442/v1/object-detection/yolov5s/detect/`
     - Gradio: your HuggingFace endpoint URL
   - **Min Confidence**: detection threshold (0.0 - 1.0)
3. Click **Create/Update**

### Uploading YOLO Weights

1. Train your model at [ai-trainer.monitait.com](https://ai-trainer.monitait.com)
2. In the Inference tab, click **Upload .pt file**
3. Select your `best.pt` file
4. Click **Activate Weights** to load on all YOLO replicas

### Managing Pipelines

Pipelines chain multiple models together:

1. Click **Create/Edit Pipeline**
2. Set **Pipeline Name** and **Description**
3. Select models from the checklist (they run in order)
4. Click **Create/Update**
5. Click **Activate** next to your pipeline to start using it

**Single model example:**
```
Pipeline: "Quality Check"
  → Model 1: "YOLO Defect Detector" (confidence: 0.3)
```

**Multi-model example:**
```
Pipeline: "Full Inspection"
  → Model 1: "YOLO Object Detector" (detect box, product)
  → Model 2: "Gradio Classifier" (classify defect type)
```

---

## 10. Process Tab ⚙️

Configure ejection, detection alerts, and image processing — all in one place.

### Ejector Configuration

| Setting | Description |
|---------|-------------|
| **Ejector Enabled** | Toggle ejector on/off. When disabled, detections are recorded but no ejection occurs |
| **Ejector Offset** | Encoder counts from camera position to ejector position |
| **Ejector Duration** | How long to activate the ejector (seconds) |

### OK Configuration

| Setting | Description |
|---------|-------------|
| **OK Counter Adjustment** | Manually adjust the OK counter value |
| **Offset Delay (ms)** | Time delay from detection to OK signal |
| **Duration Pulses** | Output pulse length (each pulse = 16 microseconds) |
| **Duration Percent (0-100)** | Duty cycle of the output signal |
| **Encoder Factor** | Scaling multiplier for encoder-based timing |

### NG Configuration

Same settings as OK but for rejected items (NG = Not Good):

| Setting | Description |
|---------|-------------|
| **NG Counter Adjustment** | Manually adjust the NG counter value |
| **Offset Delay (ms)** | Time delay from detection to NG ejection signal |
| **Duration Pulses** | Ejection pulse length |
| **Duration Percent (0-100)** | Duty cycle |
| **Encoder Factor** | Scaling multiplier |

### Ejection Procedures

Define rules for when to eject items:

```json
{
  "name": "Reject Defects",
  "enabled": true,
  "logic": "any",
  "rules": [
    {"object": "crack", "condition": "present", "min_confidence": 80},
    {"object": "scratch", "condition": "present", "min_confidence": 70}
  ]
}
```

- **logic: "any"** = eject if ANY rule matches
- **logic: "all"** = eject only if ALL rules match
- **condition: "present"** = object is detected
- **condition: "not_present"** = object is NOT detected

### Per-Object Detection & Alerts

1. Click **Fetch Classes** to load object classes from the active model
2. For each detected object class, configure:

| Setting | Description |
|---------|-------------|
| **Show** | Toggle visibility of this class in the timeline |
| **Min Confidence** | Minimum confidence to display (0.0 - 1.0) |
| **Audio Alert** | Enable audio notification when detected |
| **Audio File** | Select which sound to play |

### Image Processing Configuration

| Setting | Description |
|---------|-------------|
| **Parent Object List** | Comma-separated parent objects for hierarchy checking (e.g., `_root,box,pack`) |
| **Remove Raw Images** | Auto-delete raw images after processing to save disk space |

### DataMatrix Configuration

| Setting | Description |
|---------|-------------|
| **Valid Char Sizes** | Accepted DataMatrix character lengths (e.g., `13,19,26`) |
| **Confidence Threshold** | Minimum match confidence for DataMatrix |
| **Overlap Threshold** | IoU threshold for duplicate detection |

### Feature Toggles

| Setting | Description |
|---------|-------------|
| **Histogram Enabled** | Enable quality distribution histogram analysis |
| **Light Status Check** | Verify light status before capture |

---

## 11. Advanced Tab ⚡

Fine-tune system behavior, manage data, and configure external services.

### Timeline Configuration

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| **Camera Order** | Ascending / Descending | Ascending | Order of camera rows in timeline |
| **Image Rotation** | 0, 90, 180, 270 | 0 | Rotate all timeline images |
| **Image Quality** | 50-100% | 85 | JPEG quality for timeline thumbnails |
| **Rows per Page** | 1-50 | 20 | How many capture columns per page |
| **Total Frames Stored** | 10-5000 | 100 | Maximum frames kept in Redis |

Click **Apply Timeline Configuration** to save.

### Redis Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| **Redis Host** | `redis` | Redis server hostname |
| **Redis Port** | `6379` | Redis server port |

### AI Configuration

Add AI models for the AI Assistant tab:

| Field | Description |
|-------|-------------|
| **Model Name** | Display name (e.g., "Claude") |
| **Provider** | `Claude`, `ChatGPT`, `Gemini`, or `Local` |
| **API Key** | Your provider API key |

### Database Configuration

Add TimescaleDB connection profiles:

| Field | Default | Description |
|-------|---------|-------------|
| **Profile Name** | "Default" | Profile display name |
| **Host** | `timescaledb` | Database hostname |
| **Port** | `5432` | Database port |
| **Database** | `monitaqc` | Database name |
| **User** | `monitaqc` | Database username |
| **Password** | `monitaqc2024` | Database password |

### Data File Editor

The raw configuration JSON that controls product matching and prepared query data:

```json
[
  {
    "dm": "6263957101037",
    "chars": [["box"], ["logo_en", "logo_fa"], ["product_name"]]
  }
]
```

- **dm**: DataMatrix barcode value to match
- **chars**: Expected object classes organized by hierarchy level
- Use **Export Config** / **Import Config** to backup and restore

### Global Configuration Parameters

These can be set via the data file or API:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ejector_enabled` | `true` | Enable automatic ejection |
| `ejector_offset` | `1` | Encoder counts from camera to ejector |
| `ejector_duration` | `0.01` | Ejection pulse duration (seconds) |
| `capture_mode` | `single` | `single` or `multiple` camera capture |
| `parent_object_list` | `_root,box,pack` | Parent objects for hierarchy checking |
| `enforce_parent_object` | `true` | Require parent object in detections |
| `dm_chars_sizes` | `13,19,26` | Valid DataMatrix character lengths |
| `dm_confidence_threshold` | `0.8` | DataMatrix match confidence |
| `histogram_enabled` | `true` | Enable histogram analysis |
| `store_annotation_enabled` | `false` | Save detections to TimescaleDB |
| `check_class_counts_enabled` | `true` | Validate expected object counts |
| `check_class_counts_classes` | `socket,nozzle` | Classes to count |
| `check_class_counts_confidence` | `0.5` | Count check confidence threshold |

---

## 11. Driver & GPU Setup

### Checking GPU Status

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Updating NVIDIA Driver (Linux)

```bash
# Check available versions
apt list nvidia-driver-* 2>/dev/null | grep -v rc

# Install specific version
sudo apt update
sudo apt install -y nvidia-driver-550

# Reboot required
sudo reboot

# Verify
nvidia-smi
```

### Updating NVIDIA Container Toolkit

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Updating Docker

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify
docker --version
docker compose version
```

### After Driver Updates

Always restart MonitaQC after updating drivers:

```bash
cd /path/to/monitaqc
docker compose down
./start.sh
```

---

## 12. Troubleshooting

### Service Health Check

```bash
# Check all containers
docker compose ps

# Check health endpoint
curl http://localhost/health

# View logs
docker logs monitait_vision_engine --tail 50
docker logs yolo_inference --tail 50
docker logs monitait_redis --tail 50
docker logs monitait_timescaledb --tail 50
```

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| YOLO inference not responding | GPU not available to Docker | Install NVIDIA Container Toolkit, restart Docker |
| Cameras not detected | Missing device access | Set `PRIVILEGED=true` in `.env` (Linux) |
| Timeline empty | No active capture state | Go to Cameras tab, activate a capture state |
| FPS shows 0 | Capture delay too high | Reduce `delay` in the active capture state |
| Config lost on restart | Didn't save | Click "Save All Configuration" in the header |
| Grafana permission denied | Volume permissions | Add `user: "0"` to grafana service in docker-compose.yml |
| Redis connection refused | Redis not started | Check `docker compose ps`, restart redis service |
| Images all identical | Camera buffer caching | Normal for rapid continuous capture of same scene |

### Viewing Real-Time Logs

```bash
# Follow vision engine logs
docker logs -f monitait_vision_engine

# Follow all services
docker compose logs -f
```

### Restarting Services

```bash
# Restart a single service
docker compose restart monitait_vision_engine

# Restart everything
docker compose down && docker compose up -d

# Full rebuild (after code changes)
docker compose up -d --build monitait_vision_engine
```

### Resetting Configuration

To reset all settings to defaults:

```bash
# Remove the configuration file
rm .env.prepared_query_data

# Restart
docker compose restart monitait_vision_engine
```

---

## Quick Reference

### Access Points

| Service | URL |
|---------|-----|
| **Web Interface** | `http://<server-ip>` |
| **Health Check** | `http://<server-ip>/health` |
| **Image Gallery** | `http://<server-ip>:5000` |
| **Grafana** | `http://<server-ip>:3000` (admin/admin) |
| **Redis CLI** | `docker exec -it monitait_redis redis-cli` |
| **Database** | `docker exec -it monitait_timescaledb psql -U monitaqc` |

### Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service orchestration |
| `.env` | Auto-generated environment variables |
| `.env.prepared_query_data` | Persisted configuration (auto-created) |
| `VERSION` | Application version (single source of truth) |
| `volumes/weights/best.pt` | Active YOLO model weights |
| `start.py` | Hardware detection & auto-tune launcher |

### Save Configuration Checklist

After making changes, always:

1. Click **"Save All Configuration"** in the top-right header
2. Verify the **"Last saved"** timestamp updated
3. Settings are now persisted in `.env.prepared_query_data` and survive restarts
