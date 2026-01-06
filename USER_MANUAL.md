# MonitaQC Vision Engine - Comprehensive User Manual
**Version:** 1.0.0
**Last Updated:** 2026-01-06
**Application URL:** http://localhost:5050

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation & Setup](#3-installation--setup)
4. [Quick Start Guide](#4-quick-start-guide)
5. [Main Interface Overview](#5-main-interface-overview)
6. [Configuration Sections](#6-configuration-sections)
7. [Camera Setup](#7-camera-setup)
8. [YOLO AI Detection](#8-yolo-ai-detection)
9. [Audio Alerts](#9-audio-alerts)
10. [Image Processing](#10-image-processing)
11. [Timeline & Review](#11-timeline--review)
12. [Database Integration](#12-database-integration)
13. [Advanced Features](#13-advanced-features)
14. [Troubleshooting](#14-troubleshooting)
15. [Best Practices](#15-best-practices)
16. [FAQ](#16-faq)

---

## 1. Introduction

### What is MonitaQC?

MonitaQC Vision Engine is an industrial-grade quality control system that uses AI-powered computer vision to detect defects, count objects, and monitor production lines in real-time.

### Key Features

- **AI Object Detection** - YOLO-based defect detection with customizable models
- **Multi-Camera Support** - Monitor multiple cameras simultaneously
- **Real-time Audio Alerts** - Voice narration and beep sounds for detected objects
- **Timeline Playback** - Review captured images with frame-by-frame navigation
- **Database Storage** - PostgreSQL/TimescaleDB integration for annotation storage
- **Automatic Disk Cleanup** - Self-managing storage to prevent disk full
- **Serial Communication** - Arduino/PLC integration for motor control and sensors
- **Web Interface** - Browser-based control panel, no installation needed

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MonitaQC System                          │
├─────────────────────────────────────────────────────────────┤
│  Vision Engine (Port 5050)                                  │
│  ├─ Camera Capture & Processing                             │
│  ├─ Serial Communication (Arduino/PLC)                      │
│  ├─ Image Storage & Timeline                                │
│  └─ Web Interface                                           │
├─────────────────────────────────────────────────────────────┤
│  YOLO Inference Service (Port 4442)                         │
│  └─ AI Object Detection                                     │
├─────────────────────────────────────────────────────────────┤
│  Redis Cache (Port 6379)                                    │
│  └─ Real-time Event Queue                                   │
├─────────────────────────────────────────────────────────────┤
│  TimescaleDB (Port 5432)                                    │
│  └─ Annotation Storage                                      │
├─────────────────────────────────────────────────────────────┤
│  Grafana (Port 3000)                                        │
│  └─ Metrics Dashboard                                       │
├─────────────────────────────────────────────────────────────┤
│  Cleanup Service                                            │
│  └─ Automated Disk Space Management                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Requirements

### Hardware

- **CPU:** Intel/AMD x86_64, 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 500GB+ SSD for image storage
- **Camera:** USB/IP camera with V4L2/DirectShow support
- **Network:** 1 Gbps Ethernet for IP cameras (optional)

### Software

- **OS:** Linux (Ubuntu 20.04+), Windows 10+, or macOS
- **Docker:** Docker 20.10+ and Docker Compose 1.29+
- **Browser:** Chrome 90+, Firefox 88+, or Edge 90+
- **Python:** 3.10+ (for development only)

### Optional Hardware

- **Arduino/PLC:** For motor control and sensor integration
- **Serial Port:** USB-to-Serial adapter if needed
- **Ejector Mechanism:** Pneumatic actuator for defect rejection

---

## 3. Installation & Setup

### Step 1: Clone the Repository

```bash
cd /projects
git clone http://gitlab.virasad.ir/monitait/monitaqc.git
cd MonitaQC
```

### Step 2: Configure Environment

No manual environment configuration needed! All settings are managed through the web interface at http://localhost:5050/status.

### Step 3: Start the System

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs (optional)
docker logs -f monitait_vision_engine
```

### Step 4: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5050
```

You should see the MonitaQC Vision Engine interface.

### Step 5: Verify Services

Check that all services are running:

| Service | URL | Expected Response |
|---------|-----|-------------------|
| Vision Engine | http://localhost:5050 | Web interface loads |
| YOLO Service | http://localhost:4442/docs | Swagger API docs |
| Grafana | http://localhost:3000 | Login page (admin/admin) |
| TimescaleDB | localhost:5432 | Database connection |
| Redis | localhost:6379 | Redis PING responds |

---

## 4. Quick Start Guide

### Your First Detection

1. **Open the Interface**
   ```
   http://localhost:5050
   ```

2. **Configure Camera (Infrastructure Tab)**
   - Click **Infrastructure** tab
   - Set camera type: `USB` or `IP`
   - Set camera device/URL
   - Click **Save Service Config**

3. **Configure YOLO Model (YOLO Config Tab)**
   - Click **YOLO Config** tab
   - Set API Type: `Gradio API` or `Traditional YOLO`
   - Set YOLO URL (e.g., `http://monitait_yolo:4442`)
   - Set model name (e.g., `best.pt`)
   - Click **Save Service Config**

4. **Start Detection**
   - Click the **▶ START** button in the header
   - Watch the live stream appear
   - Detections will be highlighted with bounding boxes

5. **Review Results**
   - Click **Timeline** tab to review captured images
   - Use navigation buttons to browse frames
   - Check detection statistics in the sidebar

---

## 5. Main Interface Overview

### Header Section

```
┌───────────────────────────────────────────────────────────┐
│  MonitaQC Vision Engine              [Status Indicators]  │
│  ┌──────┐  ┌──────┐  ┌──────┐       [CPU] [MEM] [FPS]   │
│  │START │  │RESTART│  │ SAVE │                            │
│  └──────┘  └──────┘  └──────┘       [Detection: ON]      │
└───────────────────────────────────────────────────────────┘
```

**Control Buttons:**
- **▶ START** - Start the vision system (capture + detection)
- **🔄 RESTART** - Restart all services
- **💾 SAVE** - Save current configuration to disk

**Status Indicators:**
- **CPU Usage** - Current CPU percentage
- **Memory Usage** - RAM usage in MB
- **FPS** - Frames per second
- **Detection Status** - ON/OFF indicator

### Tab Navigation

| Tab | Purpose | Key Features |
|-----|---------|-------------|
| **Status** | Live view & stats | Real-time stream, detection counts, status |
| **Infrastructure** | Camera & serial config | Camera setup, serial port, motor control |
| **YOLO Config** | AI model settings | Model selection, confidence threshold, ROI |
| **Image Processing** | Capture settings | Image quality, rotation, resolution |
| **Timeline** | Review mode | Frame-by-frame navigation, slideshow |
| **Advanced** | Expert settings | Timeline buffer, audio, database, data file |

---

## 6. Configuration Sections

### Infrastructure Configuration

**Location:** Infrastructure tab

#### Camera Configuration

1. **Camera Type**
   - `USB Camera (V4L2)` - For USB webcams (Linux)
   - `USB Camera (DirectShow)` - For USB webcams (Windows)
   - `IP Camera (RTSP)` - For network cameras

2. **Camera Device/URL**
   - USB: `/dev/video0` (Linux) or `0` (Windows)
   - IP: `rtsp://username:password@192.168.1.100:554/stream`

3. **Camera FPS**
   - Default: `30`
   - Range: 1-60 fps
   - Lower FPS = less CPU usage

4. **Camera Resolution**
   - Options: `640x480`, `1280x720`, `1920x1080`, `3840x2160`
   - Higher resolution = better quality, more storage

#### Serial Port Configuration

1. **Serial Port**
   - Linux: `/dev/ttyUSB0`, `/dev/ttyACM0`
   - Windows: `COM3`, `COM4`

2. **Baud Rate**
   - Default: `115200`
   - Common: 9600, 19200, 38400, 57600, 115200

3. **Motor & Light Control**
   - **PWM Backlight** - Bottom LED intensity (0-255)
   - **PWM Uplight** - Top LED intensity (0-255)
   - **Motor Speed** - Conveyor belt speed

#### Ejector Configuration

1. **Ejector Pin**
   - Arduino digital pin for solenoid control
   - Default: `6`

2. **Ejector Mode**
   - `On Datamatrix` - Eject if datamatrix detected
   - `On NG Detection` - Eject if defect detected
   - `Off` - Disable automatic ejection

3. **Ejector Delay**
   - Milliseconds to wait before triggering ejector
   - Compensates for conveyor travel time

---

## 7. Camera Setup

### USB Camera Setup

**Linux (V4L2):**
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
ffplay /dev/video0
```

**In MonitaQC:**
1. Set Camera Type: `USB Camera (V4L2)`
2. Set Camera Device: `/dev/video0`
3. Set Resolution: `1280x720`
4. Set FPS: `30`
5. Click **Save Service Config**
6. Click **▶ START**

**Windows (DirectShow):**
```powershell
# List cameras
ffmpeg -list_devices true -f dshow -i dummy
```

**In MonitaQC:**
1. Set Camera Type: `USB Camera (DirectShow)`
2. Set Camera Device: `0` (or camera name)
3. Set Resolution: `1280x720`
4. Set FPS: `30`
5. Click **Save Service Config**
6. Click **▶ START**

### IP Camera Setup (RTSP)

**In MonitaQC:**
1. Set Camera Type: `IP Camera (RTSP)`
2. Set Camera URL:
   ```
   rtsp://admin:password@192.168.1.100:554/stream1
   ```
3. Set Resolution: `1920x1080`
4. Set FPS: `25`
5. Click **Save Service Config**
6. Click **▶ START**

**Common RTSP Formats:**
- Hikvision: `rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101`
- Dahua: `rtsp://admin:admin@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
- Axis: `rtsp://root:pass@192.168.0.90/axis-media/media.amp`
- Generic: `rtsp://username:password@ip:port/stream`

### Multi-Camera Configuration

To monitor multiple cameras, configure them individually in the **Cameras** section (coming in future version). Currently, one camera is active per vision engine instance.

---

## 8. YOLO AI Detection

### API Type Selection

MonitaQC supports two YOLO API types:

#### 1. Gradio API (Recommended)

**Best for:** HuggingFace Spaces, Gradio apps, custom inference endpoints

**Configuration:**
```yaml
API Type: Gradio API
YOLO Inference URL: https://user-space.hf.space
Gradio Model Name: best.pt
Confidence Threshold: 0.3
```

**Example:** Using HuggingFace Industrial Defect Detection
```
URL: https://shomit-industrial-defect-detection.hf.space
Model: best.pt
```

#### 2. Traditional YOLO API

**Best for:** Self-hosted YOLO services, internal APIs

**Configuration:**
```yaml
API Type: Traditional YOLO
YOLO Inference URL: http://monitait_yolo:4442
Confidence Threshold: 0.3
IOU Threshold: 0.4
```

### Model Configuration

1. **Confidence Threshold**
   - Range: 0.0 - 1.0
   - Default: 0.3
   - Higher value = fewer false positives, may miss defects
   - Lower value = more detections, may include false positives

2. **IOU Threshold** (Traditional YOLO only)
   - Range: 0.0 - 1.0
   - Default: 0.4
   - Controls overlap tolerance for duplicate detections

### Custom YOLO Models

To use your own trained model:

1. **Train Your Model**
   - Use YOLOv5, YOLOv7, or YOLOv8
   - Export as `.pt` (PyTorch) format

2. **Deploy via Gradio**
   - Create a Gradio interface wrapping your model
   - Deploy to HuggingFace Spaces or self-host
   - Use the Gradio URL in MonitaQC

3. **Or Deploy Traditional YOLO**
   ```bash
   # Copy model to weights directory
   cp your_model.pt c:/projects/MonitaQC/volumes/weights/

   # Update config
   Model Name: your_model.pt
   ```

### Region of Interest (ROI)

**Enable ROI:**
1. Set ROI Enabled: `true`
2. Define ROI Coordinates:
   ```json
   {
     "x": 100,
     "y": 100,
     "width": 800,
     "height": 600
   }
   ```

**Use Case:** Ignore detections outside the production line area.

---

## 9. Audio Alerts

### Overview

MonitaQC can announce detected objects using voice narration and play beep sounds in real-time.

**Location:** Advanced tab → Audio Alerts Configuration

### Global Audio Controls

1. **Enable Voice Narration**
   - ☑ Turn on to hear object names spoken
   - Example: "box", "defect", "crack"
   - Uses browser's Text-to-Speech engine

2. **Enable Beep Sounds**
   - ☑ Turn on to hear beep when objects detected
   - 4 different waveforms available per object

3. **Volume**
   - Slider: 0% - 100%
   - Default: 50%
   - Affects both narration and beeps

4. **Test Audio**
   - Click 🔊 **Test Audio** to verify audio is working
   - Plays a test beep

### Per-Object Configuration

Each detected object can have individual settings:

```
┌──────────────────────────────────────────┐
│  ☑ box                    [Sine ▼] [▶Test] │
│  ☐ defect               [Square ▼] [▶Test] │
│  ☑ crack              [Sawtooth ▼] [▶Test] │
│  ☐ scratch            [Triangle ▼] [▶Test] │
└──────────────────────────────────────────┘
```

**Configuration:**
1. **Checkbox** - Enable/disable audio for this object
2. **Beep Sound** - Choose waveform:
   - 🔔 **Sine** - Bell-like sound (smooth)
   - 📢 **Square** - Buzzer sound (harsh)
   - 🎺 **Sawtooth** - Horn sound (bright)
   - 🎵 **Triangle** - Soft tone (mellow)
3. **Test Button** - Play the selected beep sound

### Audio Settings Tips

- **Noisy Environment:** Use Square wave for louder, more noticeable beeps
- **Quiet Office:** Use Sine or Triangle for less intrusive alerts
- **Multiple Objects:** Assign different waveforms to distinguish objects by sound
- **Disable Unwanted:** Uncheck objects you don't want to hear (e.g., "box" if too frequent)

### Enable All Objects

Click **✅ Enable All** to quickly enable audio for all detected objects.

---

## 10. Image Processing

### Capture Configuration

**Location:** Image Processing tab

#### Image Quality

- **Range:** 1-100
- **Default:** 95
- **Storage Impact:**
  - Quality 100: ~500KB per image
  - Quality 80: ~150KB per image
  - Quality 50: ~50KB per image

**Recommendation:** Use 80-95 for production, 100 for quality assurance.

#### Image Rotation

- **Options:** 0°, 90°, 180°, 270°
- **Use Case:** Correct camera mounting orientation
- **Default:** 0° (no rotation)

#### Flip Options

- **Horizontal Flip:** Mirror left-right
- **Vertical Flip:** Mirror top-bottom

#### Color Space

- **RGB** - Color images (default)
- **Grayscale** - Black & white images (smaller file size)

#### Image Format

- **JPEG** - Lossy compression (smaller, default)
- **PNG** - Lossless compression (larger, better quality)

### Capture Modes

#### 1. Continuous Capture

Captures images at regular intervals regardless of detections.

```yaml
Capture Mode: Continuous
Capture Interval: 1000ms  # Every 1 second
```

#### 2. Event-Based Capture

Captures images only when objects are detected or serial triggers occur.

```yaml
Capture Mode: Event-Based
Min Capture Interval: 500ms  # Minimum time between captures
```

#### 3. Encoder-Based Capture

Captures images based on encoder position (for conveyor belt systems).

```yaml
Capture Mode: Encoder
Encoder Step: 100  # Capture every 100 encoder ticks
```

---

## 11. Timeline & Review

### Overview

The Timeline feature allows you to review all captured images with frame-by-frame navigation.

**Location:** Timeline tab

### Timeline Interface

```
┌────────────────────────────────────────────────────────┐
│  ← Previous   [Frame 45/100]   Next →   ▶ Slideshow   │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                    │ │
│  │           [Captured Image Display]                │ │
│  │                                                    │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  Timestamp: 2026-01-06 14:32:15                       │
│  Detections: 3 objects (box, defect, crack)           │
│  Frame ID: 20260106_143215_001234                     │
└────────────────────────────────────────────────────────┘
```

### Navigation Controls

| Button | Action | Keyboard Shortcut |
|--------|--------|-------------------|
| **← Previous** | Go to previous frame | Left Arrow |
| **Next →** | Go to next frame | Right Arrow |
| **▶ Slideshow** | Start auto-play | Space |
| **⏸ Pause** | Stop slideshow | Space |
| **🔍 Zoom In** | Zoom into image | + |
| **🔍 Zoom Out** | Zoom out of image | - |
| **🔄 Reset** | Reset zoom level | 0 |

### Slideshow Mode

1. Click **▶ Slideshow** to start auto-play
2. Images advance automatically every 2 seconds
3. Click **⏸ Pause** or press Space to stop

### Frame Information

Each frame shows:
- **Timestamp** - When the image was captured
- **Detections** - List of detected objects with confidence scores
- **Frame ID** - Unique identifier for this frame
- **File Path** - Location on disk (for debugging)

### Timeline Configuration (Advanced Tab)

1. **Buffer Size**
   - Number of frames to keep in memory
   - Default: 500 frames
   - Higher = more history, more RAM usage
   - **Note:** May be limited by image quality (see Audit Report)

2. **Auto-cleanup**
   - Automatically remove old frames beyond buffer size
   - Enabled by default

3. **Export Timeline**
   - Download all frames as ZIP archive
   - Includes CSV with detection metadata

---

## 12. Database Integration

### Overview

MonitaQC can store detection annotations in TimescaleDB (PostgreSQL) for long-term analysis and reporting.

**Location:** Advanced tab → Database Storage Configuration

### Enable Database Storage

1. **PostgreSQL Connection URI**
   ```
   postgresql://monitaqc:monitaqc2024@timescaledb:5432/monitaqc
   ```
   - Format: `postgresql://user:password@host:port/database`
   - Default uses docker-compose service name `timescaledb`

2. **Store Annotations to Database**
   - Set to: `Enabled`
   - Click **Set**

### What Gets Stored

Each detection creates a database record with:
- **Timestamp** - When the detection occurred
- **Frame ID** - Link to the captured image
- **Object Class** - Type of object detected (e.g., "defect")
- **Confidence Score** - AI confidence (0.0-1.0)
- **Bounding Box** - Coordinates `(x, y, width, height)`
- **Image Path** - File system location

### Query Annotations

Connect to the database:
```bash
# Using psql
docker exec -it monitait_timescaledb psql -U monitaqc -d monitaqc

# Query detections
SELECT * FROM detections WHERE timestamp > NOW() - INTERVAL '1 hour';

# Count detections by object class
SELECT object_class, COUNT(*)
FROM detections
GROUP BY object_class
ORDER BY COUNT(*) DESC;

# Average confidence by hour
SELECT
  time_bucket('1 hour', timestamp) AS hour,
  AVG(confidence) AS avg_confidence
FROM detections
GROUP BY hour
ORDER BY hour DESC;
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

**Create a Detection Dashboard:**
1. Add TimescaleDB as data source
   - Host: `timescaledb:5432`
   - Database: `monitaqc`
   - User: `monitaqc`
   - Password: `monitaqc2024`

2. Create panels:
   - **Detections Over Time** - Line graph
   - **Detection Count by Class** - Pie chart
   - **Average Confidence** - Gauge
   - **Recent Detections** - Table

---

## 13. Advanced Features

### Data File Editor

**Location:** Advanced tab → Data File Editor

The Data File stores all system configuration as JSON.

**Features:**
- **View** - See current configuration
- **Edit** - Manually modify settings (expert use only!)
- **Format** - Pretty-print JSON for readability
- **Export** - Download config as `.json` file
- **Import** - Upload a previously exported config

**Use Cases:**
- Backup configuration before major changes
- Clone configuration to another system
- Bulk edit settings via JSON

**⚠️ Warning:** Invalid JSON will break the system! Always validate before saving.

### Timeline Configuration

**Location:** Advanced tab → Timeline Configuration

- **Buffer Size** - Maximum frames in memory (default: 500)
- **Image Quality** - Timeline image quality 1-100 (default: 90)
- **Auto-cleanup** - Remove old frames automatically

### Service Config Export/Import

Export your entire service configuration:

1. Click **⬇️ Export Config**
2. Save the `.json` file
3. To restore, click **⬆️ Import Config**
4. Select the saved file
5. Click **💾 Save**

---

## 14. Troubleshooting

### Camera Issues

**Problem:** No video feed appears

**Solutions:**
1. Check camera is connected:
   ```bash
   # Linux
   ls /dev/video*

   # Windows
   ffmpeg -list_devices true -f dshow -i dummy
   ```

2. Verify camera permissions:
   ```bash
   # Linux - add user to video group
   sudo usermod -aG video $USER
   ```

3. Test camera outside MonitaQC:
   ```bash
   ffplay /dev/video0
   ```

4. Check docker-compose.yml includes camera device:
   ```yaml
   devices:
     - /dev/video0:/dev/video0
   ```

**Problem:** Low FPS or laggy video

**Solutions:**
- Reduce camera resolution (1920x1080 → 1280x720)
- Lower FPS setting (30 → 15)
- Reduce image quality (95 → 80)
- Check CPU usage in header

### YOLO Detection Issues

**Problem:** No detections appearing

**Solutions:**
1. Check YOLO service is running:
   ```bash
   docker logs monitait_yolo
   ```

2. Verify YOLO URL is correct:
   ```bash
   curl http://monitait_yolo:4442/health
   ```

3. Lower confidence threshold (0.3 → 0.2)

4. Check model is loaded:
   ```bash
   docker exec monitait_yolo ls /weights
   ```

**Problem:** Too many false positives

**Solutions:**
- Increase confidence threshold (0.3 → 0.5)
- Increase IOU threshold (Traditional YOLO)
- Enable ROI to limit detection area
- Retrain model with better dataset

### Audio Issues

**Problem:** No audio plays

**Solutions:**
1. Click **Test Audio** button
2. Check browser audio is not muted
3. Enable audio in per-object configuration
4. Try a different browser (Chrome recommended)

**Problem:** Audio is distorted

**Solutions:**
- Lower volume slider
- Try different waveform (Square → Sine)
- Check system audio settings

### Storage Issues

**Problem:** Disk full errors

**Solutions:**
1. Check cleanup service is running:
   ```bash
   docker logs monitait_cleanup
   ```

2. Verify disk usage:
   ```bash
   df -h /mnt/SSD-RESERVE
   ```

3. Adjust cleanup thresholds in docker-compose.yml:
   ```yaml
   environment:
     - MAX_USAGE_PERCENT=85  # Start cleanup earlier
     - MIN_USAGE_PERCENT=75  # More aggressive cleanup
   ```

4. Lower image quality to reduce file sizes

### Database Issues

**Problem:** Annotations not saving to database

**Solutions:**
1. Check TimescaleDB is running:
   ```bash
   docker logs monitait_timescaledb
   ```

2. Verify connection URI:
   ```
   postgresql://monitaqc:monitaqc2024@timescaledb:5432/monitaqc
   ```

3. Test database connection:
   ```bash
   docker exec -it monitait_timescaledb psql -U monitaqc -d monitaqc -c "SELECT 1"
   ```

4. Check "Store Annotations" is set to **Enabled**

### General Issues

**Problem:** Services won't start

**Solutions:**
1. Check Docker is running:
   ```bash
   docker ps
   ```

2. View logs:
   ```bash
   docker compose logs
   ```

3. Restart all services:
   ```bash
   docker compose down
   docker compose up -d
   ```

4. Rebuild containers:
   ```bash
   docker compose up -d --build
   ```

**Problem:** Configuration not saving

**Solutions:**
- Always click **💾 SAVE** after making changes
- Check `.env.prepared_query_data` file permissions
- View logs for save errors:
  ```bash
  docker logs monitait_vision_engine | grep "save"
  ```

---

## 15. Best Practices

### Camera Configuration

- ✅ Use **1280x720** or **1920x1080** for best balance of quality and performance
- ✅ Set FPS to match your production line speed (15-30 fps typical)
- ✅ Use **fixed focus** cameras for consistent image quality
- ✅ Position camera perpendicular to production line for best results

### YOLO Configuration

- ✅ Start with **confidence threshold 0.3**, adjust based on results
- ✅ Use **Gradio API** for easier model deployment and updates
- ✅ Test models on representative samples before production
- ✅ Enable **ROI** to reduce false positives from background

### Image Storage

- ✅ Use **image quality 80-90** for production (good balance)
- ✅ Enable **automatic disk cleanup** to prevent storage issues
- ✅ Keep timeline buffer at **200-500 frames** (adjust for available RAM)
- ✅ Use **SSD storage** for best performance

### Audio Alerts

- ✅ Disable audio for **frequent objects** (e.g., "box" every second)
- ✅ Assign **different waveforms** to distinguish critical vs. normal detections
- ✅ Keep **volume at 50%** and adjust up if needed
- ✅ Test audio **before production** to ensure it's not disruptive

### Database Usage

- ✅ Enable database storage for **production deployments**
- ✅ Use **TimescaleDB** for time-series queries and analysis
- ✅ Create **Grafana dashboards** for real-time monitoring
- ✅ Set up **automated backups** of the database

### Serial Communication

- ✅ Test **motor control** in manual mode before automation
- ✅ Use **encoder-based capture** for precise synchronization
- ✅ Set **ejector delay** to account for conveyor travel time
- ✅ Monitor **serial errors** in logs for connectivity issues

---

## 16. FAQ

### Q: How many cameras can I monitor?

**A:** Currently, one camera per vision engine instance. To monitor multiple cameras, deploy multiple vision engine instances with different ports.

### Q: Can I use MonitaQC without a camera?

**A:** Yes! You can upload images via the API or use file-based input for testing. The system is primarily designed for real-time camera input.

### Q: What YOLO models are supported?

**A:** YOLOv5, YOLOv7, YOLOv8, and YOLOv9. Any YOLO model that can be deployed as a Gradio API or traditional REST API.

### Q: How do I train a custom model?

**A:** Train your model using YOLOv5/v7/v8 frameworks with your dataset, then deploy it via Gradio or the local YOLO inference service. See [YOLO AI Detection](#8-yolo-ai-detection) section.

### Q: Can I run this on a Raspberry Pi?

**A:** Yes, but performance will be limited. Recommended for prototyping only. Use an x86_64 system with GPU for production.

### Q: How much storage do I need?

**A:** Depends on image quality, resolution, and capture rate:
- **Low quality (50)**, 720p, 1 fps = ~10GB/day
- **Medium quality (80)**, 1080p, 5 fps = ~100GB/day
- **High quality (95)**, 1080p, 30 fps = ~1TB/day

Enable automatic disk cleanup to manage storage automatically.

### Q: Can I access MonitaQC remotely?

**A:** Yes, but you'll need to:
1. Forward port 5050 through your router
2. Use HTTPS with SSL certificate (recommended)
3. Set up authentication (not built-in, use reverse proxy)
4. Consider security implications

### Q: What if I don't have Arduino/PLC?

**A:** Serial communication is optional. You can use MonitaQC for detection-only without motor control or ejector.

### Q: How do I backup my configuration?

**A:** Export service config via **Advanced → Data File Editor → ⬇️ Export Config**. Save the JSON file securely.

### Q: Can I use multiple YOLO models?

**A:** Yes (coming in future version). Currently, one YOLO model is active per vision engine instance.

### Q: Where are images stored?

**A:** Images are stored in `/mnt/SSD-RESERVE/raw_images` (Linux) or `c:\projects\MonitaQC\volumes\raw_images` (Windows) by default. Configure via docker-compose.yml volumes.

### Q: How do I update MonitaQC?

**A:**
```bash
cd /projects/MonitaQC
git pull
docker compose down
docker compose up -d --build
```

### Q: Is there a mobile app?

**A:** No native app, but the web interface is mobile-responsive. Access via mobile browser.

### Q: Can I integrate with other systems?

**A:** Yes! MonitaQC provides REST API endpoints. See `/api/docs` for Swagger documentation (coming soon).

### Q: What license is MonitaQC?

**A:** Check the LICENSE file in the repository. Typically used internally for industrial applications.

---

## Appendix A: API Endpoints

### Status API

```bash
# Get system status
GET http://localhost:5050/api/status

# Get latest detections
GET http://localhost:5050/api/latest_detections

# Get camera feed (SSE stream)
GET http://localhost:5050/api/stream
```

### Configuration API

```bash
# Get current configuration
GET http://localhost:5050/api/config

# Update configuration
POST http://localhost:5050/api/config
Content-Type: application/json
{
  "key": "value"
}

# Save configuration to disk
POST http://localhost:5050/api/save
```

### Detection API

```bash
# Run detection on uploaded image
POST http://localhost:5050/api/detect
Content-Type: multipart/form-data
file: [image file]
```

---

## Appendix B: Configuration File Reference

The `.env.prepared_query_data` file contains all system settings as JSON. Key sections:

```json
{
  "camera_type": "usb",
  "camera_device": "/dev/video0",
  "camera_fps": 30,
  "camera_resolution": "1280x720",

  "serial_port": "/dev/ttyUSB0",
  "serial_baudrate": 115200,

  "yolo_url": "http://monitait_yolo:4442",
  "yolo_api_type": "gradio",
  "yolo_model_name": "best.pt",
  "yolo_confidence": 0.3,

  "image_quality": 90,
  "image_rotation": 0,

  "timeline_buffer_size": 500,

  "db_uri": "postgresql://monitaqc:monitaqc2024@timescaledb:5432/monitaqc",
  "store_annotations": true
}
```

---

## Appendix C: Docker Commands Reference

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Restart a specific service
docker compose restart monitait_vision_engine

# View logs
docker logs -f monitait_vision_engine

# View logs (last 100 lines)
docker logs --tail 100 monitait_vision_engine

# Execute command in container
docker exec -it monitait_vision_engine bash

# Check resource usage
docker stats

# Rebuild and restart
docker compose up -d --build

# Remove all stopped containers
docker compose down --volumes
```

---

## Appendix D: Glossary

- **YOLO** - You Only Look Once, a real-time object detection algorithm
- **ROI** - Region of Interest, a defined area for detection
- **FPS** - Frames Per Second, capture/processing rate
- **Confidence Threshold** - Minimum confidence score for detections
- **IOU** - Intersection Over Union, overlap metric for bounding boxes
- **SSE** - Server-Sent Events, real-time data streaming protocol
- **RTSP** - Real-Time Streaming Protocol, IP camera standard
- **TimescaleDB** - PostgreSQL extension for time-series data
- **Gradio** - Python library for building ML web interfaces

---

## Support & Contact

**Documentation:** This manual
**System Audit:** See AUDIT_REPORT.md
**Repository:** http://gitlab.virasad.ir/monitait/monitaqc
**Version:** 1.0.0
**Last Updated:** 2026-01-06

---

**© 2026 MonitaQC - Industrial Quality Control System**
