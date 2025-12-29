# MonitaQC - Universal Quality Control Platform

**MonitaQC** is an industrial computer vision platform for automated quality control and product identification. Forked from PartQC Box Counter, this project is the foundation for a unified quality control system.

## Overview

MonitaQC combines advanced image processing, object detection, and hardware integration to provide real-time quality inspection capabilities for manufacturing and fulfillment operations.

### Key Features

- **Multi-Camera Vision System**: Supports up to 4 cameras with auto-detection
- **AI-Powered Detection**: YOLOv5-based object detection with custom model support
- **DataMatrix Recognition**: Advanced barcode decoding with multi-stage preprocessing
- **OCR Capabilities**: Text recognition using EasyOCR
- **Object Nesting Detection**: Hierarchical parent-child object relationships
- **Real-Time Streaming**: Live video feeds with annotated results
- **Hardware Integration**: Serial communication, barcode scanners, GPIO control
- **Shipment Tracking**: Django-based fulfillment management system
- **Ejector Control**: Automated defect rejection system
- **Multi-Language Support**: English and Persian/Farsi

## Architecture

MonitaQC uses a microservices architecture:

```
MonitaQC/
├── main.py                    # Core processing engine
├── yolo_inference/            # AI object detection service
├── ocr/                       # Optical character recognition
├── stream/                    # Real-time video streaming
├── scanner/                   # Barcode scanner integration
├── speaker/                   # Audio feedback system
├── shipment_fulfillment/      # Django backend for order management
├── cleanup/                   # Automated disk space management
└── docker-compose.yml         # Service orchestration
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for YOLO inference)
- NVIDIA Container Toolkit (for GPU access in Docker)
- Camera devices (USB or compatible)
- Serial device (Arduino/PLC) at `/dev/ttyUSB0`
- Optional: Barcode scanner

### 1. Load AI Weights

Train your custom model using [ai-trainer.monitait.com](https://ai-trainer.monitait.com), then:

1. Download the `best.pt` weight file
2. Place it in `yolo_inference/best.pt`

For multi-camera setups, you can use different models:
- `yolo_inference/best.pt` - Default model
- `yolo_inference/cam2_best.pt` - Camera 2 specific model

### 2. Prepare Query Data

Configure product identification mappings in `.env.prepared_query_data`:

```json
[
    {
        "dm": "6263957101037",
        "chars": [
            ["box"],
            ["logo_en", "logo_fa"],
            ["specific_object_name_in_yolo"]
        ]
    }
]
```

**How it works**: If the system detects objects matching the specified annotations (e.g., "box", "logo_en", "specific_object_name_in_yolo"), it will identify the product as DataMatrix `6263957101037`.

### 3. Configure Scanner (Optional)

If using a barcode scanner:

1. Find the scanner device: `ls /dev/input/event*`
2. Plug/unplug scanner to identify the new device
3. Update `scanner/.env.scanner` with the device path

### 4. Pre-load Docker Image (Optional)

For faster build times, load pre-built YOLO image:

```bash
sudo docker load -i yolo.tar
```

### 5. Build and Run

```bash
sudo docker compose up -d
```

Access the web interface at `http://localhost:8000`

## Configuration

### Environment Variables

Key configuration options in `docker-compose.yml`:

```yaml
# Ejector Configuration
EJECTOR_OFFSET=0              # Encoder offset from camera to ejector
EJECTOR_DURATION=0.4          # Ejection duration in seconds

# Capture Configuration
CAPTURE_MODE=single           # single or continuous
TIME_BETWEEN_TWO_PACKAGE=0.305

# Camera Paths
CAM_1_PATH=/dev/video0
CAM_2_PATH=/dev/video2
CAM_3_PATH=/dev/video4
CAM_4_PATH=/dev/video6

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# YOLO Configuration
YOLO_INFERENCE_URL=http://yolo_inference:4442/v1/object-detection/yolov5s/detect/
YOLO_CONF_THRESHOLD=0.3
YOLO_IOU_THRESHOLD=0.4

# Serial/Watcher Configuration
WATCHER_USB=/dev/ttyUSB0
BAUD_RATE=57600
SERIAL_MODE=legacy            # legacy or new

# Web Server
WEB_SERVER_PORT=5050
WEB_SERVER_HOST=0.0.0.0
```

### Hardware Commands

Arduino/PLC command mapping:

```yaml
WATCHER_CMD_U_ON_B_OFF=1      # Uplight on, backlight off
WATCHER_CMD_B_ON_U_OFF=2      # Backlight on, uplight off
WATCHER_CMD_RST_ENCODER=3     # Reset encoder
WATCHER_CMD_U_SET_PWM=4       # Set uplight PWM (0-255)
WATCHER_CMD_B_SET_PWM=5       # Set backlight PWM (0-255)
WATCHER_CMD_WARNING_ON=6      # Warning indicator on
WATCHER_CMD_WARNING_OFF=7     # Warning indicator off
WATCHER_CMD_B_OFF_U_OFF=8     # Both lights off
WATCHER_CMD_U_ON_B_ON=9       # Both lights on
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **counter** | 5050 | Main processing engine and status page |
| **web** | 8000 | Django shipment fulfillment interface |
| **stream** | 5000 | Real-time video streaming |
| **pigallery2** | 80 | Image gallery browser |
| **redis** | 6379 | Message queue and cache |
| **db** | 5432 | PostgreSQL database |
| **yolo_inference** | 4442 | AI inference service (2 replicas) |

## Data Storage

- **Raw Images**: `/mnt/SSD-RESERVE/raw_images/` (configurable volume)
- **Processed Images**: `./volumes/images/`
- **Database**: PostgreSQL persistent volume
- **Weights**: `./volumes/weights/`
- **Logs**: `./volumes/logs/`

Automatic cleanup service monitors disk usage and removes old images when reaching 90% capacity.

## API Endpoints

### Counter Service (Port 5050)
- `GET /` - Status monitoring web interface
- `GET /health` - Health check
- WebSocket `/ws` - Real-time updates

### Shipment Fulfillment (Port 8000)
- Django admin interface at `/admin`
- REST API for shipment management
- WebSocket at port 6789

### Stream Service (Port 5000)
- `GET /video_feed/{stream_id}` - Live video stream
- `GET /stream/` - Stream status

## Development

### Project Structure

```
main.py (3371 lines)
├── ArduinoSocket        # Hardware interface and capture orchestration
├── StateManager         # Multi-phase capture state machine
├── CameraBuffer         # Video stream buffering
└── Processing Pipeline  # YOLO → Nesting → DataMatrix → Matching
```

### Key Technologies

- **Python 3.10.4**
- **FastAPI** - API framework
- **Django 5.1.2** - Admin backend
- **OpenCV 4.7.0** - Image processing
- **YOLOv5** (PyTorch) - Object detection
- **EasyOCR** - Text recognition
- **pylibdmtx** - DataMatrix decoding
- **Redis 4.3.4** - Message broker
- **PostgreSQL 13** - Database
- **Celery** - Task queue

## Roadmap

MonitaQC is evolving into a unified quality control platform. Planned features:

- [ ] Merge with fabric inspection capabilities (from FabriQC)
- [ ] Merge with signal counting capabilities (from PartQC Signal Counter)
- [ ] Unified admin interface for all QC modes
- [ ] Multi-application mode support
- [ ] Enhanced API with OpenAPI documentation
- [ ] Advanced analytics and reporting
- [ ] Cloud synchronization improvements

Submit feature requests and ideas to the [project issues backlog](https://gitlab.virasad.ir/monitait/monitaqc/-/issues).

## Support

For issues or questions:
- Email: [contact@virasad.ir](mailto:contact@virasad.ir)
- Contact the system administrator

## Acknowledgments

Special thanks to the teams at [Zarrin Roya](https://www.zarrinroya.com/en) who contributed to this project:
- Digital Transformation Department
- IT/ICT Department
- Logistics Department
- Mr. Ebrahimi, Mr. Ehsani, Ms. Nobahari, Mr. Pourmand, Mr. Solimani, Mr. Salehi, Ms. Samaneh

## Project Status

**Active Development** - Currently deployed at Zarrin Roya fulfillment center

## License

Proprietary - VirasAd / Monitait

---

**Note**: This project is forked from PartQC Box Counter and serves as the foundation for the unified MonitaQC platform.
