# MonitaQC Vision Engine

This directory contains the computer vision processing engine for MonitaQC - the core service that handles image capture, AI detection, and quality analysis.

## Files

- **main.py** (3371 lines) - Core vision processing engine
  - `ArduinoSocket` - Hardware interface and capture orchestration
  - `StateManager` - Multi-phase capture state machine
  - `CameraBuffer` - Video stream buffering
  - Processing Pipeline: YOLO → Nesting → DataMatrix → Matching

- **Dockerfile** - Container build configuration
- **requirements.txt** - Python dependencies

## Functionality

The vision engine handles:
- **Multi-camera capture** (unlimited cameras with dynamic auto-detection)
- **Serial communication** with Arduino/PLC
- **YOLO-based object detection** via inference service
- **DataMatrix barcode decoding** with multi-stage preprocessing
- **OCR text recognition** integration
- **Object nesting and classification** (parent-child relationships)
- **Encoder-based triggering** for conveyor systems
- **Ejector control** for defect rejection
- **Real-time status web interface** with live camera stream (port 5050)
- **Image histogram analysis** and feature extraction

## Environment Variables

See main [docker-compose.yml](../docker-compose.yml) for configuration options.

Key variables:
- `CAM_1_PATH` to `CAM_4_PATH` - Camera device paths
- `WATCHER_USB` - Serial device for Arduino/PLC
- `YOLO_INFERENCE_URL` - AI inference service endpoint
- `REDIS_HOST` - Redis cache and message queue
- `EJECTOR_OFFSET` - Encoder position offset for rejection
- `CAPTURE_MODE` - single or continuous capture

## Dependencies

```
opencv-python==4.7.0.72
numpy==1.23.2
fastapi==0.81.0
uvicorn==0.18.3
pylibdmtx              # DataMatrix decoding
pyserial==3.5          # Arduino communication
redis==4.3.4           # Message queue
requests==2.28.1       # HTTP client
Pillow                 # Image processing
arabic-reshaper        # Multi-language support
python-bidi            # Bi-directional text
```

## Build

```bash
docker build -t monitaqc-vision .
```

## Run Standalone

```bash
docker run -d \
  --name monitaqc_vision \
  --privileged \
  --device /dev:/dev \
  -p 5050:5050 \
  -v /mnt/SSD-RESERVE/raw_images:/code/raw_images \
  -e REDIS_HOST=redis \
  -e YOLO_INFERENCE_URL=http://yolo_inference:4442/v1/object-detection/yolov5s/detect/ \
  monitaqc-vision
```

For full configuration, use the main docker-compose.yml.

## Architecture

The vision engine is the heart of MonitaQC, coordinating:
1. Hardware (cameras, sensors, Arduino)
2. AI services (YOLO, OCR)
3. Data processing (nesting, matching, classification)
4. Output control (ejector, displays, Redis)

It operates as a continuous processing loop, capturing images based on encoder triggers and processing them through the AI pipeline.
