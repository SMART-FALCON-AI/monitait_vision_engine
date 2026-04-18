# MonitaQC Vision Engine - VS Code Claude Extension Context

## Application Overview
MonitaQC Vision Engine is an industrial quality control and vision inspection system for fabric/textile manufacturing. It provides real-time YOLO-based defect detection, camera management, hardware integration (serial/USB devices), and production line monitoring with AI-powered analytics.

---

## System Architecture

### **Technology Stack**
- **Backend**: Python 3.10, FastAPI, Uvicorn
- **Computer Vision**: OpenCV, YOLOv5, Gradio Client
- **Database**:
  - Redis (cache, real-time data)
  - TimescaleDB/PostgreSQL (time-series metrics)
  - Batch Tracking DB (PostgreSQL on port 5432)
- **Frontend**: HTML5, Vanilla JavaScript, Server-Sent Events (SSE)
- **Visualization**: Grafana dashboards
- **Containerization**: Docker Compose

### **Service Ports**
- Vision Engine: `5050` (main web interface)
- Redis: `6379`
- YOLO Inference: `4442` (internal)
- TimescaleDB: `5432`
- Grafana: `3000`
- Pigallery2: `5000` (image gallery)

---

## Key Components

### **1. Vision Engine (`vision_engine/main.py`)**

#### **Core Classes**
```python
class State:
    """Camera capture state with multi-phase support"""
    name: str
    phases: List[StatePhase]  # Light combinations (U_ON_B_OFF, etc.)
    fps: int
    target_detections: int

class StatePhase:
    """Single phase in capture sequence"""
    light_mode: str  # U_ON_B_OFF, U_OFF_B_ON, U_ON_B_ON, B_OFF_U_OFF
    duration_ms: int

class StateManager:
    """Manages camera capture states and transitions"""
    states: Dict[str, State]
    current_state: State
```

#### **Key Global Variables**
```python
# Hardware Configuration
EJECTOR_ENABLED = True
EJECTOR_OFFSET = 0  # Encoder counts from camera to ejector
EJECTOR_DURATION = 0.4  # Seconds
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 57600

# Vision Configuration
YOLO_INFERENCE_URL = "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/"
GRADIO_MODEL = "Data Matrix"
GRADIO_CONFIDENCE_THRESHOLD = 0.3

# Redis Configuration
REDIS_HOST = "redis"
REDIS_PORT = 6379

# Database Configuration
POSTGRES_HOST = "timescaledb"
POSTGRES_PORT = 5432
POSTGRES_DB = "monitaqc"
POSTGRES_USER = "monitaqc"
POSTGRES_PASSWORD = "monitaqc2024"
```

### **2. API Endpoints**

#### **Status & Health**
```python
GET /health
# Returns: {"status": "healthy", "device": "camera-only", "redis": "connected", "yolo": "connected"}

GET /api/status
# Returns: Real-time encoder values, counters, movement status, ejector state

GET /api/inference/stats
# Returns: {"service_type": "YOLO (Local)", "avg_inference_time_ms": 392.9, "inference_fps": 1.42}

GET /api/cameras
# Returns: [{"id": 1, "path": "rtsp://...", "connected": false, "running": true}]
```

#### **Configuration Management**
```python
POST /api/config
# Body: {"shipment": "SHIP123", "ejector_offset": 100, "yolo_url": "..."}
# Updates runtime configuration

POST /api/save_service_config
# Saves current configuration to .env.prepared_query_data

POST /api/load_service_config
# Reloads configuration from file

GET /api/export_service_config
# Downloads configuration as JSON file

# POST /api/import_service_config  # TEMPORARILY DISABLED (waiting for python-multipart)
# Uploads and applies JSON configuration
```

#### **Camera Management**
```python
GET /api/camera/{camera_id}/snapshot?t={timestamp}
# Returns: JPEG snapshot from camera

POST /api/camera/{camera_id}/restart
# Restarts camera stream

POST /api/camera/{camera_id}/config
# Body: {"fps": 30, "width": 1920, "height": 1080}
```

#### **State Management**
```python
GET /api/states
# Returns: List of available capture states

POST /api/states
# Body: {"name": "max", "fps": 120, "phases": [...]}
# Creates new capture state

POST /api/states/{state_name}/activate
# Activates specified state

POST /api/states/save
# Saves states to configuration

POST /api/states/load
# Loads states from configuration
```

#### **AI Assistant (NEW - Frontend Ready)**
```python
POST /api/ai_config
# Body: {"model": "claude", "api_key": "sk-ant-..."}
# Saves AI model configuration

POST /api/ai_query
# Body: {"query": "What's the defect rate for the last hour?"}
# Queries AI with production data from TimescaleDB
```

### **3. Frontend Structure (`vision_engine/static/status.html`)**

#### **Tab Navigation**
1. **📊 Dashboard** - Real-time metrics, camera streams
2. **📷 Cameras** - Camera discovery, configuration
3. **🎛️ Control Panel** - U/B/Warning LEDs, PWM control
4. **⚙️ Ejector/Counter** - Serial communication, counters
5. **🔧 Hardware** - Serial port, watcher configuration
6. **🏗️ Infrastructure** - Redis, YOLO, PostgreSQL settings
7. **🖼️ Gallery** - Embedded Pigallery2 (iframe to localhost:5000)
8. **📈 Grafana** - Embedded Grafana dashboards (iframe to localhost:3000)
9. **🤖 AI Assistant** - Multi-model AI chatbot (Claude, ChatGPT, Gemini, Local)
10. **⚡ Advanced** - Image processing, DataMatrix, feature toggles

#### **Key JavaScript Functions**
```javascript
// Configuration Management
saveAllServiceConfig()  // Saves both service config and data file
loadServiceConfig()     // Reloads from file and refreshes page
exportServiceConfig()   // Downloads config as JSON
// importServiceConfig() // DISABLED - waiting for rebuild

// AI Assistant
sendAIMessage()         // Sends user query to AI
saveAIConfig()          // Saves AI model and API key
updateAIModelFields()   // Updates UI based on selected model

// Real-time Updates
fetchStatus()           // Polls /api/status every 0.5s
updateInferenceStats()  // Polls /api/inference/stats

// Camera Controls
sendCommand(cmd, value) // Sends hardware commands (U_ON, B_ON, etc.)
```

---

## Database Schemas

### **TimescaleDB (Planned - Not Yet Implemented)**
```sql
-- Inference results table
CREATE TABLE inference_results (
    time TIMESTAMPTZ NOT NULL,
    camera_id INT,
    shipment TEXT,
    detections JSONB,
    inference_time FLOAT,
    defect_type TEXT,
    confidence FLOAT
);
SELECT create_hypertable('inference_results', 'time');

-- Production metrics table
CREATE TABLE production_metrics (
    time TIMESTAMPTZ NOT NULL,
    encoder_value INT,
    speed_ppm FLOAT,
    ok_count INT,
    ng_count INT,
    downtime_seconds INT
);
SELECT create_hypertable('production_metrics', 'time');
```

### **Redis Keys**
```
shipment: "no_shipment"  # Current shipment ID
encoder: 0               # Current encoder value
ok_counter: 0            # OK detection count
ng_counter: 0            # NG detection count
```

---

## Configuration Files

### **docker-compose.yml**
```yaml
services:
  monitait_vision_engine:
    build: ./vision_engine
    ports: ["5050:5050"]
    devices: ["/dev:/dev"]
    privileged: true

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  yolo_inference:
    build: ./yolo_inference
    expose: [4442]

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: monitaqc
      POSTGRES_PASSWORD: monitaqc2024
      POSTGRES_DB: monitaqc

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
```

### **requirements.txt**
```
fastapi==0.81.0
fastapi-utils==0.2.1
python-multipart==0.0.5  # For file uploads (import endpoint)
opencv-python==4.7.0.72
redis==4.3.4
requests==2.28.1
uvicorn==0.18.3
numpy==1.23.2
pyserial==3.5
pylibdmtx  # DataMatrix decoding
gradio_client
```

---

## Current Implementation Status

### ✅ **Working Features**
- Real-time vision processing with YOLO
- Camera stream management (IP cameras via RTSP)
- Hardware integration (serial port, encoder, ejector)
- Redis caching and state management
- Multi-state camera capture (default, test, max)
- Configuration save/load/export
- Shipment ID tracking and editing
- Live Dashboard with metrics
- Grafana integration (embedded iframe)
- Gallery integration (Pigallery2)
- AI Assistant UI (Claude, ChatGPT, Gemini, Local model support)

### ⚠️ **Pending Implementation**
1. **AI Assistant Backend**:
   - `/api/ai_config` endpoint (save AI model config to Redis)
   - `/api/ai_query` endpoint (query TimescaleDB and send to AI model)
   - API key encryption/storage

2. **TimescaleDB Data Logging**:
   - Create database schemas
   - Log inference results to `inference_results` table
   - Log production metrics to `production_metrics` table

3. **Import Configuration**:
   - Currently disabled (waiting for `python-multipart` package installation)
   - Endpoint `/api/import_service_config` commented out in main.py:2819

4. **Grafana Dashboards**:
   - Create default dashboards for production metrics
   - Configure TimescaleDB data source
   - Add panels for defect rates, speed, downtime

---

## File Structure

```
MonitaQC/
├── docker-compose.yml
├── vision_engine/
│   ├── main.py                 # FastAPI application (2843 lines)
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container build
│   ├── static/
│   │   └── status.html         # Web interface (2600+ lines)
│   ├── .env.prepared_query_data  # Persistent configuration
│   └── raw_images/             # Captured images (mounted volume)
├── yolo_inference/
│   ├── main.py                 # YOLO inference service
│   └── weights/                # YOLO model weights
├── cleanup/
│   └── cleanup.py              # Automatic disk cleanup service
└── README.md
```

---

## Common Development Tasks

### **Adding a New API Endpoint**
```python
# In main.py, before the command endpoint section
@app.post("/api/my_new_endpoint")
async def my_new_endpoint(data: Dict[str, Any]):
    """Description of what this endpoint does."""
    try:
        # Implementation
        return JSONResponse(content={"success": True})
    except Exception as e:
        logger.error(f"Error in my_new_endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
```

### **Adding Database Logging**
```python
import psycopg2

# Connect to TimescaleDB
conn = psycopg2.connect(
    host="timescaledb",
    port=5432,
    database="monitaqc",
    user="monitaqc",
    password="monitaqc2024"
)

# Insert inference result
cur = conn.cursor()
cur.execute("""
    INSERT INTO inference_results
    (time, camera_id, shipment, detections, inference_time, defect_type, confidence)
    VALUES (NOW(), %s, %s, %s, %s, %s, %s)
""", (camera_id, shipment, json.dumps(detections), inference_time, defect_type, confidence))
conn.commit()
```

### **Adding a New Tab to UI**
1. Add tab button in status.html navigation (line ~790)
2. Add tab content div with `id="tab-{name}"` (around line 1600)
3. Tab will auto-work with existing `switchTab()` function

---

## Known Issues & Workarounds

### **Issue 1: Import Config Button Disabled**
**Cause**: `python-multipart` package not installed in container
**Status**: Container rebuild in progress
**Workaround**: Use Export/Manual Edit/Load instead
**Fix**: After rebuild completes, uncomment lines 2819-2839 in main.py

### **Issue 2: Camera Shows "Not Connected"**
**Cause**: IP camera at 192.168.0.108 not reachable
**Status**: Expected (camera-only mode)
**Workaround**: Configure correct camera IP in Cameras tab

### **Issue 3: AI Assistant Shows "No response"**
**Cause**: Backend endpoints `/api/ai_config` and `/api/ai_query` not implemented
**Status**: Frontend complete, backend pending
**Fix**: Implement AI query endpoints (see Pending Implementation section)

---

## Environment Variables

### **Vision Engine**
```bash
WEB_SERVER_PORT=5050
WEB_SERVER_HOST=0.0.0.0

# All other settings configured via web interface at:
# http://localhost:5050/status
# Saved to: .env.prepared_query_data
```

### **TimescaleDB**
```bash
POSTGRES_USER=monitaqc
POSTGRES_PASSWORD=monitaqc2024
POSTGRES_DB=monitaqc
```

### **Grafana**
```bash
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Testing & Debugging

### **Check Service Health**
```bash
# Vision Engine
curl http://localhost:5050/health

# Redis
docker exec monitait_redis redis-cli ping

# TimescaleDB
docker exec monitait_timescaledb pg_isready -U monitaqc

# View Logs
docker logs monitait_vision_engine --tail 100
docker logs monitait_yolo --tail 50
```

### **Test API Endpoints**
```bash
# Get current status
curl http://localhost:5050/api/status | jq

# Update shipment ID
curl -X POST http://localhost:5050/api/config \
  -H "Content-Type: application/json" \
  -d '{"shipment": "BATCH001"}'

# Export configuration
curl http://localhost:5050/api/export_service_config -o config.json
```

---

## Next Development Priorities

1. **Implement AI Assistant Backend**:
   - Create `/api/ai_config` endpoint
   - Create `/api/ai_query` endpoint with TimescaleDB integration
   - Add support for Claude, OpenAI, Gemini APIs

2. **Enable TimescaleDB Logging**:
   - Create database schemas
   - Add logging calls in inference pipeline
   - Add logging calls in status update loop

3. **Create Grafana Dashboards**:
   - Production speed over time
   - Defect rate trends
   - OK/NG counters
   - Downtime analysis

4. **Complete Import Feature**:
   - Rebuild container with `python-multipart`
   - Uncomment import endpoint
   - Test file upload functionality

---

## Useful Commands

```bash
# Restart all services
cd C:\projects\MonitaQC
docker compose restart

# Rebuild vision engine
docker compose up -d --build monitait_vision_engine

# View real-time logs
docker compose logs -f monitait_vision_engine

# Connect to TimescaleDB
docker exec -it monitait_timescaledb psql -U monitaqc -d monitaqc

# Access Redis CLI
docker exec -it monitait_redis redis-cli

# Check container status
docker compose ps
```

---

## Contact & Documentation

- **Main Interface**: http://localhost:5050/status
- **Grafana**: http://localhost:3000 (admin/admin)
- **Gallery**: http://localhost:5000
- **Project Location**: `C:\projects\MonitaQC`

---

**Last Updated**: 2025-12-31
**Version**: 1.0.0
**Status**: Production-ready with AI features in development
