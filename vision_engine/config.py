"""Global configuration and data file management for MonitaQC Vision Engine.

All environment-based constants, mutable runtime state, and data file I/O live here.
Imported via `from config import *` in main.py as a temporary bridge during refactoring.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from environment variables
# =============================================================================

# Ejector configuration
EJECTOR_ENABLED = os.environ.get("EJECTOR_ENABLED", "true").lower() == "true"
EJECTOR_OFFSET = int(os.environ.get("EJECTOR_OFFSET", 0))
EJECTOR_DURATION = float(os.environ.get("EJECTOR_DURATION", 0.4))
EJECTOR_POLL_INTERVAL = float(os.environ.get("EJECTOR_POLL_INTERVAL", 0.03))

# Capture configuration
CAPTURE_MODE = os.environ.get("CAPTURE_MODE", "single")
TIME_BETWEEN_TWO_PACKAGE = float(os.environ.get("TIME_BETWEEN_TWO_PACKAGE", 0.305))

# Image processing configuration
REMOVE_RAW_IMAGE_WHEN_DM_DECODED = os.environ.get("REMOVE_RAW_IMAGE_WHEN_DM_DECODED", "false").lower() == "true"
PARENT_OBJECT_LIST = [x.strip() for x in os.environ.get("PARENT_OBJECT_LIST", "_root,box,pack,DP-105,jean,knit").split(",") if x.strip()]

# DataMatrix configuration
DM_CHARS_SIZES = [int(x) for x in os.environ.get("DM_CHARS_SIZES", "13,19,26").split(",")]
DM_CONFIDENCE_THRESHOLD = float(os.environ.get("DM_CONFIDENCE_THRESHOLD", 0.8))
DM_OVERLAP_THRESHOLD = float(os.environ.get("DM_OVERLAP_THRESHOLD", 0.2))

# Object count checking configuration
CHECK_CLASS_COUNTS_ENABLED = os.environ.get("CHECK_CLASS_COUNTS_ENABLED", "true").lower() == "true"
CHECK_CLASS_COUNTS_CLASSES = [x.strip() for x in os.environ.get("CHECK_CLASS_COUNTS_CLASSES", "socket,nozzle").split(",") if x.strip()]
CHECK_CLASS_COUNTS_CONFIDENCE = float(os.environ.get("CHECK_CLASS_COUNTS_CONFIDENCE", "0.5"))

# Light control configuration
LIGHT_STATUS_CHECK_ENABLED = os.environ.get("LIGHT_STATUS_CHECK_ENABLED", "false").lower() == "true"

# Histogram feature configuration
HISTOGRAM_ENABLED = os.environ.get("HISTOGRAM_ENABLED", "true").lower() == "true"
HISTOGRAM_SAVE_IMAGE = os.environ.get("HISTOGRAM_SAVE_IMAGE", "true").lower() == "true"

# Store annotation to PostgreSQL configuration
STORE_ANNOTATION_ENABLED = os.environ.get("STORE_ANNOTATION_ENABLED", "false").lower() == "true"
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "timescaledb")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "monitaqc")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "monitaqc")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "monitaqc2024")

# Data file configuration
DATA_FILE = os.environ.get("DATA_FILE", ".env.prepared_query_data")

# Parent object enforcement configuration
ENFORCE_PARENT_OBJECT = os.environ.get("ENFORCE_PARENT_OBJECT", "true").lower() == "true"

# Gradio/YOLO API configuration (defaults - will be overridden by DATA_FILE config)
YOLO_INFERENCE_URL = os.environ.get("YOLO_INFERENCE_URL", "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/")
GRADIO_MODEL = "Data Matrix"
GRADIO_CONFIDENCE_THRESHOLD = 0.25

# Global cache for latest detections
latest_detections = []
latest_detections_timestamp = 0.0

# Timeline buffer constants
TIMELINE_THUMBNAIL_WIDTH = 120
TIMELINE_THUMBNAIL_HEIGHT = 90
timeline_frame_counter = 0

# Redis keys for timeline storage
TIMELINE_REDIS_PREFIX = "timeline:"

# Detection events queue for audio notifications
DETECTION_EVENTS_REDIS_KEY = "detection_events"
DETECTION_EVENTS_MAX_SIZE = 100

# Inference performance tracking
inference_times = []
frame_intervals = []
last_inference_timestamp = 0.0
max_inference_samples = 10

# Camera capture FPS tracking
capture_timestamps = []
max_capture_samples = 10

# Lowercase aliases (used by ArduinoSocket)
capture_mode = CAPTURE_MODE
time_between_two_package = TIME_BETWEEN_TWO_PACKAGE
remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
parent_object_list = PARENT_OBJECT_LIST

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# Serial/Watcher configuration
WATCHER_USB = os.environ.get("WATCHER_USB", "/dev/ttyUSB0")
SERIAL_BAUDRATE = int(os.environ.get("BAUD_RATE", 57600))
SERIAL_MODE = os.environ.get("SERIAL_MODE", "new")

# Web server configuration
WEB_SERVER_PORT = int(os.environ.get("WEB_SERVER_PORT", 5050))
WEB_SERVER_HOST = os.environ.get("WEB_SERVER_HOST", "0.0.0.0")

# Command mapping for user-friendly names
USER_COMMANDS = {
    # Light control commands
    "U_ON_B_OFF": os.environ.get("WATCHER_CMD_U_ON_B_OFF", "1"),
    "B_ON_U_OFF": os.environ.get("WATCHER_CMD_B_ON_U_OFF", "2"),
    "B_OFF_U_OFF": os.environ.get("WATCHER_CMD_B_OFF_U_OFF", "8"),
    "U_ON_B_ON": os.environ.get("WATCHER_CMD_U_ON_B_ON", "9"),
    "WARNING_ON": os.environ.get("WATCHER_CMD_WARNING_ON", "6"),
    "WARNING_OFF": os.environ.get("WATCHER_CMD_WARNING_OFF", "7"),
    "RESET_ENCODER": os.environ.get("WATCHER_CMD_RST_ENCODER", "3"),
    "U_SET_PWM": os.environ.get("WATCHER_CMD_U_SET_PWM", "4"),
    "B_SET_PWM": os.environ.get("WATCHER_CMD_B_SET_PWM", "5"),
    # Counter adjustment commands
    "OK_ADJUST": os.environ.get("WATCHER_CMD_OK_ADJUST", "a"),
    "NG_ADJUST": os.environ.get("WATCHER_CMD_NG_ADJUST", "b"),
    "SET_DOWNTIME": os.environ.get("WATCHER_CMD_SET_DOWNTIME", "d"),
    # System configuration commands
    "SET_BAUD_RATE": os.environ.get("WATCHER_CMD_SET_BAUD_RATE", "s"),
    "SET_EXTERNAL_RESET": os.environ.get("WATCHER_CMD_SET_EXTERNAL_RESET", "e"),
    "SET_VERBOSE": os.environ.get("WATCHER_CMD_SET_VERBOSE", "v"),
    "SET_LEGACY": os.environ.get("WATCHER_CMD_SET_LEGACY", "l"),
    # OK configuration commands
    "OK_OFFSET_DELAY": os.environ.get("WATCHER_CMD_OK_OFFSET_DELAY", "o,od"),
    "OK_DURATION_PULSES": os.environ.get("WATCHER_CMD_OK_DURATION_PULSES", "o,dp"),
    "OK_DURATION_PERCENT": os.environ.get("WATCHER_CMD_OK_DURATION_PERCENT", "o,dl"),
    "OK_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_OK_ENCODER_FACTOR", "o,ef"),
    # NG configuration commands
    "NG_OFFSET_DELAY": os.environ.get("WATCHER_CMD_NG_OFFSET_DELAY", "n,od"),
    "NG_DURATION_PULSES": os.environ.get("WATCHER_CMD_NG_DURATION_PULSES", "n,dp"),
    "NG_DURATION_PERCENT": os.environ.get("WATCHER_CMD_NG_DURATION_PERCENT", "n,dl"),
    "NG_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_NG_ENCODER_FACTOR", "n,ef"),
}

# Commands that require a value parameter
COMMANDS_WITH_VALUE = [
    "U_SET_PWM", "B_SET_PWM",
    "OK_ADJUST", "NG_ADJUST", "SET_DOWNTIME",
    "SET_BAUD_RATE", "SET_EXTERNAL_RESET", "SET_VERBOSE", "SET_LEGACY",
    "OK_OFFSET_DELAY", "OK_DURATION_PULSES", "OK_DURATION_PERCENT", "OK_ENCODER_FACTOR",
    "NG_OFFSET_DELAY", "NG_DURATION_PULSES", "NG_DURATION_PERCENT", "NG_ENCODER_FACTOR",
]


# =============================================================================
# Data file management
# =============================================================================

def load_data_file():
    """Load entire data file. Returns dict with 'data' key for list content, or the dict itself."""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    return {"prepared_query_data": content}
                return content
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
    return {}

def save_data_file(data):
    """Save entire data file with backup."""
    try:
        if os.path.exists(DATA_FILE):
            backup_file = f"{DATA_FILE}.backup"
            with open(DATA_FILE, 'r') as f:
                backup_data = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_data)

        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data file saved to {DATA_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving data file: {e}")
        return False

def load_service_config():
    """Load service configuration from data file (under 'service_config' key)."""
    data = load_data_file()
    return data.get("service_config", {})

def save_service_config(config):
    """Save service configuration to data file (under 'service_config' key)."""
    data = load_data_file()
    config["saved_at"] = datetime.now().isoformat()
    data["service_config"] = config
    return save_data_file(data)

# Aliases for backwards compatibility
load_camera_config = load_service_config
save_camera_config = save_service_config
