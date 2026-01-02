import cv2
import time
import json
import requests
from PIL import Image
import threading
import os
import numpy as np
import serial
from collections import Counter
from datetime import datetime
from redis import Redis
from pylibdmtx import pylibdmtx
import math
import psycopg2
from psycopg2.extras import Json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import logging
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Set
import uvicorn
from enum import Enum
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from environment variables
# =============================================================================

# Ejector configuration
EJECTOR_ENABLED = os.environ.get("EJECTOR_ENABLED", "true").lower() == "true"  # Master switch for ejector logic
EJECTOR_OFFSET = int(os.environ.get("EJECTOR_OFFSET", 0))  # Encoder counts from camera to ejector
EJECTOR_DURATION = float(os.environ.get("EJECTOR_DURATION", 0.4))  # Seconds to run ejector motor
EJECTOR_POLL_INTERVAL = float(os.environ.get("EJECTOR_POLL_INTERVAL", 0.03))  # Seconds between ejector checks

# Capture configuration
CAPTURE_MODE = os.environ.get("CAPTURE_MODE", "single")  # "single" or "multiple"
TIME_BETWEEN_TWO_PACKAGE = float(os.environ.get("TIME_BETWEEN_TWO_PACKAGE", 0.305))  # Minimum seconds between captures

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
# If True, check serial status before sending light command (closed-loop)
# If False, only track phase changes locally without checking serial status (open-loop, faster)
LIGHT_STATUS_CHECK_ENABLED = os.environ.get("LIGHT_STATUS_CHECK_ENABLED", "false").lower() == "true"

# Histogram feature configuration
HISTOGRAM_ENABLED = os.environ.get("HISTOGRAM_ENABLED", "true").lower() == "true"
HISTOGRAM_SAVE_IMAGE = os.environ.get("HISTOGRAM_SAVE_IMAGE", "true").lower() == "true"

# Store annotation to PostgreSQL configuration
STORE_ANNOTATION_ENABLED = os.environ.get("STORE_ANNOTATION_ENABLED", "false").lower() == "true"
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "db-batch")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "batch_tracking")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

# Data file configuration
DATA_FILE = os.environ.get("DATA_FILE", ".env.prepared_query_data")

# Parent object enforcement configuration
ENFORCE_PARENT_OBJECT = os.environ.get("ENFORCE_PARENT_OBJECT", "true").lower() == "true"

# Gradio/YOLO API configuration (defaults - will be overridden by DATA_FILE config)
YOLO_INFERENCE_URL = "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/"
GRADIO_MODEL = "Data Matrix"
GRADIO_CONFIDENCE_THRESHOLD = 0.25

# Global cache for latest detections (used by video feeds, stored in Redis for cross-process sharing)
latest_detections = []
latest_detections_timestamp = 0.0

# Database connection pool
db_connection_pool = None

def get_db_connection():
    """Get PostgreSQL connection from pool."""
    global db_connection_pool
    if db_connection_pool is None:
        try:
            from psycopg2 import pool
            db_connection_pool = pool.SimpleConnectionPool(
                1, 10,
                host="timescaledb",
                port=5432,
                database="monitaqc",
                user="monitaqc",
                password="monitaqc2024"
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return None
    try:
        return db_connection_pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        return None

def release_db_connection(conn):
    """Release connection back to pool."""
    if db_connection_pool and conn:
        db_connection_pool.putconn(conn)

def write_inference_to_db(shipment, image_path, detections, inference_time_ms, model_used="yolov8"):
    """Write inference result to TimescaleDB."""
    if not STORE_ANNOTATION_ENABLED:
        return

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO inference_results
               (time, shipment, image_path, detections, detection_count, inference_time_ms, model_used)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s)""",
            (shipment, image_path, Json(detections), len(detections), inference_time_ms, model_used)
        )
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to write inference to database: {e}")
        conn.rollback()
    finally:
        release_db_connection(conn)

def write_production_metrics_to_db(encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds):
    """Write production metrics to TimescaleDB.

    Production metrics are always written regardless of annotation storage settings,
    as they represent different types of data (real-time system state vs. detection results).
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO production_metrics
               (time, encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s)""",
            (encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
        )
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to write production metrics to database: {e}")
        conn.rollback()
    finally:
        release_db_connection(conn)

# Inference performance tracking
inference_times = []  # Store last 10 inference times (processing time only)
frame_intervals = []  # Store last 10 frame-to-frame intervals
last_inference_timestamp = 0.0  # Timestamp of last inference
max_inference_samples = 10

# Camera capture FPS tracking (separate from inference FPS)
capture_timestamps = []  # Store last 10 capture timestamps
max_capture_samples = 10

# Camera paths configuration (optional - only used if devices exist)
CAM_1_PATH = os.environ.get("CAM_1_PATH", "")
CAM_2_PATH = os.environ.get("CAM_2_PATH", "")
CAM_3_PATH = os.environ.get("CAM_3_PATH", "")
CAM_4_PATH = os.environ.get("CAM_4_PATH", "")

# Camera discovery configuration
IP_CAMERA_USER = os.environ.get("IP_CAMERA_USER", "admin")  # Default username for auto-discovery
IP_CAMERA_PASS = os.environ.get("IP_CAMERA_PASS", "")  # Default password for auto-discovery
IP_CAMERA_SUBNET = os.environ.get("IP_CAMERA_SUBNET", "")  # Optional subnet override (e.g., "192.168.0")
IP_CAMERA_BRIGHTNESS = int(os.environ.get("IP_CAMERA_BRIGHTNESS", 128))  # 0-255
IP_CAMERA_CONTRAST = int(os.environ.get("IP_CAMERA_CONTRAST", 128))  # 0-255
IP_CAMERA_SATURATION = int(os.environ.get("IP_CAMERA_SATURATION", 128))  # 0-255

# Auto-detect camera devices
def detect_video_devices() -> List[str]:
    """Auto-detect available video devices from /dev/video* (even numbers only).

    Returns a sorted list of video device paths like ['/dev/video0', '/dev/video2', '/dev/video4'].
    Only even-numbered devices are returned as odd numbers are typically metadata devices.
    """
    import glob
    video_devices = []

    # Find all /dev/video* devices
    pattern = "/dev/video*"
    all_devices = glob.glob(pattern)

    for device in all_devices:
        try:
            # Extract the number from device path
            device_num = int(device.replace("/dev/video", ""))
            # Only use even-numbered devices (odd ones are typically metadata)
            if device_num % 2 == 0:
                video_devices.append((device_num, device))
        except ValueError:
            continue

    # Sort by device number and return just the paths
    video_devices.sort(key=lambda x: x[0])
    return [path for _, path in video_devices]


def scan_network_for_camera_devices(subnet: str = None) -> List[Dict[str, Any]]:
    """Quick scan to detect devices with camera ports open (no authentication needed).

    Returns list of potential cameras with IP, port, and protocol info.
    """
    import socket
    discovered_devices = []

    try:
        # Get subnet
        if subnet:
            scan_subnet = subnet
            logger.info(f"Using custom subnet: {scan_subnet}.0/24")
        elif IP_CAMERA_SUBNET:
            scan_subnet = IP_CAMERA_SUBNET
            logger.info(f"Using configured subnet: {scan_subnet}.0/24")
        else:
            # Auto-detect local subnet
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            ip_parts = local_ip.split('.')
            scan_subnet = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}"
            logger.info(f"Auto-detected subnet: {scan_subnet}.0/24")

        logger.info(f"Scanning subnet {scan_subnet}.0/24 for camera devices...")

        # Common camera ports and their protocols
        camera_ports = [
            (554, "RTSP", ["stream1", "Streaming/Channels/101", "cam/realmonitor?channel=1&subtype=0", "onvif1", "h264Preview_01_main", "videoMain"]),
            (8554, "RTSP", ["stream1", "live"]),
            (80, "HTTP", ["video.mjpg", "mjpg/video.mjpg", "video", "axis-media/media.amp"]),
            (8080, "HTTP", ["video.mjpg", "video"]),
        ]

        # Scan IP range
        for ip_suffix in range(1, 255):
            ip = f"{scan_subnet}.{ip_suffix}"

            for port, protocol, paths in camera_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)  # Fast scan
                    result = sock.connect_ex((ip, port))
                    sock.close()

                    if result == 0:
                        # Port is open - this is a potential camera
                        logger.info(f"📹 Found {protocol} service at {ip}:{port}")

                        # Add all potential paths for this camera
                        for path in paths:
                            if protocol == "RTSP":
                                port_str = f":{port}" if port != 554 else ""
                                url = f"rtsp://{ip}{port_str}/{path}"
                            else:  # HTTP
                                port_str = f":{port}" if port != 80 else ""
                                url = f"http://{ip}{port_str}/{path}"

                            discovered_devices.append({
                                "ip": ip,
                                "port": port,
                                "protocol": protocol,
                                "path": path,
                                "url": url
                            })

                        break  # Found open port, move to next IP

                except Exception:
                    continue

        logger.info(f"Quick scan complete. Found {len(set([d['ip'] for d in discovered_devices]))} potential cameras with {len(discovered_devices)} possible paths.")
        return discovered_devices

    except Exception as e:
        logger.error(f"Network scan failed: {e}")
        return []


def scan_network_for_cameras(timeout: float = 2.0) -> List[str]:
    """Scan local network for IP cameras using common ports and protocols.

    Returns list of discovered camera URLs.
    """
    import socket
    discovered_cameras = []

    # Get credentials for testing
    username = IP_CAMERA_USER
    password = IP_CAMERA_PASS
    auth_str = f"{username}:{password}@" if username and password else ""

    # Get local network subnet
    try:
        # Use custom subnet if specified, otherwise auto-detect
        if IP_CAMERA_SUBNET:
            subnet = IP_CAMERA_SUBNET
            logger.info(f"Using custom subnet: {subnet}.0/24")
        else:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Extract subnet (e.g., 192.168.1.0/24)
            ip_parts = local_ip.split('.')
            subnet = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}"

        logger.info(f"Scanning subnet {subnet}.0/24 for IP cameras...")
        if auth_str:
            logger.info(f"Using credentials: {username}:***")

        # Common camera ports and paths
        camera_configs = [
            # RTSP paths (port, paths)
            (554, ["stream1", "Streaming/Channels/101", "cam/realmonitor?channel=1&subtype=0",
                   "onvif1", "h264Preview_01_main", "videoMain"]),
            (8554, ["stream1", "live"]),
            # HTTP/MJPEG paths (port, paths)
            (80, ["video.mjpg", "mjpg/video.mjpg", "video", "axis-media/media.amp"]),
            (8080, ["video.mjpg", "video"]),
        ]

        # Scan IP range (typically cameras are .1-.254, scan all for thoroughness)
        for ip_suffix in range(1, 255):
            ip = f"{subnet}.{ip_suffix}"

            # Quick check if IP is reachable
            for port, paths in camera_configs:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)  # Fast scan
                    result = sock.connect_ex((ip, port))
                    sock.close()

                    if result == 0:
                        # Port is open - try camera paths
                        if port in [554, 8554]:
                            # RTSP camera - log discovery first
                            port_str = f":{port}" if port != 554 else ""
                            logger.info(f"📹 Found RTSP service at {ip}:{port}")

                            for path in paths:
                                # Try without auth first (for discovery)
                                camera_url_no_auth = f"rtsp://{ip}{port_str}/{path}"

                                # If credentials provided, use them
                                if auth_str:
                                    camera_url = f"rtsp://{auth_str}{ip}{port_str}/{path}"
                                else:
                                    camera_url = camera_url_no_auth

                                # Test if camera responds
                                if test_camera_stream(camera_url):
                                    discovered_cameras.append(camera_url)
                                    logger.info(f"✓ Verified RTSP camera: {camera_url.split('@')[-1]} (path: /{path})")
                                    break  # Found working path, move to next IP
                                else:
                                    # Log the attempted path for user reference
                                    if not auth_str:
                                        logger.info(f"  → Potential path: rtsp://{ip}{port_str}/{path} (may need credentials)")

                        elif port in [80, 8080]:
                            # HTTP/MJPEG camera - log discovery first
                            port_str = f":{port}" if port != 80 else ""
                            logger.info(f"📹 Found HTTP service at {ip}:{port}")

                            for path in paths:
                                camera_url = f"http://{ip}{port_str}/{path}"

                                # Test if camera responds
                                if test_camera_stream(camera_url):
                                    discovered_cameras.append(camera_url)
                                    logger.info(f"✓ Verified HTTP camera: {camera_url}")
                                    break  # Found working path, move to next IP
                                else:
                                    # Log the attempted path for user reference
                                    logger.info(f"  → Potential path: {camera_url} (may need credentials)")

                except Exception:
                    continue

    except Exception as e:
        logger.warning(f"Network scan failed: {e}")

    logger.info(f"IP camera scan complete. Found {len(discovered_cameras)} cameras.")
    return discovered_cameras


def test_camera_stream(url: str, timeout: float = 2.0) -> bool:
    """Test if a camera URL is accessible and returns valid video stream.

    Args:
        url: Camera URL (RTSP or HTTP)
        timeout: Connection timeout in seconds

    Returns:
        True if camera is accessible and returns video frames
    """
    try:
        # Try to open the camera stream
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000))

        # Try to read a frame
        ret, frame = cap.read()
        cap.release()

        # Check if we got a valid frame
        if ret and frame is not None and frame.size > 0:
            return True
        return False
    except Exception:
        return False


# Note: IP camera detection is now handled through the camera discovery API
# and saved to the unified "cameras" configuration structure


def get_all_cameras() -> List[str]:
    """Get all available USB cameras (auto-detected or from environment variables).

    Returns list of USB camera sources that exist as devices.
    Note: IP cameras are now loaded from the unified "cameras" configuration.

    Priority:
    1. If environment variables (CAM_X_PATH) are set, use those
    2. Otherwise, auto-detect USB cameras from /dev/video*
    """
    all_cameras = []

    # First check if any env vars are explicitly set
    usb_env_vars = [CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH]
    env_cameras_set = any(p for p in usb_env_vars if p)

    if env_cameras_set:
        # Use env vars if set - only add if device exists
        for cam_path in usb_env_vars:
            if cam_path and os.path.exists(cam_path):
                all_cameras.append(cam_path)
                logger.info(f"Found USB camera from env: {cam_path}")
    else:
        # Auto-detect USB cameras from /dev/video*
        detected = detect_video_devices()
        for cam_path in detected:
            if os.path.exists(cam_path):
                all_cameras.append(cam_path)
                logger.info(f"Auto-detected USB camera: {cam_path}")

    # Remove duplicates while preserving order
    seen = set()
    unique_cameras = []
    for cam in all_cameras:
        if cam not in seen:
            seen.add(cam)
            unique_cameras.append(cam)
        else:
            logger.info(f"Skipping duplicate camera: {cam}")

    return unique_cameras


# Detect cameras at module load time
DETECTED_CAMERAS = get_all_cameras()
logger.info(f"Auto-detected cameras: {DETECTED_CAMERAS}")


# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# YOLO inference configuration
YOLO_INFERENCE_URL = os.environ.get("YOLO_INFERENCE_URL", "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/")

# Serial/Watcher configuration
WATCHER_USB = os.environ.get("WATCHER_USB", "/dev/ttyUSB0")
SERIAL_BAUDRATE = int(os.environ.get("BAUD_RATE", 57600))
SERIAL_MODE = os.environ.get("SERIAL_MODE", "new")  # "new" (normal) or "legacy"

# Web server configuration
WEB_SERVER_PORT = int(os.environ.get("WEB_SERVER_PORT", 5050))
WEB_SERVER_HOST = os.environ.get("WEB_SERVER_HOST", "0.0.0.0")

# Command mapping for user-friendly names (configurable via environment variables)
USER_COMMANDS = {
    # Light control commands
    "U_ON_B_OFF": os.environ.get("WATCHER_CMD_U_ON_B_OFF", "1"),      # U on, B off
    "B_ON_U_OFF": os.environ.get("WATCHER_CMD_B_ON_U_OFF", "2"),      # B on, U off
    "B_OFF_U_OFF": os.environ.get("WATCHER_CMD_B_OFF_U_OFF", "8"),    # Both off
    "U_ON_B_ON": os.environ.get("WATCHER_CMD_U_ON_B_ON", "9"),        # Both on
    "WARNING_ON": os.environ.get("WATCHER_CMD_WARNING_ON", "6"),      # Warning on (ejector)
    "WARNING_OFF": os.environ.get("WATCHER_CMD_WARNING_OFF", "7"),    # Warning off
    "RESET_ENCODER": os.environ.get("WATCHER_CMD_RST_ENCODER", "3"),  # Reset encoder
    "U_SET_PWM": os.environ.get("WATCHER_CMD_U_SET_PWM", "4"),        # Set PWM for U
    "B_SET_PWM": os.environ.get("WATCHER_CMD_B_SET_PWM", "5"),        # Set PWM for B

    # Counter adjustment commands
    "OK_ADJUST": os.environ.get("WATCHER_CMD_OK_ADJUST", "a"),        # Adjust OK counter
    "NG_ADJUST": os.environ.get("WATCHER_CMD_NG_ADJUST", "b"),        # Adjust NG counter
    "SET_DOWNTIME": os.environ.get("WATCHER_CMD_SET_DOWNTIME", "d"),  # Set downtime threshold

    # System configuration commands
    "SET_BAUD_RATE": os.environ.get("WATCHER_CMD_SET_BAUD_RATE", "s"),      # Set baud rate
    "SET_EXTERNAL_RESET": os.environ.get("WATCHER_CMD_SET_EXTERNAL_RESET", "e"),  # Set encoder state (1=on, 0=off)
    "SET_VERBOSE": os.environ.get("WATCHER_CMD_SET_VERBOSE", "v"),          # Set verbose mode (1=on, 0=off)
    "SET_LEGACY": os.environ.get("WATCHER_CMD_SET_LEGACY", "l"),            # Set legacy mode (1=on, 0=off)

    # OK configuration commands
    "OK_OFFSET_DELAY": os.environ.get("WATCHER_CMD_OK_OFFSET_DELAY", "o,od"),      # Set OK offset delay in ms
    "OK_DURATION_PULSES": os.environ.get("WATCHER_CMD_OK_DURATION_PULSES", "o,dp"),  # Set OK duration in pulses
    "OK_DURATION_PERCENT": os.environ.get("WATCHER_CMD_OK_DURATION_PERCENT", "o,dl"),  # Set OK duration as %
    "OK_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_OK_ENCODER_FACTOR", "o,ef"),  # Set OK encoder factor

    # NG configuration commands
    "NG_OFFSET_DELAY": os.environ.get("WATCHER_CMD_NG_OFFSET_DELAY", "n,od"),      # Set NG offset delay in ms
    "NG_DURATION_PULSES": os.environ.get("WATCHER_CMD_NG_DURATION_PULSES", "n,dp"),  # Set NG duration in pulses
    "NG_DURATION_PERCENT": os.environ.get("WATCHER_CMD_NG_DURATION_PERCENT", "n,dl"),  # Set NG duration as %
    "NG_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_NG_ENCODER_FACTOR", "n,ef"),  # Set NG encoder factor
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
# STATE MANAGEMENT CLASSES (inspired by fabriqc-local-server)
# =============================================================================

class CaptureState(Enum):
    """Capture states for camera triggering."""
    IDLE = "idle"              # No capture pending
    READY = "ready"            # Ready to capture on trigger
    CAPTURING = "capturing"    # Currently capturing
    PROCESSING = "processing"  # Processing captured images
    ERROR = "error"           # Error state


@dataclass
class CapturePhase:
    """Represents a single capture phase (light mode + delay + cameras + trigger thresholds).

    A capture sequence can have multiple phases, each with different light settings.
    Example: Phase 1 (U light) captures cam 1,2,3, Phase 2 (B light) captures cam 4.

    steps threshold:
      -1 = infinite loop (always capture, no encoder check)
       1 = capture on every 1 step change (default)
       N = capture on every N step changes
    """
    light_mode: str = "U_ON_B_OFF"  # Light mode: U_ON_B_OFF, U_OFF_B_ON, U_ON_B_ON, U_OFF_B_OFF
    delay: float = 0.13  # Delay in seconds after setting light before capture
    cameras: List[int] = field(default_factory=lambda: [1, 2, 3])  # Camera IDs to capture in this phase
    steps: int = 1  # Step/encoder count threshold (1 = every step, -1 = infinite loop)
    analog: int = -1  # Analog value threshold (-1 = disabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "light_mode": self.light_mode,
            "delay": self.delay,
            "cameras": self.cameras,
            "steps": self.steps,
            "analog": self.analog
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapturePhase':
        return cls(
            light_mode=data.get('light_mode', 'U_ON_B_OFF'),
            delay=float(data.get('delay', 0.13)),
            cameras=data.get('cameras', [1, 2, 3]),
            steps=int(data.get('steps', 1)),  # Default: 1 step change
            analog=int(data.get('analog', -1))
        )


def _get_default_cameras() -> List[int]:
    """Get default camera IDs based on detected cameras."""
    num_cameras = len(DETECTED_CAMERAS) if DETECTED_CAMERAS else 4
    return list(range(1, num_cameras + 1))


@dataclass
class State:
    """Represents a capture state configuration with multiple phases.

    This class defines a complete capture sequence:
    - Multiple capture phases (each with light mode, delay, cameras)
    - Whether state is enabled
    - Trigger thresholds (steps and analog) for when to activate

    Default: Single phase with U_ON_B_OFF light, 0.1s delay, all detected cameras.
    """
    name: str
    phases: List[CapturePhase] = field(default_factory=lambda: [
        CapturePhase(light_mode="U_ON_B_OFF", delay=0.1, cameras=_get_default_cameras())
    ])
    enabled: bool = True
    # Trigger thresholds (kept for backward compatibility, prefer phase-level thresholds)
    steps: int = 1  # Step/encoder count threshold (1 = every step, -1 = infinite loop)
    analog: int = -1  # Analog value threshold (-1 = disabled)

    def should_trigger(self, encoder_value: int, analog_value: int) -> bool:
        """Check if state trigger conditions are met.

        Args:
            encoder_value: Current encoder/step count
            analog_value: Current analog sensor value

        Returns:
            True if both thresholds are met (or disabled), False otherwise
        """
        steps_met = self.steps < 0 or encoder_value >= self.steps
        analog_met = self.analog < 0 or analog_value >= self.analog
        return steps_met and analog_met

    def get_all_cameras(self) -> List[int]:
        """Get all cameras used across all phases."""
        cameras = []
        for phase in self.phases:
            cameras.extend(phase.cameras)
        return list(set(cameras))

    def is_camera_active(self, camera_id: int) -> bool:
        """Check if a specific camera is active in any phase."""
        return camera_id in self.get_all_cameras()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "phases": [p.to_dict() for p in self.phases],
            "enabled": self.enabled,
            "steps": self.steps,
            "analog": self.analog
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create a State instance from a dictionary."""
        phases_data = data.get('phases', [])
        if phases_data:
            phases = [CapturePhase.from_dict(p) for p in phases_data]
        else:
            # Legacy format support - convert old single-phase format
            phases = [CapturePhase(
                light_mode=data.get('light_mode', 'U_ON_B_OFF'),
                delay=float(data.get('delay', 0.13)),
                cameras=data.get('active_cameras', [1, 2, 3, 4])
            )]

        return cls(
            name=data.get('name', 'default'),
            phases=phases,
            enabled=data.get('enabled', True),
            steps=int(data.get('steps', 1)),
            analog=int(data.get('analog', -1))
        )


class StateManager:
    """Manages state transitions and camera capture synchronization.

    This class handles:
    - State transitions based on encoder and analog values
    - Camera synchronization - which cameras are active
    - Light mode changes
    - Capture timing and delays
    """

    def __init__(self, watcher=None):
        self.watcher = watcher
        self.current_state: Optional[State] = None
        self.capture_state: CaptureState = CaptureState.IDLE
        self.states: Dict[str, State] = {}  # Named states
        self.last_light_mode: Optional[str] = None

        # State synchronization
        self.state_lock = threading.Lock()
        self.state_ready = threading.Event()
        self.capture_complete = threading.Event()

        # Camera-specific events
        self.camera_ready: Dict[int, threading.Event] = {}
        self.camera_complete: Dict[int, threading.Event] = {}

        # Metrics
        self.last_error: Optional[str] = None
        self.error_time: Optional[float] = None
        self.capture_count: int = 0
        self.last_capture_time: Optional[float] = None

        # Initialize default state
        self._init_default_states()

    def _init_default_states(self):
        """Initialize default states using all detected cameras.

        Creates a simple default state that captures all available cameras
        with a single phase (U_ON_B_OFF light, 0.1s delay).
        Steps=-1 means capture on any encoder change.
        """
        # Determine available camera IDs based on watcher's initialized cameras
        if watcher_instance and watcher_instance.camera_paths:
            num_cameras = len(watcher_instance.camera_paths)
            all_camera_ids = list(range(1, num_cameras + 1))
        else:
            # No cameras available
            num_cameras = 0
            all_camera_ids = []

        if num_cameras > 0:
            logger.info(f"Initializing default state with {num_cameras} camera(s): {all_camera_ids}")
        else:
            logger.info("Initializing default state with no cameras. Add cameras via IP Camera Discovery.")

        # Default state: Single phase capturing all cameras with 0.1s delay
        # steps=1 means trigger on every 1 step change (default)
        self.states["default"] = State(
            name="default",
            phases=[
                CapturePhase(
                    light_mode="U_ON_B_OFF",
                    delay=0.1,
                    cameras=all_camera_ids,
                    steps=1,  # 1 = capture on every step change
                    analog=-1  # -1 = no analog threshold
                )
            ],
            enabled=True
        )
        self.current_state = self.states["default"]

    def add_state(self, state: State) -> bool:
        """Add or update a named state."""
        try:
            with self.state_lock:
                self.states[state.name] = state
                logger.info(f"Added/updated state: {state.name}")
                return True
        except Exception as e:
            self._handle_error(f"Error adding state: {e}")
            return False

    def remove_state(self, name: str) -> bool:
        """Remove a named state."""
        try:
            with self.state_lock:
                if name == "default":
                    logger.warning("Cannot remove default state")
                    return False
                if name in self.states:
                    del self.states[name]
                    logger.info(f"Removed state: {name}")
                    return True
                return False
        except Exception as e:
            self._handle_error(f"Error removing state: {e}")
            return False

    def set_current_state(self, name: str) -> bool:
        """Set the current active state by name."""
        try:
            with self.state_lock:
                if name not in self.states:
                    logger.error(f"State not found: {name}")
                    return False

                state = self.states[name]

                # Handle light mode change for first phase if needed
                if state.phases:
                    first_phase_mode = state.phases[0].light_mode
                    if first_phase_mode != self.last_light_mode:
                        self._handle_light_mode_change(first_phase_mode)
                    self.last_light_mode = first_phase_mode

                self.current_state = state
                self.state_ready.set()
                logger.info(f"Set current state to: {name}")
                return True
        except Exception as e:
            self._handle_error(f"Error setting state: {e}")
            return False

    def _handle_light_mode_change(self, new_mode: str) -> None:
        """Handle light mode changes via watcher commands."""
        if self.watcher is None:
            return
        try:
            # Map light mode to watcher command
            mode_commands = {
                "U_ON_B_OFF": "U_ON_B_OFF",
                "U_OFF_B_ON": "B_ON_U_OFF",
                "U_ON_B_ON": "U_ON_B_ON",
                "U_OFF_B_OFF": "B_OFF_U_OFF"
            }
            if new_mode in mode_commands:
                cmd = mode_commands[new_mode]
                # Use watcher's light control methods
                if hasattr(self.watcher, 'send_command'):
                    self.watcher.send_command(cmd)
                logger.info(f"Changed light mode to: {new_mode}")
        except Exception as e:
            self._handle_error(f"Error changing light mode: {e}")

    def _handle_error(self, error_msg: str) -> None:
        """Handle errors consistently."""
        self.last_error = error_msg
        self.error_time = time.time()
        self.capture_state = CaptureState.ERROR
        logger.error(error_msg)

    def is_camera_active(self, camera_id: int) -> bool:
        """Check if a camera is active in the current state."""
        if self.current_state is None:
            return True  # Default to active if no state set
        return self.current_state.is_camera_active(camera_id)

    def should_capture(self, encoder_value: int = 0, analog_value: int = 0) -> bool:
        """Check if capture should be triggered based on current state.

        Args:
            encoder_value: Current encoder/step count since last capture
            analog_value: Current analog sensor value

        Returns:
            True if capture should be triggered
        """
        if self.current_state is None:
            return True  # Default to always capture if no state
        if not self.current_state.enabled:
            return False
        if len(self.current_state.phases) == 0:
            return False
        # Check trigger thresholds (steps and analog)
        return self.current_state.should_trigger(encoder_value, analog_value)

    def trigger_capture(self) -> bool:
        """Trigger a capture cycle."""
        try:
            with self.state_lock:
                if self.capture_state == CaptureState.CAPTURING:
                    logger.warning("Capture already in progress")
                    return False

                self.capture_state = CaptureState.CAPTURING
                self.capture_complete.clear()

                # Note: Delays are handled per-phase in capture_frames(), not here

                self.capture_count += 1
                self.last_capture_time = time.time()
                return True
        except Exception as e:
            self._handle_error(f"Error triggering capture: {e}")
            return False

    def complete_capture(self) -> None:
        """Signal that capture is complete."""
        with self.state_lock:
            self.capture_state = CaptureState.IDLE
            self.capture_complete.set()

    def get_status(self) -> Dict[str, Any]:
        """Get current state manager status."""
        return {
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "capture_state": self.capture_state.value,
            "states": {name: s.to_dict() for name, s in self.states.items()},
            "capture_count": self.capture_count,
            "last_capture_time": self.last_capture_time,
            "last_error": self.last_error,
            "error_time": self.error_time
        }

    def save_states(self, filepath: str) -> bool:
        """Save all states to a file."""
        try:
            data = {name: s.to_dict() for name, s in self.states.items()}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved states to {filepath}")
            return True
        except Exception as e:
            self._handle_error(f"Error saving states: {e}")
            return False

    def load_states(self, filepath: str) -> bool:
        """Load states from a file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"States file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                data = json.load(f)

            for name, state_data in data.items():
                self.states[name] = State.from_dict(state_data)

            logger.info(f"Loaded {len(data)} states from {filepath}")
            return True
        except Exception as e:
            self._handle_error(f"Error loading states: {e}")
            return False


# Global state manager instance (initialized after watcher)
state_manager: Optional[StateManager] = None


# =============================================================================
# INFERENCE PIPELINE MANAGEMENT
# =============================================================================

@dataclass
class InferenceModel:
    """Represents a single inference model configuration.

    A model defines how to run inference on an image:
    - name: Human-readable name
    - model_type: "gradio" or "yolo"
    - inference_url: API endpoint URL
    - model_name: Specific model to use (for Gradio)
    - confidence_threshold: Minimum confidence for detections
    """
    name: str
    model_type: str = "gradio"  # "gradio" or "yolo"
    inference_url: str = "https://smartfalcon-ai-industrial-defect-detection.hf.space"
    model_name: str = "Data Matrix"
    confidence_threshold: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "inference_url": self.inference_url,
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceModel':
        """Create an InferenceModel from a dictionary."""
        return cls(
            name=data.get("name", "Unknown Model"),
            model_type=data.get("model_type", "gradio"),
            inference_url=data.get("inference_url", ""),
            model_name=data.get("model_name", "Data Matrix"),
            confidence_threshold=float(data.get("confidence_threshold", 0.25))
        )


@dataclass
class PipelinePhase:
    """A single phase in an inference pipeline.

    Each phase runs a specific model on the captured images.
    Multiple phases allow sequential processing (e.g., defect detection then classification).
    """
    model_id: str  # Reference to a model by ID
    enabled: bool = True
    order: int = 0  # Order in pipeline (lower = first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "enabled": self.enabled,
            "order": self.order
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelinePhase':
        return cls(
            model_id=data.get("model_id", "default_gradio"),
            enabled=data.get("enabled", True),
            order=int(data.get("order", 0))
        )


@dataclass
class Pipeline:
    """Represents an inference pipeline configuration.

    A pipeline consists of one or more phases, each running a model.
    This is similar to State for capture - Pipeline is for inference.
    """
    name: str
    phases: List[PipelinePhase] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phases": [p.to_dict() for p in self.phases],
            "enabled": self.enabled,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        phases_data = data.get("phases", [])
        phases = [PipelinePhase.from_dict(p) for p in phases_data] if phases_data else []
        # Sort phases by order
        phases.sort(key=lambda p: p.order)

        return cls(
            name=data.get("name", "default"),
            phases=phases,
            enabled=data.get("enabled", True),
            description=data.get("description", "")
        )


class PipelineManager:
    """Manages inference pipelines and models.

    Similar to StateManager for capture, this handles:
    - Available inference models (Gradio, YOLO, etc.)
    - Pipeline definitions (sequences of models)
    - Active pipeline selection
    - Running inference through the active pipeline
    """

    # Default Gradio model URL
    DEFAULT_GRADIO_URL = "https://smartfalcon-ai-industrial-defect-detection.hf.space"
    # Default YOLO model URL
    DEFAULT_YOLO_URL = "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/"

    def __init__(self):
        self.models: Dict[str, InferenceModel] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.current_pipeline: Optional[Pipeline] = None
        self.pipeline_lock = threading.Lock()

        # Gradio client cache (for reuse)
        self._gradio_clients: Dict[str, Any] = {}

        # Initialize default models and pipeline
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default models and pipeline."""
        # Default Gradio model (HuggingFace)
        self.models["default_gradio"] = InferenceModel(
            name="Gradio HuggingFace",
            model_type="gradio",
            inference_url=self.DEFAULT_GRADIO_URL,
            model_name="Data Matrix",
            confidence_threshold=0.25
        )

        # Default YOLO model (local)
        self.models["default_yolo"] = InferenceModel(
            name="Local YOLO",
            model_type="yolo",
            inference_url=self.DEFAULT_YOLO_URL,
            model_name="yolov5s",
            confidence_threshold=0.3
        )

        # Default pipeline using Gradio
        default_pipeline = Pipeline(
            name="default",
            phases=[
                PipelinePhase(model_id="default_gradio", enabled=True, order=0)
            ],
            enabled=True,
            description="Default pipeline using Gradio HuggingFace model"
        )
        self.pipelines["default"] = default_pipeline
        self.current_pipeline = default_pipeline

        logger.info(f"PipelineManager initialized with {len(self.models)} models, {len(self.pipelines)} pipelines")

    def add_model(self, model_id: str, model: InferenceModel) -> bool:
        """Add or update an inference model."""
        try:
            with self.pipeline_lock:
                self.models[model_id] = model
                logger.info(f"Added/updated model: {model_id} ({model.name})")
                return True
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return False

    def remove_model(self, model_id: str) -> bool:
        """Remove an inference model."""
        try:
            with self.pipeline_lock:
                if model_id.startswith("default_"):
                    logger.warning(f"Cannot remove default model: {model_id}")
                    return False
                if model_id in self.models:
                    del self.models[model_id]
                    logger.info(f"Removed model: {model_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            return False

    def add_pipeline(self, pipeline: Pipeline) -> bool:
        """Add or update a pipeline."""
        try:
            with self.pipeline_lock:
                self.pipelines[pipeline.name] = pipeline
                logger.info(f"Added/updated pipeline: {pipeline.name}")
                return True
        except Exception as e:
            logger.error(f"Error adding pipeline: {e}")
            return False

    def remove_pipeline(self, name: str) -> bool:
        """Remove a pipeline."""
        try:
            with self.pipeline_lock:
                if name == "default":
                    logger.warning("Cannot remove default pipeline")
                    return False
                if name in self.pipelines:
                    del self.pipelines[name]
                    logger.info(f"Removed pipeline: {name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing pipeline: {e}")
            return False

    def set_current_pipeline(self, name: str) -> bool:
        """Set the currently active pipeline."""
        try:
            with self.pipeline_lock:
                if name not in self.pipelines:
                    logger.error(f"Pipeline not found: {name}")
                    return False
                self.current_pipeline = self.pipelines[name]
                logger.info(f"Set current pipeline to: {name}")
                return True
        except Exception as e:
            logger.error(f"Error setting pipeline: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[InferenceModel]:
        """Get a model by ID."""
        return self.models.get(model_id)

    def get_current_model(self) -> Optional[InferenceModel]:
        """Get the first enabled model in the current pipeline."""
        if not self.current_pipeline or not self.current_pipeline.phases:
            return self.models.get("default_gradio")

        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if phase.enabled and phase.model_id in self.models:
                return self.models[phase.model_id]

        return self.models.get("default_gradio")

    def run_inference(self, image_bytes: bytes) -> tuple:
        """Run inference through the current pipeline.

        Returns:
            Tuple of (detections_list, model_name_used)
        """
        if not self.current_pipeline:
            logger.warning("No pipeline set, using default model")
            model = self.models.get("default_gradio")
            if model:
                return self._run_model_inference(image_bytes, model), model.name
            return [], "unknown"

        all_detections = []
        models_used = []

        # Run through each enabled phase in order
        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if not phase.enabled:
                continue

            model = self.models.get(phase.model_id)
            if not model:
                logger.warning(f"Model not found for phase: {phase.model_id}")
                continue

            try:
                detections = self._run_model_inference(image_bytes, model)
                if detections:
                    all_detections.extend(detections)
                    models_used.append(model.name)
            except Exception as e:
                logger.error(f"Error running inference with model {model.name}: {e}")

        model_names = ", ".join(models_used) if models_used else "none"
        return all_detections, model_names

    def _run_model_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference using a specific model."""
        if model.model_type == "gradio":
            return self._run_gradio_inference(image_bytes, model)
        elif model.model_type == "yolo":
            return self._run_yolo_inference(image_bytes, model)
        else:
            logger.error(f"Unknown model type: {model.model_type}")
            return []

    def _run_gradio_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference through Gradio API."""
        try:
            from gradio_client import Client, handle_file
            import tempfile

            # Get or create cached client
            if model.inference_url not in self._gradio_clients:
                logger.info(f"Initializing Gradio client for {model.inference_url}")
                self._gradio_clients[model.inference_url] = Client(model.inference_url)

            client = self._gradio_clients[model.inference_url]

            # Write image to temp file (Gradio needs file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            try:
                # Call Gradio API
                result = client.predict(
                    handle_file(tmp_path),
                    model.model_name,
                    model.confidence_threshold,
                    api_name="/detect"
                )

                # Convert Gradio response to standard format
                detections = []
                if isinstance(result, list):
                    for det in result:
                        if isinstance(det, dict):
                            detections.append({
                                "xmin": det.get("x1", det.get("xmin", 0)),
                                "ymin": det.get("y1", det.get("ymin", 0)),
                                "xmax": det.get("x2", det.get("xmax", 0)),
                                "ymax": det.get("y2", det.get("ymax", 0)),
                                "confidence": det.get("confidence", 0),
                                "class": det.get("class_id", det.get("class", 0)),
                                "name": det.get("name", f"Class {det.get('class_id', 0)}")
                            })

                return detections

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Gradio inference error: {e}")
            return []

    def _run_yolo_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference through YOLO API."""
        try:
            response = requests.post(
                model.inference_url,
                files={"image": image_bytes},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline manager status."""
        return {
            "models": {mid: m.to_dict() for mid, m in self.models.items()},
            "pipelines": {name: p.to_dict() for name, p in self.pipelines.items()},
            "current_pipeline": self.current_pipeline.name if self.current_pipeline else None,
            "current_model": self.get_current_model().to_dict() if self.get_current_model() else None
        }

    def to_config(self) -> Dict[str, Any]:
        """Export configuration for saving to DATA_FILE."""
        return {
            "models": {mid: m.to_dict() for mid, m in self.models.items()},
            "pipelines": {name: p.to_dict() for name, p in self.pipelines.items()},
            "current_pipeline": self.current_pipeline.name if self.current_pipeline else "default"
        }

    def from_config(self, config: Dict[str, Any]) -> bool:
        """Load configuration from DATA_FILE."""
        try:
            # Load models
            if "models" in config:
                for model_id, model_data in config["models"].items():
                    self.models[model_id] = InferenceModel.from_dict(model_data)

            # Load pipelines
            if "pipelines" in config:
                for name, pipeline_data in config["pipelines"].items():
                    self.pipelines[name] = Pipeline.from_dict(pipeline_data)

            # Set current pipeline
            current_name = config.get("current_pipeline", "default")
            if current_name in self.pipelines:
                self.current_pipeline = self.pipelines[current_name]

            logger.info(f"Loaded pipeline config: {len(self.models)} models, {len(self.pipelines)} pipelines")
            return True
        except Exception as e:
            logger.error(f"Error loading pipeline config: {e}")
            return False


# Global pipeline manager instance (initialized at startup)
pipeline_manager: Optional[PipelineManager] = None

# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Counter Service",
    description="Service for counting and tracking packages with camera processing",
    version="1.0.0"
)

# Mount static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Static directory might not exist

# Global reference to watcher instance (set after ArduinoSocket initialization)
watcher_instance = None

def load_data_file():
    """Load entire data file. Returns dict with 'data' key for list content, or the dict itself."""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                content = json.load(f)
                # If the file contains a list (original format), wrap it in a dict
                if isinstance(content, list):
                    return {"prepared_query_data": content}
                return content
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
    return {}

def save_data_file(data):
    """Save entire data file with backup."""
    try:
        # Create backup if file exists
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

# Alias for backwards compatibility
load_camera_config = load_service_config
save_camera_config = save_service_config


def apply_config_settings(config, watcher_inst=None):
    """Apply configuration settings from a config dict.

    This is a helper function used by startup loading, API endpoints, and config upload.
    Returns tuple of (settings_applied: dict, cameras_loaded: int).
    """
    global EJECTOR_ENABLED, EJECTOR_OFFSET, EJECTOR_DURATION, EJECTOR_POLL_INTERVAL
    global CAPTURE_MODE, TIME_BETWEEN_TWO_PACKAGE, REMOVE_RAW_IMAGE_WHEN_DM_DECODED
    global PARENT_OBJECT_LIST, HISTOGRAM_ENABLED, HISTOGRAM_SAVE_IMAGE
    global CHECK_CLASS_COUNTS_ENABLED, CHECK_CLASS_COUNTS_CLASSES, CHECK_CLASS_COUNTS_CONFIDENCE
    global DM_CHARS_SIZES, DM_CONFIDENCE_THRESHOLD, DM_OVERLAP_THRESHOLD
    global STORE_ANNOTATION_ENABLED, ENFORCE_PARENT_OBJECT
    global YOLO_INFERENCE_URL, GRADIO_MODEL, GRADIO_CONFIDENCE_THRESHOLD
    global capture_mode, time_between_two_package, remove_raw_image_when_dm_decoded, parent_object_list
    global SERIAL_MODE

    settings_applied = {}

    # Apply inference module configuration
    if "inference" in config:
        modules = config["inference"].get("modules", {})
        current_module = config["inference"].get("current_module")

        if current_module and current_module in modules:
            active_module = modules[current_module]
            YOLO_INFERENCE_URL = active_module.get("inference_url", YOLO_INFERENCE_URL)
            GRADIO_MODEL = active_module.get("inference_model", GRADIO_MODEL)
            GRADIO_CONFIDENCE_THRESHOLD = active_module.get("inference_min_confidence", GRADIO_CONFIDENCE_THRESHOLD)
            settings_applied["inference"] = True

            logger.info(f"Inference module loaded: {active_module.get('name', current_module)}")
            logger.info(f"  URL: {YOLO_INFERENCE_URL}")
            logger.info(f"  Model: {GRADIO_MODEL}")
            logger.info(f"  Min Confidence: {GRADIO_CONFIDENCE_THRESHOLD}")
    elif "infrastructure" in config:
        # Legacy support for old config format
        YOLO_INFERENCE_URL = config["infrastructure"].get("yolo_url", YOLO_INFERENCE_URL)
        GRADIO_MODEL = config["infrastructure"].get("gradio_model", GRADIO_MODEL)
        GRADIO_CONFIDENCE_THRESHOLD = config["infrastructure"].get("gradio_confidence", GRADIO_CONFIDENCE_THRESHOLD)
        settings_applied["infrastructure"] = True
        logger.info(f"Infrastructure (legacy): YOLO URL={YOLO_INFERENCE_URL}, Gradio Model={GRADIO_MODEL}, Confidence={GRADIO_CONFIDENCE_THRESHOLD}")

    # Apply serial mode from infrastructure config
    if "infrastructure" in config:
        saved_serial_mode = config["infrastructure"].get("serial_mode")
        if saved_serial_mode:
            SERIAL_MODE = saved_serial_mode
            # Update watcher if it exists
            if watcher_inst:
                watcher_inst.serial_mode = SERIAL_MODE
            settings_applied["serial_mode"] = True
            logger.info(f"Serial mode loaded from config: {SERIAL_MODE}")

    # Apply AI configuration (load from DATA_FILE to Redis for runtime use)
    if "ai" in config and watcher_inst and watcher_inst.redis_connection:
        ai_model = config["ai"].get("model", "claude")
        ai_api_key = config["ai"].get("api_key", "")
        if ai_api_key:  # Only load if API key exists
            watcher_inst.redis_connection.redis_connection.set("ai_model", ai_model)
            watcher_inst.redis_connection.redis_connection.set("ai_api_key", ai_api_key)
            settings_applied["ai"] = True
            logger.info(f"AI configuration loaded: model={ai_model}")

    # Apply ejector settings
    if "ejector" in config:
        EJECTOR_ENABLED = config["ejector"].get("enabled", EJECTOR_ENABLED)
        EJECTOR_OFFSET = config["ejector"].get("offset", EJECTOR_OFFSET)
        EJECTOR_DURATION = config["ejector"].get("duration", EJECTOR_DURATION)
        EJECTOR_POLL_INTERVAL = config["ejector"].get("poll_interval", EJECTOR_POLL_INTERVAL)
        settings_applied["ejector"] = True

    # Apply capture settings
    if "capture" in config:
        CAPTURE_MODE = config["capture"].get("mode", CAPTURE_MODE)
        TIME_BETWEEN_TWO_PACKAGE = config["capture"].get("time_between_packages", TIME_BETWEEN_TWO_PACKAGE)
        capture_mode = CAPTURE_MODE
        time_between_two_package = TIME_BETWEEN_TWO_PACKAGE
        settings_applied["capture"] = True

    # Apply image processing settings
    if "image_processing" in config:
        REMOVE_RAW_IMAGE_WHEN_DM_DECODED = config["image_processing"].get("remove_raw_image_when_dm_decoded", REMOVE_RAW_IMAGE_WHEN_DM_DECODED)
        PARENT_OBJECT_LIST = config["image_processing"].get("parent_object_list", PARENT_OBJECT_LIST)
        ENFORCE_PARENT_OBJECT = config["image_processing"].get("enforce_parent_object", ENFORCE_PARENT_OBJECT)
        remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
        parent_object_list = PARENT_OBJECT_LIST
        settings_applied["image_processing"] = True

    # Apply histogram settings
    if "histogram" in config:
        HISTOGRAM_ENABLED = config["histogram"].get("enabled", HISTOGRAM_ENABLED)
        HISTOGRAM_SAVE_IMAGE = config["histogram"].get("save_image", HISTOGRAM_SAVE_IMAGE)
        settings_applied["histogram"] = True

    # Apply class count check settings
    if "class_count_check" in config:
        CHECK_CLASS_COUNTS_ENABLED = config["class_count_check"].get("enabled", CHECK_CLASS_COUNTS_ENABLED)
        CHECK_CLASS_COUNTS_CLASSES = config["class_count_check"].get("classes", CHECK_CLASS_COUNTS_CLASSES)
        CHECK_CLASS_COUNTS_CONFIDENCE = config["class_count_check"].get("confidence", CHECK_CLASS_COUNTS_CONFIDENCE)
        settings_applied["class_count_check"] = True

    # Apply datamatrix settings
    if "datamatrix" in config:
        DM_CHARS_SIZES = config["datamatrix"].get("chars_sizes", DM_CHARS_SIZES)
        DM_CONFIDENCE_THRESHOLD = config["datamatrix"].get("confidence_threshold", DM_CONFIDENCE_THRESHOLD)
        DM_OVERLAP_THRESHOLD = config["datamatrix"].get("overlap_threshold", DM_OVERLAP_THRESHOLD)
        settings_applied["datamatrix"] = True

    # Apply store annotation settings
    if "store_annotation" in config:
        STORE_ANNOTATION_ENABLED = config["store_annotation"].get("enabled", STORE_ANNOTATION_ENABLED)
        settings_applied["store_annotation"] = True

    # Apply/load cameras from unified cameras structure
    cameras_loaded = 0
    if "cameras" in config and watcher_inst is not None:
        for cam_id_str, cam_config in config["cameras"].items():
            if not cam_config.get("enabled", True):
                logger.info(f"  Skipping disabled camera {cam_id_str}")
                continue

            cam_id = int(cam_id_str)
            cam_type = cam_config.get("type", "usb")

            # Check if camera already exists
            cam = watcher_inst.cameras.get(cam_id) if hasattr(watcher_inst, 'cameras') else None

            if cam is not None:
                # Camera exists - apply config to existing camera
                apply_camera_config_from_saved(cam, cam_config)
                cameras_loaded += 1
                logger.info(f"  Updated existing camera {cam_id}: {cam_config.get('name', f'Camera {cam_id}')}")
            elif cam_type == "ip":
                # IP camera doesn't exist - create it
                cam_url = cam_config.get("source") or cam_config.get("url")  # Support both "source" and legacy "url"
                cam_name = cam_config.get("name", f"IP Camera {cam_id}")

                if not cam_url:
                    logger.warning(f"  Skipping camera {cam_name}: missing URL")
                    continue

                try:
                    logger.info(f"  Initializing IP camera {cam_id}: {cam_name}")

                    # Create camera buffer for IP camera
                    cam = CameraBuffer(cam_url, exposure=cam_config.get("exposure", 100), gain=cam_config.get("gain", 100))
                    watcher_inst.cameras[cam_id] = cam

                    # Ensure camera_paths list is large enough
                    while len(watcher_inst.camera_paths) < cam_id:
                        watcher_inst.camera_paths.append(None)
                    if cam_id > len(watcher_inst.camera_paths):
                        watcher_inst.camera_paths.append(cam_url)
                    else:
                        watcher_inst.camera_paths[cam_id - 1] = cam_url

                    # Store camera metadata
                    if not hasattr(watcher_inst, 'camera_metadata'):
                        watcher_inst.camera_metadata = {}
                    watcher_inst.camera_metadata[cam_id] = {
                        "name": cam_name,
                        "type": "ip",
                        "ip": cam_config.get("ip"),
                        "path": cam_config.get("path"),
                        "url": cam_url
                    }

                    # Apply camera settings (ROI, etc.)
                    apply_camera_config_from_saved(cam, cam_config)

                    cameras_loaded += 1
                    logger.info(f"  ✓ IP Camera {cam_id} initialized: {cam_name} (success={cam.success})")

                except Exception as e:
                    logger.error(f"  ✗ Failed to initialize camera {cam_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    # IP cameras are now loaded from the unified "cameras" structure above

    return settings_applied, cameras_loaded

def apply_saved_config_at_startup(watcher_inst):
    """Load and apply saved service configuration at startup."""
    try:
        config = load_service_config()
        if not config:
            logger.info("No saved configuration found - using defaults")
            return False

        logger.info(f"Loading saved configuration from {DATA_FILE} (saved at: {config.get('saved_at', 'unknown')})")

        # Use the helper function to apply all settings
        settings_applied, cameras_loaded = apply_config_settings(config, watcher_inst)

        # Log what was applied
        for setting in settings_applied:
            logger.info(f"  {setting}: configured")
        if cameras_loaded > 0:
            logger.info(f"  Cameras: {cameras_loaded} camera(s) configured")

        # States are loaded separately when state_manager is available
        if "states" in config:
            logger.info(f"  States config found: {len(config['states'])} state(s) (will load after state_manager init)")

        logger.info("Saved configuration applied successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading saved config at startup: {e}")
        return False


def apply_states_from_config(sm):
    """Apply saved states configuration to state manager."""
    global state_manager
    try:
        config = load_service_config()
        if not config or "states" not in config:
            logger.info("No saved states configuration found")
            return False

        states_config = config["states"]
        states_loaded = 0

        for state_name, state_data in states_config.items():
            state = State.from_dict(state_data)
            sm.add_state(state)
            states_loaded += 1

        # Set current state if specified
        current_state_name = config.get("current_state_name", "default")
        if current_state_name in sm.states:
            sm.set_current_state(current_state_name)

        logger.info(f"Loaded {states_loaded} states from saved config, current: {current_state_name}")
        return True

    except Exception as e:
        logger.error(f"Error applying states from config: {e}")
        return False


def apply_pipeline_config_at_startup(pm):
    """Apply saved pipeline configuration to pipeline manager."""
    global pipeline_manager
    try:
        config = load_service_config()
        if not config or "pipeline_config" not in config:
            logger.info("No saved pipeline configuration found - using defaults")
            return False

        pipeline_config = config["pipeline_config"]
        pm.from_config(pipeline_config)

        logger.info(f"Loaded pipeline config: current pipeline={pm.current_pipeline.name if pm.current_pipeline else 'none'}")
        return True

    except Exception as e:
        logger.error(f"Error applying pipeline config: {e}")
        return False


def get_camera_config_for_save(cam, cam_id, camera_metadata=None):
    """Extract camera configuration for saving."""
    if cam is None:
        return None

    config = {
        "roi_enabled": getattr(cam, 'roi_enabled', False),
        "roi_xmin": getattr(cam, 'roi_xmin', 0),
        "roi_ymin": getattr(cam, 'roi_ymin', 0),
        "roi_xmax": getattr(cam, 'roi_xmax', 1280),
        "roi_ymax": getattr(cam, 'roi_ymax', 720),
    }

    # Include name from metadata if available
    if camera_metadata and cam_id in camera_metadata:
        meta = camera_metadata[cam_id]
        config["name"] = meta.get("name", f"Camera {cam_id}")
        # Also include type from metadata if available
        if "type" in meta:
            config["type"] = meta["type"]

    # Add camera source (URL/path)
    if hasattr(cam, 'source'):
        config["source"] = cam.source
        # Determine camera type if not already set from metadata
        if "type" not in config:
            if isinstance(cam.source, str) and cam.source.startswith(("rtsp://", "http://", "https://")):
                config["type"] = "ip"
            else:
                config["type"] = "usb"

        # Extract IP and path from RTSP URL if IP camera
        if config.get("type") == "ip":
            try:
                from urllib.parse import urlparse
                parsed = urlparse(cam.source)
                config["ip"] = parsed.hostname
                config["path"] = parsed.path + ("?" + parsed.query if parsed.query else "")
            except:
                pass

    # Set default name if not already set
    if "name" not in config:
        cam_type = config.get("type", "usb")
        config["name"] = f"{'IP' if cam_type == 'ip' else 'USB'} Camera {cam_id}"

    if hasattr(cam, 'camera'):
        try:
            config["exposure"] = int(cam.camera.get(cv2.CAP_PROP_EXPOSURE))
            config["gain"] = int(cam.camera.get(cv2.CAP_PROP_GAIN))
            config["brightness"] = int(cam.camera.get(cv2.CAP_PROP_BRIGHTNESS))
            config["contrast"] = int(cam.camera.get(cv2.CAP_PROP_CONTRAST))
            config["saturation"] = int(cam.camera.get(cv2.CAP_PROP_SATURATION))
            config["fps"] = int(cam.camera.get(cv2.CAP_PROP_FPS))
        except:
            pass

    return config

def apply_camera_config_from_saved(cam, saved_config):
    """Apply saved configuration to camera."""
    if cam is None or saved_config is None:
        return

    # Apply ROI settings
    cam.roi_enabled = saved_config.get('roi_enabled', False)
    cam.roi_xmin = saved_config.get('roi_xmin', 0)
    cam.roi_ymin = saved_config.get('roi_ymin', 0)
    cam.roi_xmax = saved_config.get('roi_xmax', 1280)
    cam.roi_ymax = saved_config.get('roi_ymax', 720)

    # Apply OpenCV settings
    if hasattr(cam, 'camera'):
        try:
            if 'exposure' in saved_config:
                cam.camera.set(cv2.CAP_PROP_EXPOSURE, saved_config['exposure'])
            if 'gain' in saved_config:
                cam.camera.set(cv2.CAP_PROP_GAIN, saved_config['gain'])
            if 'brightness' in saved_config:
                cam.camera.set(cv2.CAP_PROP_BRIGHTNESS, saved_config['brightness'])
            if 'contrast' in saved_config:
                cam.camera.set(cv2.CAP_PROP_CONTRAST, saved_config['contrast'])
            if 'saturation' in saved_config:
                cam.camera.set(cv2.CAP_PROP_SATURATION, saved_config['saturation'])
            if 'fps' in saved_config:
                cam.camera.set(cv2.CAP_PROP_FPS, saved_config['fps'])
        except Exception as e:
            logger.error(f"Error applying camera config: {e}")

def format_relative_time(timestamp):
    """Format timestamp as relative time (e.g., '5 seconds ago')."""
    if not timestamp:
        return "never"
    diff = time.time() - timestamp
    if diff < 5:
        return "just now"
    elif diff < 60:
        return f"{int(diff)} seconds ago"
    elif diff < 3600:
        return f"{int(diff/60)} minutes ago"
    else:
        return f"{int(diff/3600)} hours ago"

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    global YOLO_INFERENCE_URL

    try:
        # Get watcher from app state (more reliable than global)
        watcher = getattr(request.app.state, 'watcher', None)

        # Check if watcher is available
        device_ok = watcher is not None and watcher.health_check

        # Check Redis connection
        redis_ok = False
        if watcher and watcher.redis_connection:
            try:
                watcher.redis_connection.redis_connection.ping()
                redis_ok = True
            except:
                pass

        # Check cameras dynamically
        cameras_status = {}
        if watcher and hasattr(watcher, 'cameras'):
            for cam_id, cam in watcher.cameras.items():
                cameras_status[f"cam_{cam_id}"] = cam is not None and cam.success if cam else False

        # Check serial availability
        serial_available = watcher.serial_available if watcher else False

        # Check YOLO/Gradio inference availability
        yolo_ok = False
        try:
            # Check if Gradio client is initialized (for HuggingFace URLs)
            if "hf.space" in YOLO_INFERENCE_URL or "huggingface" in YOLO_INFERENCE_URL:
                gradio_health_status = "unknown"
                gradio_needs_restart = False

                # Try comprehensive health check
                try:
                    # 1. Check if health endpoint responds
                    base_url = YOLO_INFERENCE_URL.split('/api/')[0] if '/api/' in YOLO_INFERENCE_URL else YOLO_INFERENCE_URL.rsplit('/', 1)[0]
                    health_url = f"{base_url}/api/health"

                    health_response = session.get(health_url, timeout=5)

                    if health_response.status_code == 200:
                        # 2. Check recent successful inference from Redis
                        last_detection_ts = 0.0
                        if watcher and watcher.redis_connection:
                            try:
                                cached_ts = watcher.redis_connection.redis_connection.get("gradio_last_detection_timestamp")
                                if cached_ts:
                                    last_detection_ts = float(cached_ts.decode('utf-8'))
                            except Exception as e:
                                logger.warning(f"Failed to read detection timestamp from Redis: {e}")

                        # 3. Determine health based on recent activity
                        if last_detection_ts > 0:
                            age = time.time() - last_detection_ts
                            if age < 60.0:  # Less than 1 minute - healthy
                                yolo_ok = True
                                gradio_health_status = "healthy"
                            elif age < 300.0:  # Less than 5 minutes - warning
                                yolo_ok = True
                                gradio_health_status = "warning"
                            else:  # More than 5 minutes - may need restart
                                yolo_ok = False
                                gradio_health_status = "stale"
                                gradio_needs_restart = True
                            logger.info(f"Gradio health: {gradio_health_status}, last inference {age:.1f}s ago")
                        else:
                            # Health endpoint OK but no recent inferences
                            yolo_ok = True
                            gradio_health_status = "idle"
                            logger.info(f"Gradio health: {gradio_health_status} (no recent inferences)")
                    else:
                        raise Exception(f"Health endpoint returned {health_response.status_code}")

                except Exception as e:
                    # Health endpoint failed - space may be sleeping or needs restart
                    yolo_ok = False
                    gradio_health_status = "offline"
                    gradio_needs_restart = True
                    logger.error(f"Gradio health check failed: {e}")

                # Store status in Redis for monitoring
                try:
                    if watcher and watcher.redis_connection:
                        watcher.redis_connection.redis_connection.set("gradio_health_status", gradio_health_status)
                        watcher.redis_connection.redis_connection.set("gradio_needs_restart", str(gradio_needs_restart))
                except:
                    pass
            else:
                # For traditional YOLO endpoint, try a quick ping
                health_url = YOLO_INFERENCE_URL.replace('/detect/', '/health').replace('/detect', '/health')
                response = session.get(health_url, timeout=1)
                yolo_ok = response.status_code == 200
        except Exception as e:
            logger.error(f"[HEALTH] Error checking YOLO status: {e}")
            yolo_ok = False

        # Consider healthy if redis works and either serial is connected or we're in camera-only mode
        status_code = 200 if redis_ok else 503

        # Get Gradio-specific health info from Redis
        gradio_health_info = {}
        if "hf.space" in YOLO_INFERENCE_URL or "huggingface" in YOLO_INFERENCE_URL:
            try:
                if watcher and watcher.redis_connection:
                    health_status = watcher.redis_connection.redis_connection.get("gradio_health_status")
                    needs_restart = watcher.redis_connection.redis_connection.get("gradio_needs_restart")
                    gradio_health_info = {
                        "gradio_status": health_status.decode('utf-8') if health_status else "unknown",
                        "gradio_needs_restart": needs_restart.decode('utf-8') == "True" if needs_restart else False
                    }
            except:
                pass

        response_content = {
            "status": "healthy" if status_code == 200 else "unhealthy",
            "device": "connected" if device_ok else ("camera-only" if not serial_available else "disconnected"),
            "serial": "connected" if serial_available else "not available",
            "redis": "connected" if redis_ok else "disconnected",
            "yolo": "connected" if yolo_ok else "not available",
            "cameras": cameras_status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add Gradio-specific info if available
        if gradio_health_info:
            response_content.update(gradio_health_info)

        return JSONResponse(
            status_code=status_code,
            content=response_content
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

@app.get("/api/inference/stats")
async def get_inference_stats():
    """Get inference performance statistics."""
    global inference_times, frame_intervals, YOLO_INFERENCE_URL, watcher

    # Determine service type
    if "hf.space" in YOLO_INFERENCE_URL or "huggingface" in YOLO_INFERENCE_URL:
        service_type = "Gradio (Cloud)"
        service_url = YOLO_INFERENCE_URL
    else:
        service_type = "YOLO (Local)"
        service_url = YOLO_INFERENCE_URL

    # Try to get timing data from Redis (cross-process)
    redis_inference_times = []
    redis_frame_intervals = []
    redis_capture_timestamps = []

    try:
        if watcher and watcher.redis_connection:
            # Get inference times from Redis
            times_raw = watcher.redis_connection.redis_connection.lrange("inference_times", 0, -1)
            redis_inference_times = [float(t.decode('utf-8')) for t in times_raw if t]

            # Get frame intervals from Redis
            intervals_raw = watcher.redis_connection.redis_connection.lrange("frame_intervals", 0, -1)
            redis_frame_intervals = [float(i.decode('utf-8')) for i in intervals_raw if i]

            # Get capture timestamps from Redis
            capture_raw = watcher.redis_connection.redis_connection.lrange("capture_timestamps", 0, -1)
            redis_capture_timestamps = [float(t.decode('utf-8')) for t in capture_raw if t]
    except Exception as e:
        logger.warning(f"Failed to read timing from Redis: {e}")

    # Use Redis data if available, otherwise fall back to in-memory
    use_inference_times = redis_inference_times if redis_inference_times else inference_times
    use_frame_intervals = redis_frame_intervals if redis_frame_intervals else frame_intervals

    # Calculate average inference time (processing time only)
    if use_inference_times:
        avg_inference = sum(use_inference_times) / len(use_inference_times)
        min_inference = min(use_inference_times)
        max_inference = max(use_inference_times)
    else:
        avg_inference = 0
        min_inference = 0
        max_inference = 0

    # Calculate average frame interval (time between frames)
    if use_frame_intervals:
        avg_interval = sum(use_frame_intervals) / len(use_frame_intervals)
        min_interval = min(use_frame_intervals)
        max_interval = max(use_frame_intervals)
        # Calculate inference FPS from average interval (1000ms / interval_ms = fps)
        inference_fps = 1000.0 / avg_interval if avg_interval > 0 else 0
    else:
        avg_interval = 0
        min_interval = 0
        max_interval = 0
        inference_fps = 0

    # Calculate capture FPS from capture timestamps
    capture_fps = 0
    if redis_capture_timestamps and len(redis_capture_timestamps) >= 2:
        # Sort timestamps and calculate intervals
        sorted_timestamps = sorted(redis_capture_timestamps)
        capture_intervals = [
            (sorted_timestamps[i] - sorted_timestamps[i-1]) * 1000  # Convert to ms
            for i in range(1, len(sorted_timestamps))
        ]
        if capture_intervals:
            avg_capture_interval = sum(capture_intervals) / len(capture_intervals)
            capture_fps = 1000.0 / avg_capture_interval if avg_capture_interval > 0 else 0

    return JSONResponse(content={
        "service_type": service_type,
        "service_url": service_url,
        # Processing time (from capture to result)
        "avg_inference_time_ms": round(avg_inference, 1),
        "min_inference_time_ms": round(min_inference, 1),
        "max_inference_time_ms": round(max_inference, 1),
        # Frame-to-frame interval (time between processes)
        "avg_frame_interval_ms": round(avg_interval, 1),
        "min_frame_interval_ms": round(min_interval, 1),
        "max_frame_interval_ms": round(max_interval, 1),
        # Inference FPS based on frame intervals
        "inference_fps": round(inference_fps, 2),
        # Capture FPS based on camera capture rate
        "capture_fps": round(capture_fps, 2),
        # Sample counts
        "inference_sample_count": len(use_inference_times),
        "interval_sample_count": len(use_frame_intervals),
        "capture_sample_count": len(redis_capture_timestamps),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.get("/")
async def root_redirect():
    """Redirect root to status page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/status")

@app.get("/status")
async def status_page():
    """Serve the status page with control panel."""
    try:
        return FileResponse("static/status.html")
    except Exception as e:
        logger.error(f"Status page error: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/api/status")
async def api_status():
    """API endpoint for real-time status data."""
    try:
        return JSONResponse(content={
            "encoder_value": watcher_instance.encoder_value if watcher_instance else 0,
            "pps": getattr(watcher_instance, "pulses_per_second", 0) if watcher_instance else 0,
            "ppm": getattr(watcher_instance, "pulses_per_minute", 0) if watcher_instance else 0,
            "downtime_seconds": getattr(watcher_instance, "downtime_seconds", 0) if watcher_instance else 0,
            "ok_counter": getattr(watcher_instance, "ok_counter", 0) if watcher_instance else 0,
            "ng_counter": getattr(watcher_instance, "ng_counter", 0) if watcher_instance else 0,
            "analog_value": getattr(watcher_instance, "analog_value", 0) if watcher_instance else 0,
            "power_value": getattr(watcher_instance, "power_value", 0) if watcher_instance else 0,
            "is_moving": watcher_instance.is_moving if watcher_instance else False,
            "shipment": watcher_instance.shipment if watcher_instance else "no_shipment",
            "ejector_queue_length": len(watcher_instance.ejection_queue) if watcher_instance else 0,
            "ejector_running": watcher_instance.ejector_running if watcher_instance else False,
            "ejector_offset": EJECTOR_OFFSET,
            "ejector_enabled": EJECTOR_ENABLED,
            "histogram_enabled": HISTOGRAM_ENABLED,
            "status": {
                "U": getattr(watcher_instance, "u_status", False) if watcher_instance else False,
                "B": getattr(watcher_instance, "b_status", False) if watcher_instance else False,
                "warning": getattr(watcher_instance, "warning_status", False) if watcher_instance else False,
                "raw": getattr(watcher_instance, "status_value", 0) if watcher_instance else 0,
            } if watcher_instance else {"U": False, "B": False, "warning": False, "raw": 0},
            "verbose_data": {
                "OOD": getattr(watcher_instance, "ok_offset_delay", 0) if watcher_instance else 0,
                "ODP": getattr(watcher_instance, "ok_duration_pulses", 0) if watcher_instance else 0,
                "ODL": getattr(watcher_instance, "ok_duration_percent", 0) if watcher_instance else 0,
                "OEF": getattr(watcher_instance, "ok_encoder_factor", 0) if watcher_instance else 0,
                "NOD": getattr(watcher_instance, "ng_offset_delay", 0) if watcher_instance else 0,
                "NDP": getattr(watcher_instance, "ng_duration_pulses", 0) if watcher_instance else 0,
                "NDL": getattr(watcher_instance, "ng_duration_percent", 0) if watcher_instance else 0,
                "NEF": getattr(watcher_instance, "ng_encoder_factor", 0) if watcher_instance else 0,
                "EXT": getattr(watcher_instance, "external_reset", 0) if watcher_instance else 0,
                "BUD": getattr(watcher_instance, "baud_rate", SERIAL_BAUDRATE) if watcher_instance else SERIAL_BAUDRATE,
                "DWT": getattr(watcher_instance, "downtime_threshold", 0) if watcher_instance else 0,
            },
            "data": watcher_instance.data if watcher_instance else {},
            "serial_device": {
                "connected": getattr(watcher_instance, "serial_available", False) if watcher_instance else False,
                "port": getattr(watcher_instance, "serial_port", WATCHER_USB) if watcher_instance else WATCHER_USB,
                "baudrate": getattr(watcher_instance, "serial_baudrate", SERIAL_BAUDRATE) if watcher_instance else SERIAL_BAUDRATE,
                "mode": getattr(watcher_instance, "serial_mode", SERIAL_MODE) if watcher_instance else SERIAL_MODE,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"API status error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/status/stream")
async def status_stream():
    """Server-Sent Events stream for real-time status updates."""
    def generate():
        last_data = None
        while True:
            try:
                # Build current status data
                current_data = {
                    "encoder_value": watcher_instance.encoder_value if watcher_instance else 0,
                    "pps": getattr(watcher_instance, "pulses_per_second", 0) if watcher_instance else 0,
                    "ppm": getattr(watcher_instance, "pulses_per_minute", 0) if watcher_instance else 0,
                    "downtime_seconds": getattr(watcher_instance, "downtime_seconds", 0) if watcher_instance else 0,
                    "ok_counter": getattr(watcher_instance, "ok_counter", 0) if watcher_instance else 0,
                    "ng_counter": getattr(watcher_instance, "ng_counter", 0) if watcher_instance else 0,
                    "analog_value": getattr(watcher_instance, "analog_value", 0) if watcher_instance else 0,
                    "power_value": getattr(watcher_instance, "power_value", 0) if watcher_instance else 0,
                    "is_moving": watcher_instance.is_moving if watcher_instance else False,
                    "shipment": watcher_instance.shipment if watcher_instance else "no_shipment",
                    "ejector_queue_length": len(watcher_instance.ejection_queue) if watcher_instance else 0,
                    "ejector_running": watcher_instance.ejector_running if watcher_instance else False,
                    "status": {
                        "U": getattr(watcher_instance, "u_status", False) if watcher_instance else False,
                        "B": getattr(watcher_instance, "b_status", False) if watcher_instance else False,
                        "warning": getattr(watcher_instance, "warning_status", False) if watcher_instance else False,
                        "raw": getattr(watcher_instance, "status_value", 0) if watcher_instance else 0,
                    },
                    "verbose_data": {
                        "OOD": getattr(watcher_instance, "ok_offset_delay", 0) if watcher_instance else 0,
                        "ODP": getattr(watcher_instance, "ok_duration_pulses", 0) if watcher_instance else 0,
                        "ODL": getattr(watcher_instance, "ok_duration_percent", 0) if watcher_instance else 0,
                        "OEF": getattr(watcher_instance, "ok_encoder_factor", 0) if watcher_instance else 0,
                        "NOD": getattr(watcher_instance, "ng_offset_delay", 0) if watcher_instance else 0,
                        "NDP": getattr(watcher_instance, "ng_duration_pulses", 0) if watcher_instance else 0,
                        "NDL": getattr(watcher_instance, "ng_duration_percent", 0) if watcher_instance else 0,
                        "NEF": getattr(watcher_instance, "ng_encoder_factor", 0) if watcher_instance else 0,
                        "EXT": getattr(watcher_instance, "external_reset", 0) if watcher_instance else 0,
                        "BUD": getattr(watcher_instance, "baud_rate", SERIAL_BAUDRATE) if watcher_instance else SERIAL_BAUDRATE,
                        "DWT": getattr(watcher_instance, "downtime_threshold", 0) if watcher_instance else 0,
                    },
                    "serial_device": {
                        "connected": getattr(watcher_instance, "serial_available", False) if watcher_instance else False,
                        "port": getattr(watcher_instance, "serial_port", WATCHER_USB) if watcher_instance else WATCHER_USB,
                        "baudrate": getattr(watcher_instance, "serial_baudrate", SERIAL_BAUDRATE) if watcher_instance else SERIAL_BAUDRATE,
                        "mode": getattr(watcher_instance, "serial_mode", SERIAL_MODE) if watcher_instance else SERIAL_MODE,
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Only send if data changed or every 5 seconds as heartbeat
                data_json = json.dumps(current_data)
                if data_json != last_data:
                    yield f"data: {data_json}\n\n"
                    last_data = data_json

                time.sleep(0.1)  # Check for changes every 100ms
            except Exception as e:
                logger.error(f"SSE status stream error: {e}")
                time.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/video_feed")
async def video_feed():
    """Lightweight MJPEG stream from first available camera with detection overlays."""
    def generate():
        frame_count = 0
        last_log_time = time.time()
        while True:
            try:
                if watcher_instance and watcher_instance.cameras:
                    # Get first available camera
                    cam = next((c for c in watcher_instance.cameras.values() if hasattr(c, 'frame')), None)
                    if cam and hasattr(cam, 'frame') and cam.frame is not None:
                        # Make a copy of the frame to draw on
                        frame = cam.frame.copy()
                        frame_count += 1

                        # Draw bounding boxes from latest detections (if recent) from module globals
                        global latest_detections, latest_detections_timestamp
                        detections = latest_detections
                        timestamp = latest_detections_timestamp

                        # Log periodically
                        if time.time() - last_log_time > 5.0:
                            logger.info(f"[VIDEO_FEED] Frame #{frame_count}, latest_detections={len(detections) if detections else 0}, age={time.time() - timestamp if timestamp else 'N/A'}s")
                            last_log_time = time.time()
                        if detections and timestamp:
                            detection_age = time.time() - timestamp
                            # Only show detections if they're less than 10 seconds old
                            if detection_age < 10.0:
                                for det in detections:
                                    try:
                                        # Get coordinates (YOLO format: xmin, ymin, xmax, ymax)
                                        x1 = int(det.get('xmin', 0))
                                        y1 = int(det.get('ymin', 0))
                                        x2 = int(det.get('xmax', 0))
                                        y2 = int(det.get('ymax', 0))
                                        confidence = det.get('confidence', 0)
                                        name = det.get('name', f"Class {det.get('class', 0)}")

                                        # Draw thick green rectangle
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                                        # Draw label with background
                                        label = f"{name} {confidence:.2f}"
                                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                                        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    except Exception as e:
                                        logger.error(f"Error drawing detection box: {e}")

                        # Encode with lower quality for lightweight streaming
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)  # 10 FPS max
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_detections")
async def video_feed_detections():
    """Video feed showing the LAST PROCESSED IMAGE with bounding boxes (Gradio inference results).

    This shows the actual processed images from raw_images/ directory with detections drawn on them,
    NOT the live camera stream. Updates when new images are processed by the state machine.
    """
    def generate():
        logger.info("[DETECTION_FEED] Stream started - showing processed images with detections")
        last_frame_id = None
        while True:
            try:
                global latest_detections, latest_detections_timestamp, watcher_instance
                detections = latest_detections
                timestamp = latest_detections_timestamp
                latest_frame_id = getattr(watcher_instance, 'latest_frame_id', None) if watcher_instance else None

                # Only update when a new frame has been processed
                if latest_frame_id and latest_frame_id != last_frame_id and detections:
                    last_frame_id = latest_frame_id

                    # Try to load the processed image from raw_images directory
                    frame_path = os.path.join("raw_images", f"{latest_frame_id}.jpg")
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            # Draw all detections with bounding boxes
                            for det in detections:
                                try:
                                    x1 = int(det.get('xmin', 0))
                                    y1 = int(det.get('ymin', 0))
                                    x2 = int(det.get('xmax', 0))
                                    y2 = int(det.get('ymax', 0))
                                    confidence = det.get('confidence', 0)
                                    name = det.get('name', f"Class {det.get('class', 0)}")

                                    # Draw thick green rectangle
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                                    # Draw label with background
                                    label = f"{name} {confidence:.2f}"
                                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                except Exception as e:
                                    logger.error(f"Error drawing detection: {e}")

                            # Encode and yield
                            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                                logger.info(f"[DETECTION_FEED] Displayed frame {latest_frame_id} with {len(latest_detections)} detections")

                time.sleep(0.5)  # 2 FPS - only updates when new images are processed
            except Exception as e:
                logger.error(f"Detection stream error: {e}")
                time.sleep(1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/latest_detection_image")
async def latest_detection_image():
    """Serve the latest image with detection bounding boxes."""
    try:
        import glob
        # Find all _DETECTED.jpg files in raw_images directory (including subdirectories)
        pattern = os.path.join("raw_images", "**", "*_DETECTED.jpg")
        detected_files = glob.glob(pattern, recursive=True)

        if detected_files:
            # Get the most recent file by modification time
            latest_file = max(detected_files, key=os.path.getmtime)

            # Read and return the image
            with open(latest_file, 'rb') as f:
                image_data = f.read()

            return Response(content=image_data, media_type="image/jpeg")
        else:
            # Return a placeholder image or 404
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="No detection images found")
    except Exception as e:
        logger.error(f"Error serving latest detection image: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


def generate_detection_stream():
    """Generator function that yields MJPEG frames from latest detections."""
    import glob
    last_file = None
    last_mtime = 0

    while True:
        try:
            # Find all _DETECTED.jpg files in raw_images directory
            pattern = os.path.join("raw_images", "**", "*_DETECTED.jpg")
            detected_files = glob.glob(pattern, recursive=True)

            if detected_files:
                # Get the most recent file by modification time
                latest_file = max(detected_files, key=os.path.getmtime)
                current_mtime = os.path.getmtime(latest_file)

                # Only send if file has changed or it's a new file
                if latest_file != last_file or current_mtime != last_mtime:
                    # Read the image
                    with open(latest_file, 'rb') as f:
                        image_data = f.read()

                    # Yield the frame in MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')

                    last_file = latest_file
                    last_mtime = current_mtime

            # Wait a bit before checking for new frames (2 times per second)
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in detection stream: {e}")
            time.sleep(1)


@app.get("/detection_stream")
async def detection_stream():
    """Serve a continuous MJPEG stream of detection results."""
    return StreamingResponse(
        generate_detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return JSONResponse(content={
        "ejector": {
            "enabled": EJECTOR_ENABLED,
            "offset": EJECTOR_OFFSET,
            "duration": EJECTOR_DURATION,
            "poll_interval": EJECTOR_POLL_INTERVAL
        },
        "capture": {
            "mode": CAPTURE_MODE,
            "time_between_packages": TIME_BETWEEN_TWO_PACKAGE
        },
        "image_processing": {
            "parent_object_list": PARENT_OBJECT_LIST,
            "remove_raw_image_when_dm_decoded": REMOVE_RAW_IMAGE_WHEN_DM_DECODED,
            "enforce_parent_object": ENFORCE_PARENT_OBJECT
        },
        "datamatrix": {
            "chars_sizes": DM_CHARS_SIZES,
            "confidence_threshold": DM_CONFIDENCE_THRESHOLD,
            "overlap_threshold": DM_OVERLAP_THRESHOLD
        },
        "class_count_check": {
            "enabled": CHECK_CLASS_COUNTS_ENABLED,
            "classes": CHECK_CLASS_COUNTS_CLASSES,
            "confidence": CHECK_CLASS_COUNTS_CONFIDENCE
        },
        "light_control": {
            "status_check_enabled": LIGHT_STATUS_CHECK_ENABLED
        },
        "histogram": {
            "enabled": HISTOGRAM_ENABLED,
            "save_image": HISTOGRAM_SAVE_IMAGE
        },
        "store_annotation": {
            "enabled": STORE_ANNOTATION_ENABLED,
            "postgres_host": POSTGRES_HOST,
            "postgres_port": POSTGRES_PORT,
            "postgres_db": POSTGRES_DB
        },
        "data_file": DATA_FILE,
        "serial": {
            "port": WATCHER_USB,
            "baudrate": SERIAL_BAUDRATE,
            "mode": SERIAL_MODE
        },
        "cameras": {
            "cam_1": CAM_1_PATH,
            "cam_2": CAM_2_PATH,
            "cam_3": CAM_3_PATH,
            "cam_4": CAM_4_PATH
        },
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT
        },
        "commands": USER_COMMANDS
    })

@app.post("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update counter service configuration at runtime.

    Supported keys:
        - ejector_offset: int - Encoder counts from camera to ejector
        - ejector_duration: float - Seconds to run ejector motor
        - ejector_poll_interval: float - Seconds between ejector checks
        - time_between_packages: float - Minimum seconds between captures
        - capture_mode: str - "single" or "multiple"
        - parent_object_list: str - Comma-separated list of parent object names
        - remove_raw_image_when_dm_decoded: bool - Remove raw images after DM decode
        - dm_chars_sizes: str - Comma-separated list of valid DM character lengths
        - dm_confidence_threshold: float - Confidence threshold for DM matching
        - dm_overlap_threshold: float - Overlap threshold for DM detection
        - check_class_counts_enabled: bool - Enable/disable class count checking
        - check_class_counts_classes: str - Comma-separated list of classes to check
        - yolo_url: str - YOLO/Gradio API inference URL
        - gradio_model: str - Gradio model name (e.g., "Data Matrix", "Tire Cord")
        - gradio_confidence: float - Gradio detection confidence threshold (0-1)
        - redis_host: str - Redis server hostname
        - redis_port: int - Redis server port
    """
    global EJECTOR_ENABLED, EJECTOR_OFFSET, EJECTOR_DURATION, EJECTOR_POLL_INTERVAL
    global TIME_BETWEEN_TWO_PACKAGE, CAPTURE_MODE, time_between_two_package, capture_mode
    global PARENT_OBJECT_LIST, parent_object_list, REMOVE_RAW_IMAGE_WHEN_DM_DECODED, remove_raw_image_when_dm_decoded
    global DM_CHARS_SIZES, DM_CONFIDENCE_THRESHOLD, DM_OVERLAP_THRESHOLD
    global CHECK_CLASS_COUNTS_ENABLED, CHECK_CLASS_COUNTS_CLASSES, CHECK_CLASS_COUNTS_CONFIDENCE
    global LIGHT_STATUS_CHECK_ENABLED
    global HISTOGRAM_ENABLED, HISTOGRAM_SAVE_IMAGE, STORE_ANNOTATION_ENABLED
    global ENFORCE_PARENT_OBJECT

    updated = {}

    try:
        # Ejector configuration
        if "ejector_enabled" in config:
            value = config["ejector_enabled"]
            EJECTOR_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["ejector_enabled"] = EJECTOR_ENABLED
            logger.info(f"Updated EJECTOR_ENABLED to {EJECTOR_ENABLED}")

        if "ejector_offset" in config:
            EJECTOR_OFFSET = int(config["ejector_offset"])
            updated["ejector_offset"] = EJECTOR_OFFSET
            logger.info(f"Updated EJECTOR_OFFSET to {EJECTOR_OFFSET}")

        if "ejector_duration" in config:
            EJECTOR_DURATION = float(config["ejector_duration"])
            updated["ejector_duration"] = EJECTOR_DURATION
            logger.info(f"Updated EJECTOR_DURATION to {EJECTOR_DURATION}")

        if "ejector_poll_interval" in config:
            EJECTOR_POLL_INTERVAL = float(config["ejector_poll_interval"])
            updated["ejector_poll_interval"] = EJECTOR_POLL_INTERVAL
            logger.info(f"Updated EJECTOR_POLL_INTERVAL to {EJECTOR_POLL_INTERVAL}")

        # Capture configuration
        if "time_between_packages" in config:
            TIME_BETWEEN_TWO_PACKAGE = float(config["time_between_packages"])
            time_between_two_package = TIME_BETWEEN_TWO_PACKAGE
            updated["time_between_packages"] = TIME_BETWEEN_TWO_PACKAGE
            logger.info(f"Updated TIME_BETWEEN_TWO_PACKAGE to {TIME_BETWEEN_TWO_PACKAGE}")

        if "capture_mode" in config:
            mode = config["capture_mode"]
            if mode in ["single", "multiple"]:
                CAPTURE_MODE = mode
                capture_mode = mode
                updated["capture_mode"] = CAPTURE_MODE
                logger.info(f"Updated CAPTURE_MODE to {CAPTURE_MODE}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid capture_mode: {mode}. Must be 'single' or 'multiple'")

        # Image processing configuration
        if "parent_object_list" in config:
            value = config["parent_object_list"]
            if isinstance(value, str):
                PARENT_OBJECT_LIST = [x.strip() for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                PARENT_OBJECT_LIST = value
            parent_object_list = PARENT_OBJECT_LIST
            updated["parent_object_list"] = PARENT_OBJECT_LIST
            logger.info(f"Updated PARENT_OBJECT_LIST to {PARENT_OBJECT_LIST}")

        if "remove_raw_image_when_dm_decoded" in config:
            value = config["remove_raw_image_when_dm_decoded"]
            REMOVE_RAW_IMAGE_WHEN_DM_DECODED = str(value).lower() in ("true", "1", "yes")
            remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
            updated["remove_raw_image_when_dm_decoded"] = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
            logger.info(f"Updated REMOVE_RAW_IMAGE_WHEN_DM_DECODED to {REMOVE_RAW_IMAGE_WHEN_DM_DECODED}")

        # DataMatrix configuration
        if "dm_chars_sizes" in config:
            value = config["dm_chars_sizes"]
            if isinstance(value, str):
                DM_CHARS_SIZES = [int(x.strip()) for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                DM_CHARS_SIZES = [int(x) for x in value]
            updated["dm_chars_sizes"] = DM_CHARS_SIZES
            logger.info(f"Updated DM_CHARS_SIZES to {DM_CHARS_SIZES}")

        if "dm_confidence_threshold" in config:
            DM_CONFIDENCE_THRESHOLD = float(config["dm_confidence_threshold"])
            updated["dm_confidence_threshold"] = DM_CONFIDENCE_THRESHOLD
            logger.info(f"Updated DM_CONFIDENCE_THRESHOLD to {DM_CONFIDENCE_THRESHOLD}")

        if "dm_overlap_threshold" in config:
            DM_OVERLAP_THRESHOLD = float(config["dm_overlap_threshold"])
            updated["dm_overlap_threshold"] = DM_OVERLAP_THRESHOLD
            logger.info(f"Updated DM_OVERLAP_THRESHOLD to {DM_OVERLAP_THRESHOLD}")

        # Class count checking configuration
        if "check_class_counts_enabled" in config:
            value = config["check_class_counts_enabled"]
            CHECK_CLASS_COUNTS_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["check_class_counts_enabled"] = CHECK_CLASS_COUNTS_ENABLED
            logger.info(f"Updated CHECK_CLASS_COUNTS_ENABLED to {CHECK_CLASS_COUNTS_ENABLED}")

        if "check_class_counts_classes" in config:
            value = config["check_class_counts_classes"]
            if isinstance(value, str):
                CHECK_CLASS_COUNTS_CLASSES = [x.strip() for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                CHECK_CLASS_COUNTS_CLASSES = value
            updated["check_class_counts_classes"] = CHECK_CLASS_COUNTS_CLASSES
            logger.info(f"Updated CHECK_CLASS_COUNTS_CLASSES to {CHECK_CLASS_COUNTS_CLASSES}")

        if "check_class_counts_confidence" in config:
            value = config["check_class_counts_confidence"]
            CHECK_CLASS_COUNTS_CONFIDENCE = float(value)
            updated["check_class_counts_confidence"] = CHECK_CLASS_COUNTS_CONFIDENCE
            logger.info(f"Updated CHECK_CLASS_COUNTS_CONFIDENCE to {CHECK_CLASS_COUNTS_CONFIDENCE}")

        # Light control configuration
        if "light_status_check_enabled" in config:
            value = config["light_status_check_enabled"]
            LIGHT_STATUS_CHECK_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["light_status_check_enabled"] = LIGHT_STATUS_CHECK_ENABLED
            logger.info(f"Updated LIGHT_STATUS_CHECK_ENABLED to {LIGHT_STATUS_CHECK_ENABLED}")

        # Histogram configuration
        if "histogram_enabled" in config:
            value = config["histogram_enabled"]
            HISTOGRAM_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["histogram_enabled"] = HISTOGRAM_ENABLED
            logger.info(f"Updated HISTOGRAM_ENABLED to {HISTOGRAM_ENABLED}")

        if "histogram_save_image" in config:
            value = config["histogram_save_image"]
            HISTOGRAM_SAVE_IMAGE = str(value).lower() in ("true", "1", "yes")
            updated["histogram_save_image"] = HISTOGRAM_SAVE_IMAGE
            logger.info(f"Updated HISTOGRAM_SAVE_IMAGE to {HISTOGRAM_SAVE_IMAGE}")

        # Store annotation configuration
        if "store_annotation_enabled" in config:
            value = config["store_annotation_enabled"]
            STORE_ANNOTATION_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["store_annotation_enabled"] = STORE_ANNOTATION_ENABLED
            logger.info(f"Updated STORE_ANNOTATION_ENABLED to {STORE_ANNOTATION_ENABLED}")

        # Parent object enforcement configuration
        if "enforce_parent_object" in config:
            value = config["enforce_parent_object"]
            ENFORCE_PARENT_OBJECT = str(value).lower() in ("true", "1", "yes")
            updated["enforce_parent_object"] = ENFORCE_PARENT_OBJECT
            logger.info(f"Updated ENFORCE_PARENT_OBJECT to {ENFORCE_PARENT_OBJECT}")

        # Infrastructure configuration (Gradio/YOLO API)
        if "yolo_url" in config:
            global YOLO_INFERENCE_URL
            YOLO_INFERENCE_URL = str(config["yolo_url"])
            updated["yolo_url"] = YOLO_INFERENCE_URL
            logger.info(f"Updated YOLO_INFERENCE_URL to {YOLO_INFERENCE_URL}")

        if "gradio_model" in config:
            global GRADIO_MODEL
            GRADIO_MODEL = str(config["gradio_model"])
            updated["gradio_model"] = GRADIO_MODEL
            logger.info(f"Updated GRADIO_MODEL to {GRADIO_MODEL}")

        if "gradio_confidence" in config:
            global GRADIO_CONFIDENCE_THRESHOLD
            GRADIO_CONFIDENCE_THRESHOLD = float(config["gradio_confidence"])
            updated["gradio_confidence"] = GRADIO_CONFIDENCE_THRESHOLD
            logger.info(f"Updated GRADIO_CONFIDENCE_THRESHOLD to {GRADIO_CONFIDENCE_THRESHOLD}")

        if "redis_host" in config:
            global REDIS_HOST
            REDIS_HOST = str(config["redis_host"])
            updated["redis_host"] = REDIS_HOST
            logger.info(f"Updated REDIS_HOST to {REDIS_HOST}")

        if "redis_port" in config:
            global REDIS_PORT
            REDIS_PORT = int(config["redis_port"])
            updated["redis_port"] = REDIS_PORT
            logger.info(f"Updated REDIS_PORT to {REDIS_PORT}")

        # Serial mode configuration
        if "serial_mode" in config:
            global SERIAL_MODE
            mode = str(config["serial_mode"])
            if mode in ["new", "legacy"]:
                SERIAL_MODE = mode
                # Update watcher if available
                if watcher_instance:
                    watcher_instance.serial_mode = SERIAL_MODE
                updated["serial_mode"] = SERIAL_MODE
                logger.info(f"Updated SERIAL_MODE to {SERIAL_MODE}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid serial_mode: {mode}. Must be 'new' or 'legacy'")

        # Shipment ID
        if "shipment" in config:
            shipment_id = str(config["shipment"])
            # Store in Redis and update watcher instance
            if watcher_instance and watcher_instance.redis_connection:
                watcher_instance.redis_connection.redis_connection.set("shipment", shipment_id)
                watcher_instance.shipment = shipment_id
                updated["shipment"] = shipment_id
                logger.info(f"Updated shipment ID to {shipment_id}")
            elif watcher_instance:
                # No Redis, just update watcher instance
                watcher_instance.shipment = shipment_id
                updated["shipment"] = shipment_id
                logger.info(f"Updated shipment ID to {shipment_id} (no Redis)")
            else:
                logger.warning("Watcher not available, cannot update shipment ID")

        if not updated:
            raise HTTPException(status_code=400, detail="No valid configuration keys provided")

        return JSONResponse(content={
            "status": "ok",
            "updated": updated,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/data-file")
async def get_data_file():
    """Get the contents of the DATA_FILE (prepared_query_data)."""
    try:
        with open(DATA_FILE, "r") as f:
            content = f.read()
        return JSONResponse(content={
            "file_path": DATA_FILE,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read data file: {str(e)}")

@app.post("/api/data-file")
async def update_data_file(data: Dict[str, Any]):
    """Update the DATA_FILE content and reload prepared_query_data.

    Expected body:
        - content: str - The new JSON content for the data file
    """
    global prepared_query_data

    if "content" not in data:
        raise HTTPException(status_code=400, detail="Missing 'content' field")

    try:
        # Validate JSON
        new_data = json.loads(data["content"])

        # Backup the old file
        backup_path = f"{DATA_FILE}.backup"
        try:
            with open(DATA_FILE, "r") as f:
                old_content = f.read()
            with open(backup_path, "w") as f:
                f.write(old_content)
        except FileNotFoundError:
            pass  # No existing file to backup

        # Write new content
        with open(DATA_FILE, "w") as f:
            f.write(data["content"])

        # Reload the prepared_query_data (handle both formats)
        if isinstance(new_data, list):
            prepared_query_data = new_data
        elif isinstance(new_data, dict) and "prepared_query_data" in new_data:
            prepared_query_data = new_data["prepared_query_data"]
        else:
            prepared_query_data = new_data
        logger.info(f"Updated DATA_FILE ({DATA_FILE}) with {len(prepared_query_data)} entries")

        return JSONResponse(content={
            "status": "ok",
            "file_path": DATA_FILE,
            "entries_count": len(prepared_query_data),
            "backup_path": backup_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating data file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update data file: {str(e)}")

# ===== CAMERA CONTROL ENDPOINTS =====
# Direct control of counter service cameras (cam_1 through cam_4)

@app.get("/api/cameras")
async def get_cameras_status():
    """Get status of all cameras (dynamically detected)."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    cameras = []
    # Use dynamic cameras dict
    for cam_id, cam in watcher_instance.cameras.items():
        # Get camera path from stored paths
        cam_path = watcher_instance.camera_paths[cam_id - 1] if cam_id <= len(watcher_instance.camera_paths) else "unknown"

        # Get camera metadata if available
        metadata = {}
        if hasattr(watcher_instance, 'camera_metadata') and cam_id in watcher_instance.camera_metadata:
            metadata = watcher_instance.camera_metadata[cam_id]

        cam_info = {
            "id": cam_id,
            "path": cam_path,
            "name": metadata.get("name", f"Camera {cam_id}"),
            "type": metadata.get("type", "usb"),
            "connected": cam is not None and getattr(cam, 'success', False),
            "running": cam is not None and not getattr(cam, 'stop', True),
        }

        # Add IP camera specific info
        if metadata.get("type") == "ip":
            cam_info["ip"] = metadata.get("ip")
            cam_info["camera_path"] = metadata.get("path")

        if cam is not None and hasattr(cam, 'camera'):
            try:
                cam_info["config"] = {
                    "fps": int(cam.camera.get(cv2.CAP_PROP_FPS)),
                    "width": int(cam.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cam.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "exposure": int(cam.camera.get(cv2.CAP_PROP_EXPOSURE)),
                    "gain": int(cam.camera.get(cv2.CAP_PROP_GAIN)),
                    "brightness": int(cam.camera.get(cv2.CAP_PROP_BRIGHTNESS)),
                    "contrast": int(cam.camera.get(cv2.CAP_PROP_CONTRAST)),
                    "saturation": int(cam.camera.get(cv2.CAP_PROP_SATURATION)),
                    "roi_enabled": getattr(cam, 'roi_enabled', False),
                    "roi_xmin": getattr(cam, 'roi_xmin', 0),
                    "roi_ymin": getattr(cam, 'roi_ymin', 0),
                    "roi_xmax": getattr(cam, 'roi_xmax', 1280),
                    "roi_ymax": getattr(cam, 'roi_ymax', 720),
                }
            except:
                cam_info["config"] = {}
        cameras.append(cam_info)

    return JSONResponse(content=cameras)

@app.post("/api/camera/{camera_id}/restart")
async def restart_camera(camera_id: int):
    """Restart a specific camera."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher_instance.cameras.get(camera_id)
    if cam is None:
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    try:
        cam.restart_camera()
        return JSONResponse(content={"success": True, "message": f"Camera {camera_id} restarted"})
    except Exception as e:
        logger.error(f"Error restarting camera {camera_id}: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/camera/{camera_id}/config")
async def update_camera_config(camera_id: int, request: Request):
    """Update camera configuration."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher_instance.cameras.get(camera_id)
    if cam is None or not hasattr(cam, 'camera'):
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    try:
        body = await request.json()

        # Apply configuration
        prop_map = {
            'fps': cv2.CAP_PROP_FPS,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
        }

        updated = []
        for key, value in body.items():
            if key in prop_map:
                cam.camera.set(prop_map[key], value)
                updated.append(key)

        # Handle ROI settings (stored on CameraBuffer object, not OpenCV)
        roi_props = ['roi_enabled', 'roi_xmin', 'roi_ymin', 'roi_xmax', 'roi_ymax']
        for key in roi_props:
            if key in body:
                setattr(cam, key, body[key])
                updated.append(key)

        return JSONResponse(content={"success": True, "updated": updated})
    except Exception as e:
        logger.error(f"Error updating camera {camera_id} config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/camera/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: int):
    """Get current frame from a camera as JPEG."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher_instance.cameras.get(camera_id)
    if cam is None:
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    try:
        frame = cam.read()
        if frame is None:
            return JSONResponse(content={"error": "No frame available"}, status_code=404)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error getting snapshot from camera {camera_id}: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/cameras/config/save")
async def save_all_config():
    """Save all service configurations (cameras + settings) to data file."""
    global EJECTOR_ENABLED, EJECTOR_OFFSET, EJECTOR_DURATION, EJECTOR_POLL_INTERVAL
    global CAPTURE_MODE, TIME_BETWEEN_TWO_PACKAGE, REMOVE_RAW_IMAGE_WHEN_DM_DECODED
    global PARENT_OBJECT_LIST, HISTOGRAM_ENABLED, HISTOGRAM_SAVE_IMAGE
    global CHECK_CLASS_COUNTS_ENABLED, CHECK_CLASS_COUNTS_CLASSES, CHECK_CLASS_COUNTS_CONFIDENCE
    global DM_CHARS_SIZES, DM_CONFIDENCE_THRESHOLD, DM_OVERLAP_THRESHOLD
    global STORE_ANNOTATION_ENABLED

    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        # Get AI configuration from Redis (if available)
        ai_config = {}
        if watcher_instance.redis_connection:
            r = watcher_instance.redis_connection.redis_connection
            ai_model = r.get("ai_model")
            ai_api_key = r.get("ai_api_key")
            if ai_model or ai_api_key:
                ai_config = {
                    "model": ai_model.decode('utf-8') if isinstance(ai_model, bytes) else (ai_model or "claude"),
                    "api_key": ai_api_key.decode('utf-8') if isinstance(ai_api_key, bytes) else (ai_api_key or "")
                }

        config = {
            "cameras": {},
            "infrastructure": {
                "redis_host": REDIS_HOST,
                "redis_port": REDIS_PORT,
                "serial_port": WATCHER_USB,
                "serial_baudrate": SERIAL_BAUDRATE,
                "serial_mode": SERIAL_MODE,
                "postgres_host": POSTGRES_HOST,
                "postgres_port": POSTGRES_PORT,
                "postgres_db": POSTGRES_DB,
                "postgres_user": POSTGRES_USER
            },
            "inference": {
                "current_module": "gradio_hf",  # Currently active module
                "modules": {
                    "gradio_hf": {
                        "name": "Gradio HuggingFace",
                        "inference_url": "https://smartfalcon-ai-industrial-defect-detection.hf.space",
                        "inference_model": "Data Matrix",
                        "inference_min_confidence": 0.25,
                        "type": "gradio"
                    },
                    "local_yolo": {
                        "name": "Local YOLO",
                        "inference_url": "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/",
                        "inference_model": "N/A",
                        "inference_min_confidence": 0.25,
                        "type": "yolo"
                    }
                }
            },
            "ai": ai_config if ai_config else {
                "model": "claude",
                "api_key": ""
            },
            "ejector": {
                "enabled": EJECTOR_ENABLED,
                "offset": EJECTOR_OFFSET,
                "duration": EJECTOR_DURATION,
                "poll_interval": EJECTOR_POLL_INTERVAL
            },
            "capture": {
                "mode": CAPTURE_MODE,
                "time_between_packages": TIME_BETWEEN_TWO_PACKAGE
            },
            "image_processing": {
                "remove_raw_image_when_dm_decoded": REMOVE_RAW_IMAGE_WHEN_DM_DECODED,
                "parent_object_list": PARENT_OBJECT_LIST
            },
            "histogram": {
                "enabled": HISTOGRAM_ENABLED,
                "save_image": HISTOGRAM_SAVE_IMAGE
            },
            "class_count_check": {
                "enabled": CHECK_CLASS_COUNTS_ENABLED,
                "classes": CHECK_CLASS_COUNTS_CLASSES,
                "confidence": CHECK_CLASS_COUNTS_CONFIDENCE
            },
            "datamatrix": {
                "chars_sizes": DM_CHARS_SIZES,
                "confidence_threshold": DM_CONFIDENCE_THRESHOLD,
                "overlap_threshold": DM_OVERLAP_THRESHOLD
            },
            "store_annotation": {
                "enabled": STORE_ANNOTATION_ENABLED
            }
        }

        # Add camera configs dynamically
        camera_metadata = getattr(watcher_instance, 'camera_metadata', {})
        for cam_id, cam in watcher_instance.cameras.items():
            cam_config = get_camera_config_for_save(cam, cam_id, camera_metadata)
            if cam_config:
                config["cameras"][str(cam_id)] = cam_config

        # Add states configuration
        if state_manager:
            config["states"] = {name: s.to_dict() for name, s in state_manager.states.items()}
            config["current_state_name"] = state_manager.current_state.name if state_manager.current_state else "default"

        # Add pipeline configuration
        if pipeline_manager:
            config["pipeline_config"] = pipeline_manager.to_config()

        if save_service_config(config):
            return JSONResponse(content={
                "success": True,
                "message": f"Configuration saved to {DATA_FILE}",
                "cameras_saved": len(config["cameras"]),
                "states_saved": len(config.get("states", {})),
                "pipelines_saved": len(config.get("pipeline_config", {}).get("pipelines", {}))
            })
        else:
            return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/cameras/config/load")
async def load_all_config():
    """Load all service configurations from data file and apply them."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        config = load_service_config()
        if not config:
            return JSONResponse(content={"error": "No saved configuration found"}, status_code=404)

        # Use the helper function to apply all settings
        settings_applied, cameras_loaded = apply_config_settings(config, watcher_instance)

        # Apply states configuration
        states_loaded = 0
        if "states" in config and state_manager:
            for state_name, state_data in config["states"].items():
                state = State.from_dict(state_data)
                state_manager.add_state(state)
                states_loaded += 1

            # Set current state if specified
            current_state_name = config.get("current_state_name", "default")
            if current_state_name in state_manager.states:
                state_manager.set_current_state(current_state_name)

        return JSONResponse(content={
            "success": True,
            "message": f"Configuration loaded from {DATA_FILE}",
            "cameras_loaded": cameras_loaded,
            "states_loaded": states_loaded,
            "saved_at": config.get("saved_at", "unknown")
        })
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/cameras/config")
async def get_saved_config():
    """Get the saved service configuration from data file."""
    try:
        config = load_service_config()
        if not config:
            return JSONResponse(content={"exists": False, "config": None})
        return JSONResponse(content={"exists": True, "config": config})
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/inference")
async def get_inference_config():
    """Get inference module configuration (fast endpoint for UI)."""
    try:
        config = load_service_config()
        if not config or "inference" not in config:
            return JSONResponse(content={
                "current_module": "gradio_hf",
                "modules": {}
            })
        return JSONResponse(content=config["inference"])
    except Exception as e:
        logger.error(f"Error reading inference config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ===== PIPELINE MANAGEMENT ENDPOINTS =====

@app.get("/api/pipelines")
async def get_pipelines():
    """Get all pipelines and models configuration."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content=pipeline_manager.get_status())


@app.get("/api/pipelines/current")
async def get_current_pipeline():
    """Get the currently active pipeline."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pipeline_manager.current_pipeline:
        return JSONResponse(content={
            "pipeline": pipeline_manager.current_pipeline.to_dict(),
            "current_model": pipeline_manager.get_current_model().to_dict() if pipeline_manager.get_current_model() else None
        })
    return JSONResponse(content={"pipeline": None, "current_model": None})


@app.post("/api/pipelines/activate/{pipeline_name}")
async def activate_pipeline(pipeline_name: str):
    """Set the active pipeline by name."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pipeline_manager.set_current_pipeline(pipeline_name):
        return JSONResponse(content={
            "success": True,
            "message": f"Activated pipeline: {pipeline_name}",
            "current_model": pipeline_manager.get_current_model().to_dict() if pipeline_manager.get_current_model() else None
        })
    return JSONResponse(content={"error": f"Pipeline not found: {pipeline_name}"}, status_code=404)


@app.post("/api/pipelines")
async def create_or_update_pipeline(request: Request):
    """Create or update a pipeline."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = await request.json()
        pipeline = Pipeline.from_dict(body)

        if pipeline_manager.add_pipeline(pipeline):
            return JSONResponse(content={
                "success": True,
                "message": f"Pipeline '{pipeline.name}' created/updated",
                "pipeline": pipeline.to_dict()
            })
        return JSONResponse(content={"error": "Failed to add pipeline"}, status_code=500)
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.delete("/api/pipelines/{pipeline_name}")
async def delete_pipeline(pipeline_name: str):
    """Delete a pipeline."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pipeline_manager.remove_pipeline(pipeline_name):
        return JSONResponse(content={"success": True, "message": f"Pipeline '{pipeline_name}' deleted"})
    return JSONResponse(content={"error": f"Cannot delete pipeline: {pipeline_name}"}, status_code=400)


@app.get("/api/models")
async def get_models():
    """Get all available inference models."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content={
        "models": {mid: m.to_dict() for mid, m in pipeline_manager.models.items()}
    })


@app.post("/api/models")
async def create_or_update_model(request: Request):
    """Create or update an inference model."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = await request.json()
        model_id = body.get("model_id")
        if not model_id:
            return JSONResponse(content={"error": "model_id is required"}, status_code=400)

        model = InferenceModel.from_dict(body)

        if pipeline_manager.add_model(model_id, model):
            return JSONResponse(content={
                "success": True,
                "message": f"Model '{model_id}' created/updated",
                "model": model.to_dict()
            })
        return JSONResponse(content={"error": "Failed to add model"}, status_code=500)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete an inference model."""
    global pipeline_manager
    if pipeline_manager is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pipeline_manager.remove_model(model_id):
        return JSONResponse(content={"success": True, "message": f"Model '{model_id}' deleted"})
    return JSONResponse(content={"error": f"Cannot delete model: {model_id}"}, status_code=400)


@app.get("/api/gradio/models")
async def get_gradio_models(url: str):
    """Fetch available models from a Gradio API endpoint."""
    try:
        # Try to fetch models from Gradio API
        # Most Gradio apps expose model list through /info endpoint or as part of the main interface
        response = requests.get(f"{url.rstrip('/')}/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            # Extract model names if available
            if "models" in info:
                return JSONResponse(content={"models": info["models"]})

        # Fallback: Try to get from config endpoint
        response = requests.get(f"{url.rstrip('/')}/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            # Look for model dropdown in components
            models = []
            if "components" in config:
                for comp in config["components"]:
                    if comp.get("type") == "dropdown" and "choices" in comp:
                        # Assume first dropdown with choices is the model selector
                        models = comp["choices"]
                        break
            if models:
                return JSONResponse(content={"models": models})

        # If no specific endpoint works, return common default models
        return JSONResponse(content={
            "models": ["Data Matrix", "N/A"],
            "note": "Could not fetch from Gradio API, showing defaults"
        })
    except Exception as e:
        logger.error(f"Error fetching Gradio models from {url}: {e}")
        return JSONResponse(content={
            "models": ["Data Matrix", "N/A"],
            "error": str(e)
        }, status_code=500)

@app.post("/api/cameras/config/upload")
async def upload_config(request: Request):
    """Upload and apply a service configuration (for restoring from backup)."""
    if watcher_instance is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        body = await request.json()
        config = body.get("config")

        if not config:
            return JSONResponse(content={"error": "Invalid configuration format"}, status_code=400)

        # Save the uploaded config
        if not save_service_config(config):
            return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)

        # Use the helper function to apply all settings
        settings_applied, cameras_loaded = apply_config_settings(config, watcher_instance)

        return JSONResponse(content={
            "success": True,
            "message": "Configuration uploaded and applied",
            "cameras_loaded": cameras_loaded
        })
    except Exception as e:
        logger.error(f"Error uploading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# =============================================================================
# CAMERA DISCOVERY API ENDPOINTS
# =============================================================================

class CameraDiscoveryRequest(BaseModel):
    subnet: str = "192.168.0"

@app.post("/api/cameras/discover")
async def discover_cameras(request: CameraDiscoveryRequest):
    """Quick scan to discover IP cameras on the network (no credentials needed).

    This scans for devices with camera ports open and returns all possible camera paths.
    Users can then test each path with their own credentials.
    """
    try:
        logger.info(f"Starting quick camera discovery on subnet {request.subnet}.0/24")

        # Perform port scan (no authentication needed)
        discovered_devices = scan_network_for_camera_devices(subnet=request.subnet)

        # Group by IP address to avoid duplicates
        cameras_by_ip = {}
        for device in discovered_devices:
            ip = device['ip']
            if ip not in cameras_by_ip:
                cameras_by_ip[ip] = {
                    "ip": ip,
                    "port": device['port'],
                    "protocol": device['protocol'],
                    "paths": []
                }
            cameras_by_ip[ip]['paths'].append({
                "path": device['path'],
                "url": device['url']
            })

        cameras = list(cameras_by_ip.values())
        logger.info(f"Discovery complete. Found {len(cameras)} potential camera device(s).")

        return JSONResponse(content={
            "success": True,
            "cameras": cameras,
            "count": len(cameras),
            "subnet": request.subnet
        })

    except Exception as e:
        logger.error(f"Error during camera discovery: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "cameras": []
        }, status_code=500)

@app.post("/api/cameras/test")
async def test_camera(request: Request):
    """Test a camera URL with credentials and return a snapshot."""
    try:
        body = await request.json()
        url = body.get("url")
        username = body.get("username", "admin")
        password = body.get("password", "")

        if not url:
            return JSONResponse(content={"success": False, "error": "URL is required"}, status_code=400)

        # Build authenticated URL if credentials provided
        if username and password:
            # Check if URL already has credentials
            if "@" in url:
                # Replace existing credentials
                protocol = url.split("://")[0]
                rest = url.split("@")[-1]
                test_url = f"{protocol}://{username}:{password}@{rest}"
            else:
                # Add credentials
                protocol = url.split("://")[0]
                rest = url.split("://")[-1]
                test_url = f"{protocol}://{username}:{password}@{rest}"
        else:
            test_url = url

        # Test the camera
        cap = cv2.VideoCapture(test_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                import base64
                img_str = base64.b64encode(buffer).decode()

                return JSONResponse(content={
                    "success": True,
                    "message": "Camera test successful",
                    "image": f"data:image/jpeg;base64,{img_str}",
                    "resolution": {"width": frame.shape[1], "height": frame.shape[0]},
                    "authenticated_url": test_url.split("@")[-1] if "@" in test_url else test_url,  # URL without credentials (for display)
                    "full_url": test_url  # Full URL with credentials (for saving)
                })

        return JSONResponse(content={
            "success": False,
            "error": "Failed to capture frame from camera"
        }, status_code=400)

    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/cameras/save")
async def save_camera(request: Request):
    """Save a discovered camera to the configuration."""
    try:
        body = await request.json()
        camera_name = body.get("name", "Unnamed Camera")
        camera_url = body.get("url")
        camera_ip = body.get("ip")
        camera_path = body.get("path")
        resolution = body.get("resolution", {})

        if not camera_url:
            return JSONResponse(content={"success": False, "error": "Camera URL is required"}, status_code=400)

        logger.info(f"Saving camera: {camera_name} ({camera_ip}) - {camera_url}")

        # Load existing service config or create new one
        config = load_service_config()
        if not config:
            config = {}

        # Ensure cameras dict exists
        if "cameras" not in config:
            config["cameras"] = {}

        # Find next available camera ID or check if camera already exists
        existing_id = None
        for cam_id, cam_config in config["cameras"].items():
            if cam_config.get("source") == camera_url:
                existing_id = cam_id
                break

        # Determine camera ID
        if existing_id:
            camera_id = existing_id
        else:
            # Find next available ID
            existing_ids = [int(k) for k in config["cameras"].keys() if k.isdigit()]
            camera_id = str(max(existing_ids) + 1) if existing_ids else "1"

        # Prepare camera entry
        camera_entry = {
            "name": camera_name,
            "source": camera_url,
            "type": "ip",
            "ip": camera_ip,
            "path": camera_path,
            "roi_enabled": False,
            "roi_xmin": 0,
            "roi_ymin": 0,
            "roi_xmax": resolution.get("width", 1920),
            "roi_ymax": resolution.get("height", 1080),
            "exposure": 0,
            "gain": 0,
            "brightness": 0,
            "contrast": 0,
            "saturation": 0,
            "fps": 25,
            "enabled": True
        }

        # Save camera
        config["cameras"][camera_id] = camera_entry
        if existing_id:
            logger.info(f"Updated existing camera ID {camera_id}")
        else:
            logger.info(f"Added new camera ID {camera_id}, total cameras: {len(config['cameras'])}")

        # Save configuration
        if save_service_config(config):
            return JSONResponse(content={
                "success": True,
                "message": f"Camera '{camera_name}' saved successfully as Camera {camera_id}",
                "camera": camera_entry,
                "camera_id": camera_id,
                "total_cameras": len(config["cameras"])
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": "Failed to save configuration"
            }, status_code=500)

    except Exception as e:
        logger.error(f"Error saving camera: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

# =============================================================================
# STATE MANAGEMENT API ENDPOINTS
# =============================================================================

@app.get("/api/states")
async def get_states():
    """Get all configured states and current state status."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)
    return JSONResponse(content=state_manager.get_status())


@app.get("/api/states/{state_name}")
async def get_state(state_name: str):
    """Get a specific state configuration."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    if state_name not in state_manager.states:
        return JSONResponse(content={"error": f"State '{state_name}' not found"}, status_code=404)

    return JSONResponse(content=state_manager.states[state_name].to_dict())


@app.post("/api/states")
async def create_or_update_state(request: Request):
    """Create or update a state configuration.

    Request body example:
    {
        "name": "uplight_capture",
        "active_cameras": [1, 2],
        "delay": 0.1,
        "light_mode": "U_ON_B_OFF",
        "encoder_threshold": 100,
        "analog_threshold": -1,
        "enabled": true
    }
    """
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        data = await request.json()

        if "name" not in data:
            return JSONResponse(content={"error": "State name is required"}, status_code=400)

        state = State.from_dict(data)
        success = state_manager.add_state(state)

        if success:
            return JSONResponse(content={
                "success": True,
                "state": state.to_dict(),
                "message": f"State '{state.name}' created/updated"
            })
        else:
            return JSONResponse(content={"error": "Failed to add state"}, status_code=500)

    except Exception as e:
        logger.error(f"Error creating state: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/api/states/{state_name}")
async def delete_state(state_name: str):
    """Delete a state configuration."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    if state_name == "default":
        return JSONResponse(content={"error": "Cannot delete default state"}, status_code=400)

    success = state_manager.remove_state(state_name)
    if success:
        return JSONResponse(content={"success": True, "message": f"State '{state_name}' deleted"})
    else:
        return JSONResponse(content={"error": f"State '{state_name}' not found"}, status_code=404)


@app.post("/api/states/{state_name}/activate")
async def activate_state(state_name: str):
    """Activate a specific state as the current state."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    success = state_manager.set_current_state(state_name)
    if success:
        return JSONResponse(content={
            "success": True,
            "message": f"State '{state_name}' activated",
            "current_state": state_manager.current_state.to_dict() if state_manager.current_state else None
        })
    else:
        return JSONResponse(content={"error": f"Failed to activate state '{state_name}'"}, status_code=400)


@app.post("/api/states/trigger-capture")
async def trigger_capture():
    """Manually trigger a capture cycle."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    success = state_manager.trigger_capture()
    if success:
        return JSONResponse(content={
            "success": True,
            "message": "Capture triggered",
            "capture_count": state_manager.capture_count
        })
    else:
        return JSONResponse(content={
            "error": "Failed to trigger capture (capture already in progress?)",
            "capture_state": state_manager.capture_state.value
        }, status_code=400)


@app.post("/api/states/save")
async def save_states():
    """Save all states to the main config file (.env.prepared)."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        # Load existing config and update states section
        config = load_service_config() or {}
        config["states"] = {name: s.to_dict() for name, s in state_manager.states.items()}
        config["current_state_name"] = state_manager.current_state.name if state_manager.current_state else "default"

        if save_service_config(config):
            return JSONResponse(content={
                "success": True,
                "message": f"States saved to {DATA_FILE}",
                "states_saved": len(config["states"])
            })
        else:
            return JSONResponse(content={"error": "Failed to save states"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving states: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/states/load")
async def load_states():
    """Load states from the main config file (.env.prepared)."""
    if state_manager is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        config = load_service_config()
        if not config or "states" not in config:
            return JSONResponse(content={"error": "No saved states found in config"}, status_code=404)

        states_loaded = 0
        for state_name, state_data in config["states"].items():
            state = State.from_dict(state_data)
            state_manager.add_state(state)
            states_loaded += 1

        # Set current state if specified
        current_state_name = config.get("current_state_name", "default")
        if current_state_name in state_manager.states:
            state_manager.set_current_state(current_state_name)

        return JSONResponse(content={
            "success": True,
            "message": f"States loaded from {DATA_FILE}",
            "states_count": states_loaded
        })
    except Exception as e:
        logger.error(f"Error loading states: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/save_service_config")
async def api_save_service_config():
    """Save current service configuration to file."""
    try:
        config = load_service_config() or {}
        if save_service_config(config):
            return JSONResponse(content={"success": True, "message": "Service configuration saved"})
        else:
            return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving service config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/save_data_file")
async def api_save_data_file():
    """Save current data file (same as service config)."""
    try:
        config = load_service_config() or {}
        if save_service_config(config):
            return JSONResponse(content={"success": True, "message": "Data file saved"})
        else:
            return JSONResponse(content={"error": "Failed to save data file"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving data file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/load_service_config")
async def api_load_service_config():
    """Load service configuration from file (triggers page reload on client)."""
    try:
        config = load_service_config()
        if config:
            return JSONResponse(content={"success": True, "message": "Service configuration loaded"})
        else:
            return JSONResponse(content={"error": "No saved configuration found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error loading service config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/export_service_config")
async def api_export_service_config():
    """Export current service configuration as downloadable JSON file."""
    try:
        config = load_service_config() or {}

        # Create JSON response with proper headers for download
        from fastapi.responses import Response
        import json

        json_str = json.dumps(config, indent=2)
        filename = f"monitaqc_config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

        return Response(
            content=json_str,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting service config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Temporarily disabled until python-multipart is installed
# @app.post("/api/import_service_config")
# async def api_import_service_config(file: UploadFile = File(...)):
#     """Import service configuration from uploaded JSON file."""
#     try:
#         import json
#
#         # Read uploaded file
#         contents = await file.read()
#         config = json.loads(contents.decode('utf-8'))
#
#         # Save the imported configuration
#         if save_service_config(config):
#             return JSONResponse(content={"success": True, "message": "Configuration imported successfully"})
#         else:
#             return JSONResponse(content={"error": "Failed to save imported configuration"}, status_code=500)
#     except json.JSONDecodeError:
#         return JSONResponse(content={"error": "Invalid JSON file"}, status_code=400)
#     except Exception as e:
#         logger.error(f"Error importing service config: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# AI ASSISTANT ENDPOINTS
# =============================================================================

@app.post("/api/ai_config")
async def save_ai_config(config: Dict[str, Any]):
    """Save AI model configuration to DATA_FILE and Redis (for runtime use)."""
    try:
        model = config.get("model", "claude")
        api_key = config.get("api_key", "")

        if not api_key:
            return JSONResponse(content={"error": "API key is required"}, status_code=400)

        # Store in Redis for runtime use
        if watcher and watcher.redis_connection:
            watcher.redis_connection.redis_connection.set("ai_model", model)
            watcher.redis_connection.redis_connection.set("ai_api_key", api_key)

        # Load existing config and update AI section
        existing_config = load_service_config() or {}
        existing_config["ai"] = {
            "model": model,
            "api_key": api_key
        }

        # Save to DATA_FILE for persistence
        if save_service_config(existing_config):
            logger.info(f"AI configuration saved to {DATA_FILE}: model={model}")
            return JSONResponse(content={"success": True, "message": f"AI configuration saved ({model})"})
        else:
            return JSONResponse(content={"error": "Failed to save AI configuration to file"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving AI config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/ai_query")
async def query_ai(request: Dict[str, Any]):
    """Query AI with production data from TimescaleDB and real-time metrics."""
    try:
        user_query = request.get("query", "")
        if not user_query:
            return JSONResponse(content={"error": "Query is required"}, status_code=400)

        # Get AI configuration from Redis
        if not watcher or not watcher.redis_connection:
            return JSONResponse(content={"error": "Redis not available"}, status_code=503)

        r = watcher.redis_connection.redis_connection
        model = r.get("ai_model")
        api_key = r.get("ai_api_key")

        if not model or not api_key:
            return JSONResponse(content={
                "error": "AI not configured. Please set your API key in the AI Configuration panel."
            }, status_code=400)

        model = model.decode('utf-8') if isinstance(model, bytes) else model
        api_key = api_key.decode('utf-8') if isinstance(api_key, bytes) else api_key

        # Gather current system context
        system_context = {
            "current_time": datetime.now().isoformat(),
            "real_time_data": {
                "encoder_value": r.get("encoder") or 0,
                "ok_counter": r.get("ok_counter") or 0,
                "ng_counter": r.get("ng_counter") or 0,
                "shipment": r.get("shipment") or "no_shipment",
                "is_moving": r.get("is_moving") == b"true",
                "downtime_seconds": r.get("downtime_seconds") or 0,
            },
            "inference_stats": {
                "service_type": "YOLO" if YOLO_INFERENCE_URL else "Gradio",
                "service_url": YOLO_INFERENCE_URL or f"https://{GRADIO_MODEL}.hf.space",
            }
        }

        # Build concise AI prompt
        system_prompt = f"""AI assistant for MonitaQC quality control system.

Current: Encoder={system_context['real_time_data']['encoder_value']}, OK={system_context['real_time_data']['ok_counter']}, NG={system_context['real_time_data']['ng_counter']}, Shipment={system_context['real_time_data']['shipment']}, Moving={'Yes' if system_context['real_time_data']['is_moving'] else 'No'}

Use tools to query TimescaleDB (timescaledb:5432/monitaqc) for historical data. Provide actionable insights."""

        # Call AI API based on model
        ai_response = await call_ai_model(model, api_key, system_prompt, user_query)

        return JSONResponse(content={"response": ai_response, "model": model})

    except Exception as e:
        logger.error(f"Error querying AI: {e}")
        return JSONResponse(content={"error": f"AI query failed: {str(e)}"}, status_code=500)


def get_ai_tools():
    """Define tools that AI can use to query data autonomously."""
    return [
        {
            "name": "query_database",
            "description": "Execute SQL query on TimescaleDB to retrieve production data, metrics, defects, or any historical data. Returns query results as JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute. Can query tables like production_metrics, inference_results, etc."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_redis_data",
            "description": "Get current real-time data from Redis cache. Returns current values for encoder, counters, shipment, movement status, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Redis keys to retrieve. Common keys: encoder, ok_counter, ng_counter, shipment, is_moving, downtime_seconds"
                    }
                },
                "required": ["keys"]
            }
        },
        {
            "name": "get_system_logs",
            "description": "Retrieve recent application logs to diagnose issues or understand system behavior.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "Number of recent log lines to retrieve",
                        "default": 50
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional filter pattern to match in logs"
                    }
                },
                "required": []
            }
        }
    ]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return results."""
    try:
        if tool_name == "query_database":
            # Execute TimescaleDB query
            import psycopg2
            conn = psycopg2.connect(
                host="timescaledb",
                port=5432,
                database="monitaqc",
                user="monitaqc",
                password="monitaqc2024"
            )
            cursor = conn.cursor()
            cursor.execute(tool_input["query"])
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            cursor.close()
            conn.close()

            # Format as JSON
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(column_names, row)))
            return json.dumps(formatted_results, default=str)

        elif tool_name == "get_redis_data":
            # Get Redis data
            if not watcher or not watcher.redis_connection:
                return json.dumps({"error": "Redis not available"})

            r = watcher.redis_connection.redis_connection
            results = {}
            for key in tool_input["keys"]:
                value = r.get(key)
                if value:
                    results[key] = value.decode('utf-8') if isinstance(value, bytes) else value
                else:
                    results[key] = None
            return json.dumps(results)

        elif tool_name == "get_system_logs":
            # Get recent logs (simplified - returns last N log entries)
            lines = tool_input.get("lines", 50)
            filter_pattern = tool_input.get("filter")

            # For now, return a simple status message
            # In production, this would read from actual log files
            return json.dumps({
                "message": f"Log retrieval with {lines} lines" + (f" filtered by '{filter_pattern}'" if filter_pattern else ""),
                "note": "Full log integration pending"
            })

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return json.dumps({"error": str(e)})


async def call_ai_model(model: str, api_key: str, system_prompt: str, user_query: str) -> str:
    """Call the appropriate AI model API with tool support."""
    try:
        if model == "claude":
            # Anthropic Claude API with tool use
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            tools = get_ai_tools()
            messages = [{"role": "user", "content": user_query}]

            # Agentic loop - allow multiple tool calls
            max_iterations = 5
            for iteration in range(max_iterations):
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools
                )

                # Check if Claude wants to use tools
                if response.stop_reason == "end_turn":
                    # No more tools, return final answer
                    for block in response.content:
                        if hasattr(block, 'text'):
                            return block.text
                    return "No response generated"

                elif response.stop_reason == "tool_use":
                    # Add assistant response to messages
                    messages.append({"role": "assistant", "content": response.content})

                    # Execute each tool call
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_result = execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result
                            })

                    # Add tool results to messages
                    messages.append({"role": "user", "content": tool_results})
                    # Continue loop to let Claude process results
                else:
                    # Unexpected stop reason
                    return f"Unexpected response: {response.stop_reason}"

            return "Maximum iterations reached. Please simplify your query."

        elif model == "chatgpt":
            # OpenAI ChatGPT API with function calling
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Convert tools to OpenAI format
            functions = []
            for tool in get_ai_tools():
                functions.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                })

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # Agentic loop
            max_iterations = 5
            for iteration in range(max_iterations):
                response = client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for 128K context
                    messages=messages,
                    functions=functions,
                    max_tokens=4096
                )

                message = response.choices[0].message

                if message.function_call:
                    # Execute function
                    function_name = message.function_call.name
                    function_args = json.loads(message.function_call.arguments)
                    function_result = execute_tool(function_name, function_args)

                    # Add to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": message.function_call.arguments
                        }
                    })
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_result
                    })
                else:
                    # No function call, return answer
                    return message.content

            return "Maximum iterations reached. Please simplify your query."

        elif model == "gemini":
            # Gemini doesn't support function calling in same way, use basic response
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel('gemini-pro')
            full_prompt = f"{system_prompt}\n\nUser Question: {user_query}"
            response = model_instance.generate_content(full_prompt)
            return response.text

        elif model == "local":
            # Local model (Ollama) - basic response
            import requests
            endpoint = api_key
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": f"{system_prompt}\n\nUser: {user_query}\n\nAssistant:",
                    "stream": False
                }
            )
            return response.json().get("response", "No response from local model")
        else:
            return f"Unsupported model: {model}"

    except ImportError as e:
        return f"Required AI library not installed: {str(e)}. Please install: pip install anthropic openai google-generativeai"
    except Exception as e:
        logger.error(f"AI model call failed: {e}")
        return f"AI request failed: {str(e)}"


# =============================================================================
# COMMAND ENDPOINT (must be last due to catch-all pattern)
# =============================================================================

@app.post("/{command}")
async def send_command(command: str, value: Optional[int] = None):
    """Send command to watcher device.

    Examples:
        POST /U_ON_B_OFF - Turn U light on, B light off
        POST /SET_VERBOSE?value=1 - Enable verbose mode
        POST /OK_OFFSET_DELAY?value=100 - Set OK offset delay to 100ms
    """
    if watcher_instance is None:
        raise HTTPException(status_code=503, detail="Watcher device not initialized")

    try:
        # Convert user-friendly command to actual command
        if command in USER_COMMANDS:
            cmd = USER_COMMANDS[command]
            # Add value for commands that require it
            if command in COMMANDS_WITH_VALUE:
                if value is None:
                    raise HTTPException(status_code=400, detail=f"Command '{command}' requires a value parameter")
                cmd = f"{cmd},{value}"
        else:
            cmd = command

        watcher_instance._send_message(f"{cmd}\n")
        return {"status": "ok", "command": cmd, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send command: {str(e)}")

def start_web_server():
    """Start the FastAPI web server in a separate thread."""
    try:
        logger.info(f"Starting web server on {WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
        import asyncio
        config = uvicorn.Config(
            app,
            host=WEB_SERVER_HOST,
            port=WEB_SERVER_PORT,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            limit_concurrency=100,
            timeout_keep_alive=5
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except Exception as e:
        logger.error(f"Web server error: {e}")

# =============================================================================

# Legacy variable names for backward compatibility
capture_mode = CAPTURE_MODE
time_between_two_package = TIME_BETWEEN_TWO_PACKAGE
remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
parent_object_list = PARENT_OBJECT_LIST

try:
    os.mkdir("raw_images")
except:
    pass

session = requests.Session()
def req_predict(image):
    """
    Unified prediction function supporting both YOLO and Gradio APIs.
    Returns detections in standardized format with keys: x1, y1, x2, y2, confidence, class_id, name
    """
    global inference_times, frame_intervals, last_inference_timestamp, max_inference_samples
    start_time = time.time()

    # Track frame-to-frame interval
    if last_inference_timestamp > 0:
        interval = (start_time - last_inference_timestamp) * 1000  # Convert to ms
        frame_intervals.append(interval)
        if len(frame_intervals) > max_inference_samples:
            frame_intervals.pop(0)

    # Check if we're using Gradio API (HuggingFace Space)
    if "hf.space" in YOLO_INFERENCE_URL or "huggingface" in YOLO_INFERENCE_URL:
        try:
            from gradio_client import Client, handle_file
            import tempfile
            import cv2
            import numpy as np

            # Initialize Gradio client (cached globally for reuse)
            if not hasattr(req_predict, 'gradio_client'):
                logger.info(f"Initializing Gradio client for {YOLO_INFERENCE_URL}")
                req_predict.gradio_client = Client(YOLO_INFERENCE_URL)
                logger.info("Gradio client initialized successfully")

            # Decode image bytes and save to temp file
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, img)
                tmp_path = tmp.name

            # Get model name and confidence from globals (loaded from config)
            model_name = globals().get('GRADIO_MODEL', 'Data Matrix')
            confidence = globals().get('GRADIO_CONFIDENCE_THRESHOLD', 0.25)

            logger.info(f"Calling Gradio API: model={model_name}, confidence={confidence}")

            # Call Gradio /api/detect endpoint
            result = req_predict.gradio_client.predict(
                handle_file(tmp_path),
                model_name,
                confidence,
                api_name="/detect"
            )

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

            logger.info(f"Gradio API response: {len(result) if isinstance(result, list) else 0} detection(s)")
            logger.info(f"Gradio result type: {type(result)}")
            if result:
                logger.info(f"First detection sample: {result[0] if isinstance(result, list) else result}")

            # Convert Gradio response format to match YOLO format exactly
            # YOLO format: {"xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"}
            # Gradio format: {"x1", "y1", "x2", "y2", "confidence", "class_id", "name"}
            normalized_detections = []
            if result and isinstance(result, list):
                for det in result:
                    # Convert Gradio format to YOLO format
                    normalized_det = {
                        "xmin": det.get('x1', det.get('xmin', 0)),
                        "ymin": det.get('y1', det.get('ymin', 0)),
                        "xmax": det.get('x2', det.get('xmax', 0)),
                        "ymax": det.get('y2', det.get('ymax', 0)),
                        "confidence": det.get('confidence', 0.0),
                        "class": det.get('class_id', det.get('class', 0)),
                        "name": det.get('name', f"Class {det.get('class_id', det.get('class', 0))}")
                    }
                    normalized_detections.append(normalized_det)

                logger.info(f"Normalized {len(normalized_detections)} detections to YOLO format")
                if normalized_detections:
                    logger.info(f"First normalized detection: {normalized_detections[0]}")

            # Cache detections in module-level variables and Redis (for cross-process sharing)
            global latest_detections, latest_detections_timestamp, watcher_instance
            latest_detections = normalized_detections
            latest_detections_timestamp = time.time()

            # Also store in Redis for cross-process access
            if watcher_instance and watcher_instance.redis_connection:
                try:
                    watcher_instance.redis_connection.redis_connection.setex(
                        "gradio_last_detection_timestamp",
                        60,  # Expire after 60 seconds
                        str(latest_detections_timestamp)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache detection timestamp in Redis: {e}")

            logger.info(f"Cached {len(normalized_detections)} detections at timestamp {latest_detections_timestamp}")

            # Track inference time and update timestamp
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(elapsed)
            if len(inference_times) > max_inference_samples:
                inference_times.pop(0)
            last_inference_timestamp = time.time()

            # Store timing in Redis for cross-process access
            try:
                if watcher_instance and watcher_instance.redis_connection:
                    watcher_instance.redis_connection.redis_connection.lpush("inference_times", str(elapsed))
                    watcher_instance.redis_connection.redis_connection.ltrim("inference_times", 0, max_inference_samples - 1)
                    if last_inference_timestamp > 0:
                        interval_ms = (time.time() - last_inference_timestamp) * 1000
                        watcher_instance.redis_connection.redis_connection.lpush("frame_intervals", str(interval_ms))
                        watcher_instance.redis_connection.redis_connection.ltrim("frame_intervals", 0, max_inference_samples - 1)
                    watcher_instance.redis_connection.redis_connection.set("last_inference_timestamp", str(time.time()))
            except Exception as e:
                logger.warning(f"Failed to store timing in Redis: {e}")

            return json.dumps(normalized_detections)

        except Exception as e:
            logger.error(f"Gradio prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "[]"
    else:
        # Original YOLO inference endpoint
        try:
            # Use Connection: close header to prevent keep-alive issues
            headers = {'Connection': 'close'}
            response = requests.post(YOLO_INFERENCE_URL, files={"image": image}, headers=headers, timeout=10)
            response.raise_for_status()

            # Track inference time and update timestamp
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(elapsed)
            if len(inference_times) > max_inference_samples:
                inference_times.pop(0)

            # Store timing in Redis for cross-process access
            try:
                if watcher_instance and watcher_instance.redis_connection:
                    watcher_instance.redis_connection.redis_connection.lpush("inference_times", str(elapsed))
                    watcher_instance.redis_connection.redis_connection.ltrim("inference_times", 0, max_inference_samples - 1)

                    # Calculate and store frame interval
                    last_ts_str = watcher_instance.redis_connection.redis_connection.get("last_inference_timestamp")
                    if last_ts_str:
                        last_ts = float(last_ts_str.decode('utf-8'))
                        interval_ms = (time.time() - last_ts) * 1000
                        watcher_instance.redis_connection.redis_connection.lpush("frame_intervals", str(interval_ms))
                        watcher_instance.redis_connection.redis_connection.ltrim("frame_intervals", 0, max_inference_samples - 1)

                    watcher_instance.redis_connection.redis_connection.set("last_inference_timestamp", str(time.time()))
            except Exception as e:
                logger.warning(f"Failed to store timing in Redis: {e}")

            last_inference_timestamp = time.time()
            return response.text  # Return JSON string, not parsed dict
        except requests.exceptions.RequestException as e:
            logger.error(f"YOLO prediction failed: {e}")
            return "[]"  # Return empty JSON array string


# prepared data based on: https://docs.google.com/spreadsheets/d/1G1StYfEsSuQq9S6EWPO7WDpGGeqtRcZu6vN2-ixSqV8/edit?gid=1480612965#gid=1480612965&range=46:46
try:
    with open(DATA_FILE, "r") as file:
        data = json.load(file)
        # Handle both formats: direct list or wrapped in {"prepared_query_data": [...]}
        if isinstance(data, list):
            prepared_query_data = data
        elif isinstance(data, dict) and "prepared_query_data" in data:
            prepared_query_data = data["prepared_query_data"]
        else:
            # Config file doesn't have prepared_query_data - use empty list (no datamatrix validation)
            logger.info(f"No prepared_query_data found in {DATA_FILE}, using empty list (detection only mode)")
            prepared_query_data = []
except Exception as e:
    logger.error(f"Error loading data from {DATA_FILE}: {e}")
    prepared_query_data = []

set_model_url = "http://yolo_inference:4442/v1/object-detection/yolov5s/set-model"

# res = requests.post(set_model_url, data={"model_path":"best.pt"})

class CameraBuffer:
    def __init__(self, source, exposure, gain) -> None:
        self.source = source
        self.is_ip_camera = isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://"))

        # Initialize camera with appropriate backend
        if self.is_ip_camera:
            # IP camera - use default backend (FFMPEG for streams)
            self.camera = cv2.VideoCapture(source)
            logger.info(f"Initializing IP camera: {source.split('@')[-1] if '@' in source else source}")
        else:
            # USB/V4L2 camera
            self.camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
            logger.info(f"Initializing USB camera: {source}")

        # Set properties
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if self.is_ip_camera:
            # IP camera settings (limited control via RTSP)
            # Note: Not all IP cameras support these settings via OpenCV
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, IP_CAMERA_BRIGHTNESS)
            self.camera.set(cv2.CAP_PROP_CONTRAST, IP_CAMERA_CONTRAST)
            self.camera.set(cv2.CAP_PROP_SATURATION, IP_CAMERA_SATURATION)
            logger.info(f"IP camera settings - Brightness: {IP_CAMERA_BRIGHTNESS}, Contrast: {IP_CAMERA_CONTRAST}, Saturation: {IP_CAMERA_SATURATION}")
        else:
            # USB camera specific settings
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J', 'P', 'G'))
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 6000)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.camera.set(cv2.CAP_PROP_SATURATION, 50)
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 0)
            self.camera.set(cv2.CAP_PROP_GAIN, gain)
            self.camera.set(cv2.CAP_PROP_GAMMA, 1)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 0)

        success, frame = self.camera.retrieve(0)
        self.success = success
        self.frame = frame
        self.stop = False
        # ROI (Region of Interest) settings - None means full frame
        self.roi_enabled = False
        self.roi_xmin = 0
        self.roi_ymin = 0
        self.roi_xmax = 1280  # Default to full width
        self.roi_ymax = 720   # Default to full height
        threading.Thread(target=self.buffer).start()

    def buffer(self):
        failure_count = 0
        max_failures = 100  # Reconnect after 100 consecutive failures

        while True:
            try:
                if self.stop:
                    break
                self.camera.grab()
                success, frame = self.camera.retrieve(0)
                if success:
                    self.frame = frame
                    self.success = success
                    failure_count = 0  # Reset counter on successful read
                else:
                    self.success = success
                    failure_count += 1

                    # Auto-reconnect after consecutive failures
                    if failure_count >= max_failures:
                        logger.warning(f"Camera connection lost after {failure_count} failures. Attempting reconnect...")
                        try:
                            self.camera.release()
                            time.sleep(1)  # Brief pause before reconnecting

                            # Reinitialize camera
                            if self.is_ip_camera:
                                self.camera = cv2.VideoCapture(self.source)
                                logger.info(f"Reconnecting IP camera: {self.source.split('@')[-1] if '@' in self.source else self.source}")
                            else:
                                self.camera = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                                logger.info(f"Reconnecting USB camera: {self.source}")

                            # Reset failure counter after reconnect attempt
                            failure_count = 0
                            logger.info("Camera reconnection attempted")
                        except Exception as e:
                            logger.error(f"Camera reconnection failed: {e}")
                            time.sleep(5)  # Wait longer before next attempt

                time.sleep(0.0001)
            except Exception as e:
                logger.error(f"Camera buffer error: {e}")
                time.sleep(0.1)

    def read(self):
        frame = self.frame
        if frame is not None and self.roi_enabled:
            frame = frame[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax]
        return frame
    
    def restart_camera(self):
        self.stop = True
        self.camera.release()
        # Reinitialize with appropriate backend
        if self.is_ip_camera:
            self.camera = cv2.VideoCapture(self.source)
        else:
            self.camera = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        success, frame = self.camera.retrieve(0)
        self.success = success
        self.frame = frame
        self.stop = False
        threading.Thread(target=self.buffer).start()

    def grab(self):
        """Expose the underlying camera's grab method"""
        return self.camera.grab()

    def retrieve(self):
        """Expose the retrieve functionality"""
        return self.camera.retrieve(0)

class RedisConnection:
    def __init__(self, redis_hostname, redis_port):
        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_connection = self.connect_to_redis()

    # Connecting to Radis database
    def connect_to_redis(self):
        return Redis(self.redis_hostname, self.redis_port, db=3)

    def set_flag(self, list_lenght):
        with self.redis_connection.pipeline() as pipe:
            pipe.delete("camera_list")
            for i in range(list_lenght):
                pipe.rpush("camera_list", 0)
            pipe.execute()

    def update_encoder_redis(self, encoder):
        self.redis_connection.set("encoder_values", json.dumps(encoder))

    def set_captuting_flag(self, key):
        self.redis_connection.set(key, 1)

    def set_redis(self, key, value):
        self.redis_connection.set(key, value)

    def get_redis(self, key):
        return self.redis_connection.get(key)

    def set_light_mode(self, mode):
        self.redis_connection.set("light_mode", mode)

    def update_queue_messages_redis(self, queue_messages , stream_name="dms"):
        with self.redis_connection.pipeline() as pipe2:
            # pipe2.delete("queue_messages")
            # for i in range(queue_messages):
            pipe2.rpush(stream_name, queue_messages)
            pipe2.execute()
        # self.redis_connection.set('dms', queue_messages)

    def pop_queue_messages_redis(self, stream_name="frames_queue"):
            return self.redis_connection.lpop(stream_name)


class ArduinoSocket:
    def __init__(self, camera_paths: List[str] = None, serial_port=None, serial_baudrate=None):
        """Initialize ArduinoSocket with dynamic camera support.

        Args:
            camera_paths: List of video device paths to use. If None, uses auto-detected cameras.
            serial_port: Serial port path (default from env)
            serial_baudrate: Serial baud rate (default from env)
        """
        # Use configurable serial settings
        self.serial_port = serial_port or WATCHER_USB
        self.serial_baudrate = serial_baudrate or SERIAL_BAUDRATE
        self.serial_mode = SERIAL_MODE
        self.serial_available = False

        # Try to initialize serial, but continue without it if unavailable
        try:
            self.serial = serial.Serial(self.serial_port, self.serial_baudrate, 8, 'N', 1, timeout=1)
            self.serial.flushInput()  # clear input serial buffer
            self.serial.flushOutput()  # clear output serial buffer
            self.serial_available = True
            logger.info(f"Serial connected: {self.serial_port} @ {self.serial_baudrate}")
        except Exception as e:
            self.serial = None
            logger.warning(f"Serial not available ({self.serial_port}): {e}. Running in camera-only mode.")

        self.encoder = 0
        self.stop_thread = False
        self.redis_connection = RedisConnection(REDIS_HOST, REDIS_PORT)
        self.last_encoder_value = 0
        self.last_capture_ts = time.time()
        self.encoder_value = 0
        self.health_check = False
        self.is_moving = False
        self.step = 15
        # Extended watcher-style metrics (used in new serial mode)
        self.pulses_per_second = 0
        self.pulses_per_minute = 0
        self.downtime_seconds = 0
        self.ok_counter = 0
        self.ng_counter = 0
        self.status_value = 0
        self.u_status = False
        self.b_status = False
        self.warning_status = False
        # Extra analog / power metrics (ANG / PWR in watcher)
        self.analog_value = 0
        self.power_value = 0
        # Verbose configuration values (mirror watcher verbose data)
        self.ok_offset_delay = 0      # OOD
        self.ok_duration_pulses = 0   # ODP
        self.ok_duration_percent = 0  # ODL
        self.ok_encoder_factor = 0    # OEF
        self.ng_offset_delay = 0      # NOD
        self.ng_duration_pulses = 0   # NDP
        self.ng_duration_percent = 0  # NDL
        self.ng_encoder_factor = 0    # NEF
        self.external_reset = 0       # EXT
        self.baud_rate = self.serial_baudrate  # BUD
        self.downtime_threshold = 0   # DWT
        self.last_speed_point = 0
        self.last_speed_time = time.time()
        self.last_capture_encoder = self.step * -1
        self.last_encoder_value_captured = 0
        self.take = False
        self.d_or_k = 0
        self.last_d_or_k = time.time()
        self.on_or_off_enc = True
        self.last_move_time = time.time()
        self.ejector_start_ts = time.time()
        self.ejector_running = False
        self.ejection_queue = []  # Queue of encoder targets for ejection
        self.shipment="no_shipment"
        self.stream_histogram_data = []  # Histogram data for detected objects
        self.old_shipment = "no_shipment"
        # Minimal data snapshot coming from watcher (encoder + movement info)
        self.data = {"encoder_value": 0}

        # Dynamic camera initialization
        # Use provided paths, or auto-detected, or fallback to legacy env vars
        if camera_paths is None:
            if DETECTED_CAMERAS:
                camera_paths = DETECTED_CAMERAS
            else:
                # Only include non-empty paths from env vars
                camera_paths = [p for p in [CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH] if p]

        # Store camera paths for reference
        self.camera_paths = camera_paths
        self.cameras: Dict[int, CameraBuffer] = {}  # Dynamic camera storage {1: cam, 2: cam, ...}
        self.camera_metadata: Dict[int, Dict[str, Any]] = {}  # Metadata for each camera

        if camera_paths:
            logger.info(f"Initializing {len(camera_paths)} camera(s) from paths: {camera_paths}")
        else:
            logger.info("No USB cameras configured. Use IP Camera Discovery to add IP cameras.")

        # Initialize cameras dynamically (1-indexed for backward compatibility)
        for idx, cam_path in enumerate(camera_paths, start=1):
            try:
                cam = CameraBuffer(cam_path, exposure=100, gain=100)
                self.cameras[idx] = cam
                logger.info(f"Camera {idx} initialized: {cam_path} (success={cam.success})")

                # Store metadata for USB cameras
                self.camera_metadata[idx] = {
                    "name": f"USB Camera {idx}",
                    "type": "usb",
                    "path": cam_path,
                    "source": cam_path
                }

                time.sleep(1)
            except Exception as e:
                logger.warning(f"Failed to initialize camera {idx} at {cam_path}: {e}")
                # Create a dummy camera object with success=False
                class DummyCamera:
                    success = False
                    def read(self): return None
                self.cameras[idx] = DummyCamera()
                # Still store metadata even for failed cameras
                self.camera_metadata[idx] = {
                    "name": f"USB Camera {idx} (failed)",
                    "type": "usb",
                    "path": cam_path,
                    "source": cam_path,
                    "error": str(e)
                }

        # Legacy attribute compatibility (cam_1, cam_2, cam_3, cam_4)
        self.cam_1 = self.cameras.get(1)
        self.cam_2 = self.cameras.get(2)
        self.cam_3 = self.cameras.get(3)
        self.cam_4 = self.cameras.get(4)

        # Note: Service config is now loaded via apply_saved_config_at_startup() after watcher init

        # Turn off lights at startup to ensure known state
        self.off()
        # Turn off ejector at startup
        self.motor_stop()

        threading.Thread(target=self.capture_frames).start()
        time.sleep(0.5)
        threading.Thread(target=self.run).start()
        time.sleep(0.5)
        threading.Thread(target=self.run_ejector).start()
        time.sleep(0.5)
        threading.Thread(target=self.stream_results).start()
        time.sleep(0.5)
        threading.Thread(target=self.write_production_metrics_loop).start()
        time.sleep(0.5)
        threading.Thread(target=self.run_barcode_scanner, daemon=True).start()


    def on_up_down_off(self):
        self._send_message('1\n')

    def on_down_up_off(self):
        self._send_message('2\n')

    def on_up_down_on(self):
        """Both U and B lights ON."""
        self._send_message('9\n')

    def off(self):
        self._send_message('8\n')

    def _get_current_light_mode(self) -> str:
        """Get current light mode based on serial status feedback."""
        if self.u_status and not self.b_status:
            return "U_ON_B_OFF"
        elif not self.u_status and self.b_status:
            return "U_OFF_B_ON"
        elif self.u_status and self.b_status:
            return "U_ON_B_ON"
        else:
            return "U_OFF_B_OFF"

    def _set_light_mode(self, mode: str, force: bool = False):
        """Set light mode based on state configuration.

        If LIGHT_STATUS_CHECK_ENABLED is True (closed-loop):
            Checks actual serial status before sending command.
            Only sends command if current status differs from requested mode.

        If LIGHT_STATUS_CHECK_ENABLED is False (open-loop):
            Always sends command without checking serial status.

        Supported modes:
        - U_ON_B_OFF: U light on, B light off (command 1)
        - U_OFF_B_ON: U light off, B light on (command 2)
        - U_ON_B_ON: Both lights on (command 9)
        - U_OFF_B_OFF: Both lights off (command 8)

        Args:
            mode: Target light mode
            force: If True, send command regardless of current status
        """
        # Check if current status already matches requested mode (closed-loop mode)
        if not force and LIGHT_STATUS_CHECK_ENABLED and self.serial_available:
            current_mode = self._get_current_light_mode()
            if current_mode == mode:
                logger.debug(f"Light mode already {mode} (verified via serial), skipping command")
                return

        mode_commands = {
            "U_ON_B_OFF": '1\n',      # on_up_down_off
            "U_OFF_B_ON": '2\n',      # on_down_up_off
            "U_ON_B_ON": '9\n',       # both on
            "U_OFF_B_OFF": '8\n',     # off
        }
        command = mode_commands.get(mode, '1\n')  # Default to U_ON_B_OFF
        self._send_message(command)
        logger.debug(f"Set light mode: {mode} (command: {command.strip()})")

    def reset_encoder(self):
        if self.serial_available and self.serial:
            self.serial.write('3\n'.encode('utf-8'))
        self.last_encoder_value = 0
        self.last_capture_encoder = self.step * -1
        self.encoder_value = 0

    def close(self):
        if self.serial_available and self.serial:
            self.serial.close()

    def _send_message(self, msg):
        if self.serial_available and self.serial:
            self.serial.write(msg.encode('utf-8'))

    def set_PWM_backlight(self, pwm):
        if self.serial_available and self.serial:
            self.serial.write(f"5,{pwm}\n".encode('utf-8'))

    def set_PWM_uplight(self, pwm):
        if self.serial_available and self.serial:
            self.serial.write(f"4,{pwm}\n".encode('utf-8'))

    def motor_start(self):
        self._send_message('6\n')
    
    def motor_stop(self):
        self._send_message('7\n')

    def set_step(self, step):
        self.step = step

    def keep(self):
        self.d_or_k = 1
        self.last_d_or_k = time.time()

    def on_or_off_encoder(self, on_or_off):
        self.on_or_off_enc = on_or_off

    def buzzer(self, t):
        ...

    def led(self, t):
        ...

    def find_barcode_scanner(self):
        """Dynamically find barcode scanner device."""
        import glob
        import subprocess

        # First try /dev/input/by-id/ (works on host or if mounted)
        scanner_patterns = [
            "/dev/input/by-id/*[Bb]ar[Cc]ode*event*",
            "/dev/input/by-id/*[Ss]canner*event*",
        ]
        for pattern in scanner_patterns:
            devices = glob.glob(pattern)
            if devices:
                device_path = os.path.realpath(devices[0])
                return device_path

        # Fallback: scan all event devices and check their names
        try:
            for event_file in sorted(glob.glob("/dev/input/event*")):
                try:
                    # Try to read device name from /sys
                    event_num = event_file.split("event")[-1]
                    name_path = f"/sys/class/input/event{event_num}/device/name"
                    if os.path.exists(name_path):
                        with open(name_path, 'r') as f:
                            name = f.read().strip().lower()
                            if 'barcode' in name or 'scanner' in name or '2d' in name:
                                logger.info(f"Found barcode scanner: {event_file} ({name})")
                                return event_file
                except:
                    pass
        except Exception as e:
            logger.debug(f"Error scanning input devices: {e}")

        return None

    def run_barcode_scanner(self):
        """Thread to read barcode scanner input and set shipment ID."""
        import struct
        import select

        # Key code to character mapping (US keyboard layout)
        KEY_MAP = {
            2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '0',
            16: 'q', 17: 'w', 18: 'e', 19: 'r', 20: 't', 21: 'y', 22: 'u', 23: 'i', 24: 'o', 25: 'p',
            30: 'a', 31: 's', 32: 'd', 33: 'f', 34: 'g', 35: 'h', 36: 'j', 37: 'k', 38: 'l',
            44: 'z', 45: 'x', 46: 'c', 47: 'v', 48: 'b', 49: 'n', 50: 'm',
            12: '-', 13: '=', 26: '[', 27: ']', 39: ';', 40: "'", 41: '`',
            43: '\\', 51: ',', 52: '.', 53: '/',
            57: ' ',  # Space
        }
        KEY_MAP_SHIFT = {
            2: '!', 3: '@', 4: '#', 5: '$', 6: '%', 7: '^', 8: '&', 9: '*', 10: '(', 11: ')',
            12: '_', 13: '+',
        }

        EVENT_SIZE = struct.calcsize('llHHI')
        EV_KEY = 0x01
        KEY_ENTER = 28

        barcode_buffer = ""
        scanner_device = None
        scanner_fd = None
        last_scan_check = 0

        while not self.stop_thread:
            try:
                # Periodically check for scanner device (every 5 seconds)
                if scanner_fd is None or time.time() - last_scan_check > 5:
                    last_scan_check = time.time()
                    new_device = self.find_barcode_scanner()

                    if new_device and new_device != scanner_device:
                        # Close old device if open
                        if scanner_fd:
                            try:
                                os.close(scanner_fd)
                            except:
                                pass

                        # Open new device
                        try:
                            scanner_fd = os.open(new_device, os.O_RDONLY | os.O_NONBLOCK)
                            scanner_device = new_device
                            logger.info(f"Barcode scanner connected: {scanner_device}")
                        except Exception as e:
                            logger.warning(f"Cannot open barcode scanner {new_device}: {e}")
                            scanner_fd = None
                            scanner_device = None
                    elif not new_device and scanner_fd:
                        # Scanner disconnected
                        try:
                            os.close(scanner_fd)
                        except:
                            pass
                        scanner_fd = None
                        scanner_device = None
                        logger.info("Barcode scanner disconnected")

                if scanner_fd is None:
                    time.sleep(1)
                    continue

                # Use select to wait for data with timeout
                readable, _, _ = select.select([scanner_fd], [], [], 0.5)
                if not readable:
                    continue

                # Read events
                try:
                    data = os.read(scanner_fd, EVENT_SIZE * 10)
                except OSError as e:
                    if e.errno == 11:  # EAGAIN - no data available
                        continue
                    raise

                # Process events
                for i in range(0, len(data), EVENT_SIZE):
                    if i + EVENT_SIZE > len(data):
                        break

                    _, _, ev_type, code, value = struct.unpack('llHHI', data[i:i+EVENT_SIZE])

                    if ev_type != EV_KEY or value != 1:  # Only key press events
                        continue

                    if code == KEY_ENTER:
                        # Barcode complete - set shipment ID
                        if barcode_buffer.strip():
                            new_shipment = barcode_buffer.strip()
                            logger.info(f"Barcode scanned: {new_shipment}")

                            # Update shipment
                            self.shipment = new_shipment
                            if self.redis_connection:
                                try:
                                    self.redis_connection.redis_connection.set("shipment", new_shipment)
                                except Exception as e:
                                    logger.warning(f"Failed to set shipment in Redis: {e}")

                            # Create directory for shipment
                            try:
                                os.makedirs(f"raw_images/{new_shipment}", exist_ok=True)
                            except Exception as e:
                                logger.warning(f"Failed to create shipment directory: {e}")

                        barcode_buffer = ""
                    elif code in KEY_MAP:
                        barcode_buffer += KEY_MAP[code]

            except Exception as e:
                logger.error(f"Barcode scanner error: {e}")
                if scanner_fd:
                    try:
                        os.close(scanner_fd)
                    except:
                        pass
                scanner_fd = None
                scanner_device = None
                time.sleep(2)

    def signal_captured(self):
        """Check if capture should be triggered based on encoder change and state thresholds.

        Uses first phase's thresholds if set, otherwise falls back to state-level thresholds.
        steps=-1 means infinite loop (always capture, no encoder check).
        steps=1 means capture on every 1 step change (default).
        steps=N means capture on every N step changes.
        analog=-1 means analog threshold is disabled.
        """
        # Get thresholds from current state/phase
        current_state = state_manager.current_state if state_manager else None

        # Prefer first phase thresholds, fallback to state-level
        steps_threshold = 1  # Default: capture on every step change
        analog_threshold = -1
        if current_state:
            if current_state.phases and len(current_state.phases) > 0:
                first_phase = current_state.phases[0]
                steps_threshold = first_phase.steps
                analog_threshold = first_phase.analog
            # Fallback to state-level if phase thresholds are default
            if steps_threshold == 1 and current_state.steps != 1:
                steps_threshold = current_state.steps
            if analog_threshold == -1 and current_state.analog != -1:
                analog_threshold = current_state.analog

        # If steps threshold is -1, always trigger (infinite loop, no encoder check)
        if steps_threshold < 0:
            # Also check analog threshold if enabled (>= 0)
            if analog_threshold >= 0:
                return self.analog_value >= analog_threshold
            return True

        # Calculate encoder steps since last capture
        encoder_diff = abs(self.encoder_value - self.last_encoder_value_captured)

        # Check if encoder has moved enough steps
        if encoder_diff >= steps_threshold:
            # Also check analog threshold if enabled (>= 0)
            if analog_threshold >= 0:
                return self.analog_value >= analog_threshold
            return True
        return False

    def clear_signal(self):
        self.last_encoder_value_captured = self.encoder_value

    def queue_ejection(self, dm=None):
        """Add an ejection target to the queue based on current encoder + offset"""
        if not EJECTOR_ENABLED:
            logger.info(f"Ejector disabled, ignoring ejection request for dm={dm}")
            return
        target_encoder = self.encoder_value + EJECTOR_OFFSET
        self.ejection_queue.append({
            "target": target_encoder,
            "dm": dm,
            "queued_at": self.encoder_value
        })

    def run(self):
        """
        Read and parse serial data from the watcher device.

        Supports two modes (configured via SERIAL_MODE env variable):
            - 'legacy':  Arduino prints lines like
                'Encoder:123,Red:0,Green:0,Blue:0,Color:0,'
            - 'new':     Firmware prints CSV key/value pairs like
                'ENC:123,OKC:0,NGC:0,DWS:0,PPS:10,PPM:600,...'
        """
        buffer = ""
        MAX_BUFFER_SIZE = 1024

        while not self.stop_thread:
            try:
                # If serial is not available, just sleep and continue (camera-only mode)
                if not self.serial_available or not self.serial:
                    time.sleep(0.1)
                    continue

                if self.serial.inWaiting():
                    raw_bytes = self.serial.read(self.serial.inWaiting())
                    try:
                        chunk = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        # Drop undecodable garbage and continue
                        buffer = ""
                        time.sleep(0.03)
                        continue

                    buffer += chunk

                    # Avoid unbounded growth
                    if len(buffer) > MAX_BUFFER_SIZE:
                        logger.warning("Serial buffer exceeded max size, clearing")
                        buffer = ""

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        parsed_legacy = False

                        # Legacy format: starts with/contains 'Encoder:' prefix
                        if "Encoder:" in line:
                            try:
                                self.last_encoder_value = self.encoder_value
                                encoder_first_string_index = line.index("Encoder:")
                                # Legacy lines look like:
                                # 'Encoder:123,Red:0,Green:0,Blue:0,Color:0,'
                                # We only care about the encoder number.
                                encoder_last_string_index = line.find(",", encoder_first_string_index)
                                if encoder_last_string_index == -1:
                                    encoder_last_string_index = len(line)
                                self.encoder_value = int(
                                    line[encoder_first_string_index + len("Encoder:"):encoder_last_string_index]
                                )

                                self.data = {
                                    "encoder_value": self.encoder_value,
                                }
                                parsed_legacy = True
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse legacy line '{line}': {parse_err}")
                                continue

                        # New format: contains ENC/OKC/NGC/DWS style keys
                        elif any(key in line for key in ("ENC:", "OKC:", "NGC:", "DWS:")):
                            try:
                                # Remove trailing comma and split
                                clean = line.rstrip(",")
                                pairs = clean.split(",")
                                kv = {}
                                for pair in pairs:
                                    if ":" in pair:
                                        k, v = pair.split(":", 1)
                                        k = k.strip()
                                        v = v.strip()
                                        if not v or not v.replace("-", "").isdigit():
                                            continue
                                        kv[k] = int(v)

                                if "ENC" not in kv:
                                    continue

                                self.last_encoder_value = self.encoder_value
                                self.encoder_value = kv["ENC"]

                                # Extended counters and metrics (if provided)
                                self.ok_counter = kv.get("OKC", self.ok_counter)
                                self.ng_counter = kv.get("NGC", self.ng_counter)
                                self.downtime_seconds = kv.get("DWS", self.downtime_seconds)
                                self.pulses_per_second = kv.get("PPS", self.pulses_per_second)
                                self.pulses_per_minute = kv.get("PPM", self.pulses_per_minute)
                                # Extra metrics
                                self.analog_value = kv.get("ANG", self.analog_value)
                                self.power_value = kv.get("PWR", self.power_value)

                                # Status bits (STS) – same semantics as watcher service
                                if "STS" in kv:
                                    self.status_value = kv["STS"]
                                    self.u_status = bool(self.status_value & 0x01)   # Bit 0: U
                                    self.b_status = bool(self.status_value & 0x02)   # Bit 1: B
                                    # Bit 2: WARNING (active-low in watcher code)
                                    self.warning_status = not bool(self.status_value & 0x04)

                                # Verbose configuration values (if present)
                                # OK configuration
                                self.ok_offset_delay = kv.get("OOD", self.ok_offset_delay)
                                self.ok_duration_pulses = kv.get("ODP", self.ok_duration_pulses)
                                self.ok_duration_percent = kv.get("ODL", self.ok_duration_percent)
                                self.ok_encoder_factor = kv.get("OEF", self.ok_encoder_factor)
                                # NG configuration
                                self.ng_offset_delay = kv.get("NOD", self.ng_offset_delay)
                                self.ng_duration_pulses = kv.get("NDP", self.ng_duration_pulses)
                                self.ng_duration_percent = kv.get("NDL", self.ng_duration_percent)
                                self.ng_encoder_factor = kv.get("NEF", self.ng_encoder_factor)
                                # System configuration
                                self.external_reset = kv.get("EXT", self.external_reset)
                                self.baud_rate = kv.get("BUD", self.baud_rate)
                                self.downtime_threshold = kv.get("DWT", self.downtime_threshold)

                                # Speed & movement (use PPS/PPM if available)
                                pps = self.pulses_per_second
                                self.is_moving = pps != 0
                                if self.is_moving:
                                    self.last_move_time = time.time()

                                self.data = {
                                    "encoder_value": self.encoder_value,
                                }
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse new-mode line '{line}': {parse_err}")
                                continue
                        else:
                            # Unknown format, skip line
                            continue

                        # Common post-parse state updates
                        # Movement based on PPS (pulses per second) - more reliable than encoder comparison
                        if self.pulses_per_second > 0:
                            self.is_moving = True
                            self.last_move_time = time.time()
                        else:
                            self.is_moving = False

                        if self.encoder_value - self.last_capture_encoder > self.step:
                            self.take = True
                        else:
                            self.take = False

                        self.health_check = True
                        tmp_time = time.time()

                        if tmp_time - self.last_d_or_k > 1:
                            self.d_or_k = 0
                            self.last_d_or_k = tmp_time

                time.sleep(0.03)
            except Exception as e:
                self.health_check = False
                # Skip reconnection attempts if serial was never available
                if not self.serial_available:
                    time.sleep(0.1)
                    continue

                if "Input/output" in str(e):
                    try:
                        if self.serial:
                            self.serial.close()
                        self.serial = serial.Serial(self.serial_port, self.serial_baudrate, 8, 'N', 1, timeout=1)
                        self.serial.flushInput()  # clear input serial buffer
                        self.serial.flushOutput()  # clear output serial buffer
                        self.serial_available = True
                        logger.warning(f"Serial reconnected after I/O error: {e}")
                    except Exception as er:
                        self.serial_available = False
                        logger.error(f"Serial reconnection failed: {er}")
                if "fileno" in str(e):
                    try:
                        self.serial = serial.Serial(self.serial_port, self.serial_baudrate, 8, 'N', 1, timeout=1)
                        self.serial.flushInput()  # clear input serial buffer
                        self.serial.flushOutput()  # clear output serial buffer
                        self.serial_available = True
                    except Exception as ers:
                        self.serial_available = False
                        logger.error(f"Serial reconnection failed: {ers}")
                logger.error(f"Serial run error: {e}")
                time.sleep(0.1)

    def run_ejector(self):
        """
        Encoder-based ejector using Redis list queue.

        External service pushes to 'ejector_queue' with format:
            {"encoder": 150, "dm": "ABC123"}  (JSON string)

        This method pops from queue, calculates target (encoder + offset),
        and ejects when current encoder reaches target.
        """
        while not self.stop_thread:
            try:
                # If ejector is globally disabled, ensure it's stopped and ignore any queues
                if not EJECTOR_ENABLED:
                    if self.ejector_running:
                        self._send_message('7\n')
                        self.ejector_running = False
                    # Clear any pending ejection requests
                    self.ejection_queue.clear()
                    time.sleep(EJECTOR_POLL_INTERVAL)
                    continue

                # Pop new ejection requests from Redis list and add to local queue
                raw_ejector_data = self.redis_connection.pop_queue_messages_redis(stream_name="ejector_queue")
                if raw_ejector_data:
                    try:
                        ejector_data = json.loads(raw_ejector_data.decode('utf-8'))
                        capture_encoder = ejector_data.get("encoder", self.encoder_value)
                        dm = ejector_data.get("dm", None)
                        target_encoder = capture_encoder + EJECTOR_OFFSET
                        self.ejection_queue.append({
                            "target": target_encoder,
                            "dm": dm,
                            "queued_at": capture_encoder
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid ejector queue data: {e}")

                # Process ejection queue based on encoder position
                if self.ejection_queue:
                    # Check if we've reached the target encoder value for the first item
                    if self.encoder_value >= self.ejection_queue[0]["target"]:
                        # Start ejector
                        if not self.ejector_running:
                            self._send_message('6\n')
                            self.ejector_running = True
                            self.ejector_start_ts = time.time()
                            self.redis_connection.update_queue_messages_redis("Eject", stream_name="speaker")
                            self.ejection_queue.pop(0)

                # Stop ejector after EJECTOR_DURATION
                if self.ejector_running and (time.time() - self.ejector_start_ts > EJECTOR_DURATION):
                    self._send_message('7\n')
                    self.ejector_running = False

                time.sleep(EJECTOR_POLL_INTERVAL)

            except Exception as e:
                logger.error(f"Run ejector failed: {e}")
    
    def capture_frames(self):
        """Capture frames when a signal is detected."""
        
        light_sleep = True
        while True:
            try:
                if self.signal_captured():
                    self.clear_signal()
                    capture_timestamp = time.time()
                    d = str(datetime.now()).replace('.', "-").replace(':', "-").replace(' ', "-")
                    frames = []
                    try:
                        self.shipment = self.redis_connection.get_redis("shipment")
                        self.shipment = self.shipment.decode('utf-8') if self.shipment else "no_shipment"
                        if self.shipment != self.old_shipment:
                            os.makedirs("raw_images/" + self.shipment, exist_ok=True)
                            self.old_shipment = self.shipment
                    except Exception as e:
                        self.shipment = "no_shipment"

                    # Execute capture based on StateManager configuration
                    grabbed_frames = []
                    start_save = time.time()

                    # Get current state from state_manager
                    current_state = state_manager.current_state if state_manager else None

                    if current_state and current_state.enabled:
                        # Track last light mode for open-loop optimization
                        last_light_mode = None
                        num_phases = len(current_state.phases)

                        # Execute each capture phase from state configuration
                        for phase_idx, phase in enumerate(current_state.phases):
                            # Only send command if light mode changed between phases
                            if phase.light_mode != last_light_mode:
                                self._set_light_mode(phase.light_mode)
                                logger.info(f"Phase {phase_idx+1}/{num_phases}: {phase.light_mode}")
                                last_light_mode = phase.light_mode

                            # Wait for configured delay
                            if phase.delay > 0:
                                time.sleep(phase.delay)

                            # Clear signal before capture
                            self.clear_signal()

                            # Capture cameras specified in this phase
                            for cam_id in phase.cameras:
                                cam = self.cameras.get(cam_id)
                                if cam and cam.success:
                                    # Track capture timestamp for FPS calculation
                                    capture_ts = time.time()
                                    grabbed_frames.append((cam_id, cam.read()))

                                    # Store capture timestamp in Redis for FPS tracking
                                    try:
                                        if self.redis_connection:
                                            self.redis_connection.redis_connection.lpush("capture_timestamps", str(capture_ts))
                                            self.redis_connection.redis_connection.ltrim("capture_timestamps", 0, 9)  # Keep last 10
                                    except Exception as e:
                                        logger.debug(f"Failed to track capture FPS: {e}")
                    else:
                        # No fallback - StateManager must be properly configured
                        logger.error("StateManager not available or state disabled - no capture performed")

                    # Second loop - save captured frames
                    for camera_index, grabbed in grabbed_frames:  # Changed to match first loop
                        d_path = f"{self.shipment}/{d}_{camera_index}"
                        name = os.path.join("raw_images", f"{d_path}.jpg")
                        cv2.imwrite(name, grabbed)
                        frames.append([d_path, None])

                    if (capture_timestamp - self.last_capture_ts > time_between_two_package):
                        self.last_capture_ts = capture_timestamp  # Use the same timestamp
                        # Push the entire list of frames to Redis with encoder value
                        frames_data = {
                            "frames": frames,
                            "encoder": self.encoder_value,
                            "shipment": self.shipment
                        }
                        self.redis_connection.update_queue_messages_redis(json.dumps(frames_data), stream_name="frames_queue")
            except Exception as e:
                logger.error(f"Capture error: {e}")
                # Restart all cameras dynamically
                for cam_id, cam in self.cameras.items():
                    if cam and hasattr(cam, 'restart_camera'):
                        try:
                            cam.restart_camera()
                        except Exception as restart_err:
                            logger.warning(f"Failed to restart camera {cam_id}: {restart_err}")
                time.sleep(0.01)


                
    def off_if_not_moving(self):
        if abs(time.time() - self.last_move_time) > 1:
            self.off()


    def update_data_on_redis(self, frame_number, mode,buffer, gap, data_gathering):
        data = self.data
        data['light_mode'] = mode
        data['frame_number'] = frame_number
        data['buffer'] = buffer
        data['gap'] = gap
        data['d_or_k'] = self.d_or_k
        data['data_gathering'] = data_gathering
        self.redis_connection.update_encoder_redis(data)
    
    def send_queue_messages(self, queue_messages):
        self.redis_connection.update_queue_messages_redis(queue_messages)
    
    def get_queue_messages(self, stream_name):
        return self.redis_connection.pop_queue_messages_redis(stream_name)

    def stream_results(self):
        # Target height for stream concatenation (all frames will be resized to this height)
        TARGET_STREAM_HEIGHT = 720

        while not self.stop_thread:
            try:
                # Fetch all frames from Redis queue
                self.raw_stream_queue = self.get_queue_messages(stream_name="stream_queue")
                if self.raw_stream_queue:
                    self.stream_data = json.loads(self.raw_stream_queue)
                    stream_image = None
                    self.stream_histogram_data = []
                    remove_raw_image = remove_raw_image_when_dm_decoded
                    for stream_frame in self.stream_data:
                        frame_histogram_data = []
                        stream_path = os.path.join("raw_images", f"{stream_frame[0]}.jpg")
                        stream_frame[1] = cv2.imread(stream_path)
                        # Annotate frame_data[1] with bounding boxes and labels
                        for idx, res in enumerate(stream_frame[2]):
                            cv2.rectangle(stream_frame[1], (int(res['xmin']), int(res['ymin'])),(int(res['xmax']), int(res['ymax'])), (100, 150, 250), 4)
                            text = f"{res['name']} {res['confidence']:.2f}"
                            font = cv2.FONT_HERSHEY_COMPLEX
                            font_scale = 1
                            font_color = (255, 255, 255)
                            thickness = 1
                            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                            text_w, text_h = text_size
                            x, y = int(res['xmin']), int(res['ymin'])
                            bottom_left = (x, y)
                            cv2.rectangle(stream_frame[1], bottom_left, (x + text_w, y - text_h), (120, 120, 120), -1)
                            cv2.putText(stream_frame[1], text, bottom_left, font, font_scale, font_color, thickness)

                            # Calculate and save histogram for each detected object (if enabled)
                            if HISTOGRAM_ENABLED:
                                roi = stream_frame[1][int(res['ymin']):int(res['ymax']), int(res['xmin']):int(res['xmax'])]
                                if roi.size > 0:
                                    hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])
                                    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
                                    hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])

                                    # Create and save histogram visualization image (if enabled)
                                    if HISTOGRAM_SAVE_IMAGE:
                                        hist_height = 200
                                        hist_width = 256
                                        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
                                        hist_r_norm = hist_r.copy()
                                        hist_g_norm = hist_g.copy()
                                        hist_b_norm = hist_b.copy()
                                        cv2.normalize(hist_r_norm, hist_r_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        cv2.normalize(hist_g_norm, hist_g_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        cv2.normalize(hist_b_norm, hist_b_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        for i in range(1, 256):
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_r_norm[i-1])),
                                                    (i, hist_height - int(hist_r_norm[i])), (0, 0, 255), 1)
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_g_norm[i-1])),
                                                    (i, hist_height - int(hist_g_norm[i])), (0, 255, 0), 1)
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_b_norm[i-1])),
                                                    (i, hist_height - int(hist_b_norm[i])), (255, 0, 0), 1)

                                        # Save histogram image
                                        name_hist_obj = os.path.join("raw_images", f"{stream_frame[0]}_obj{idx}_{res['name']}_hist.jpg")
                                        cv2.imwrite(name_hist_obj, hist_image)

                                    # Convert histograms to list of 256 ints for JSON serialization
                                    hist_r_list = [int(x) for x in hist_r.flatten()]
                                    hist_g_list = [int(x) for x in hist_g.flatten()]
                                    hist_b_list = [int(x) for x in hist_b.flatten()]

                                    frame_histogram_data.append({
                                        "frame": stream_frame[0],
                                        "id": int(stream_frame[0].split('_')[-1]) if '_' in stream_frame[0] else 0,
                                        "obj": int(idx),
                                        "name": res['name'],
                                        "bbox": {
                                            "ymin": int(res['ymin']),
                                            "ymax": int(res['ymax']),
                                            "xmin": int(res['xmin']),
                                            "xmax": int(res['xmax'])
                                        },
                                        "histogram": {
                                            "r": hist_r_list,
                                            "g": hist_g_list,
                                            "b": hist_b_list
                                        }
                                    })

                        self.stream_histogram_data.append(frame_histogram_data)

                        # Resize frame to target height for consistent concatenation (handles different ROI sizes)
                        if stream_frame[1] is not None:
                            orig_h, orig_w = stream_frame[1].shape[:2]
                            if orig_h != TARGET_STREAM_HEIGHT:
                                scale = TARGET_STREAM_HEIGHT / orig_h
                                new_w = int(orig_w * scale)
                                stream_frame[1] = cv2.resize(stream_frame[1], (new_w, TARGET_STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)

                        # Create dynamic green and red images based on target stream height
                        _, frame_green_image, frame_red_image = create_dynamic_images(TARGET_STREAM_HEIGHT)

                        for k in range(len(stream_frame) - 4):
                            if stream_frame[4 + k] is not None:
                                remove_raw_image = remove_raw_image_when_dm_decoded
                                # Concatenate green image
                                stream_image = np.concatenate((stream_image, frame_green_image), axis=1) if stream_image is not None else frame_green_image
                                # Draw a semi-transparent rectangle behind the text
                                text = stream_frame[4 + k]
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                fontScale = 1.5  # Larger font size
                                fontColor = (255, 255, 255)  # White color
                                thickness = 2
                                lineType = cv2.LINE_AA

                                # Calculate text size
                                text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
                                text_x, text_y = 10, 40  # Starting position
                                box_coords = ((text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5))

                                # Draw the rectangle
                                cv2.rectangle(stream_frame[1], box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

                                # Add text with a stroke (optional for better visibility)
                                cv2.putText(stream_frame[1], text, (text_x, text_y), font, fontScale, (0, 0, 0), thickness + 2, lineType)  # Black border
                                cv2.putText(stream_frame[1], text, (text_x, text_y), font, fontScale, fontColor, thickness, lineType)  # Inner white text

                            else:
                                remove_raw_image = False
                                name_bb = os.path.join("raw_images", f"{stream_path}_nd.jpg")
                                cv2.imwrite(name_bb, stream_frame[1])
                                stream_image = np.concatenate((stream_image, frame_red_image), axis=1) if stream_image is not None else frame_red_image

                        # Check if we should remove the raw image
                        if remove_raw_image:
                            os.remove(os.path.join("raw_images", f"{stream_path}.jpg"))

                        stream_image = np.concatenate((stream_image, stream_frame[1]), axis=1) if stream_image is not None else stream_frame[1]

                    cv2.imwrite("output.jpg", stream_image)
                    requests.post(f"http://stream:5000/send_frame_from_file/{1}", files={"file": open(f"output.jpg", "rb")})   

                
                try:
                    self.raw_mismatch_queue = self.get_queue_messages(stream_name="dms_mismatch")
                    if self.raw_mismatch_queue:
                        self.mismatch_data = json.loads(self.raw_mismatch_queue)
                        if not isinstance(self.mismatch_data, list):
                            continue

                        for mismatch_data in self.mismatch_data:
                            if not isinstance(mismatch_data, list):
                                continue

                            for mismatch_frame in mismatch_data:
                                if not isinstance(mismatch_frame, dict) or "shipment" not in mismatch_frame or "ts" not in mismatch_frame:
                                    continue

                                shipment = mismatch_frame["shipment"]
                                ts = mismatch_frame["ts"]
                                src_folder = os.path.join("raw_images", shipment)
                                dest_folder = os.path.join("raw_images", shipment, "mismatch")

                                if not os.path.exists(src_folder):
                                    continue

                                if not os.path.exists(dest_folder):
                                    os.makedirs(dest_folder, exist_ok=True)

                                for file in os.listdir(src_folder):
                                    if file.startswith(ts) and file.endswith(".jpg"):
                                        src_path = os.path.join(src_folder, file)
                                        dest_path = os.path.join(dest_folder, file)
                                        try:
                                            os.rename(src_path, dest_path)
                                        except OSError:
                                            pass

                except Exception as e:
                    logger.error(f"Error on dms_mismatch: {e}")

            except Exception as e:
                logger.error(f"Error streaming frame: {e}")

    def write_production_metrics_loop(self):
        """Background thread that periodically writes production metrics to TimescaleDB."""
        logger.info("Production metrics database writer started")
        write_interval = 60  # Write every 60 seconds

        while not self.stop_thread:
            try:
                time.sleep(write_interval)

                # Write current production metrics to database
                write_production_metrics_to_db(
                    encoder_value=self.encoder_value,
                    ok_counter=self.ok_counter,
                    ng_counter=self.ng_counter,
                    shipment=self.shipment or "no_shipment",
                    is_moving=self.is_moving,
                    downtime_seconds=self.downtime_seconds
                )

            except Exception as e:
                logger.error(f"Error writing production metrics to database: {e}")
                time.sleep(5)  # Wait before retrying


def create_dynamic_images(frame_height):
    h = frame_height # Extract height from the frame's image (frame[1] contains the image)
    blank_image = np.zeros((h, 1, 3), dtype=np.uint8)  # (height, width, channels)
    
    # Create green and red 7px width images with height from the frame
    green_image = np.zeros((h, 7, 3), dtype=np.uint8)  # (height, width, channels)
    green_image[:, :, 1] = 255  # Set green channel to full (R=0, G=255, B=0)

    red_image = np.zeros((h, 7, 3), dtype=np.uint8)
    red_image[:, :, 2] = 255  # Set red channel to full (R=255, G=0, B=0)

    return blank_image, green_image, red_image

# Function to calculate distance from a box center to object center
def calculate_distance(box, obj):
    box_center_x = (box['xmin'] + box['xmax']) / 2
    box_center_y = (box['ymin'] + box['ymax']) / 2
    obj_center_x = (obj['xmin'] + obj['xmax']) / 2
    obj_center_y = (obj['ymin'] + obj['ymax']) / 2
    return math.sqrt((box_center_x - obj_center_x) ** 2 + (box_center_y - obj_center_y) ** 2)


def calculate_iou_child(parent, child):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Parameters:
        parent (dict): Bounding box 1 with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        child (dict): Bounding box 2 with keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        float: IoU value (0 to 1), or 0 if no collision.
    """
    # Find the coordinates of the intersection rectangle
    inter_x_min = max(parent['xmin'], child['xmin'])
    inter_y_min = max(parent['ymin'], child['ymin'])
    inter_x_max = min(parent['xmax'], child['xmax'])
    inter_y_max = min(parent['ymax'], child['ymax'])

    # Check if there is an intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        # parent_area = (parent['xmax'] - parent['xmin']) * (parent['ymax'] - parent['ymin'])
        child_area = (child['xmax'] - child['xmin']) * (child['ymax'] - child['ymin'])

        # IoU = Intersection Area / Union Area
        # union_area = parent_area + child_area - inter_area
        return inter_area / child_area
    else:
        return 0.0

def otsu_threshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def adaptive_gaussian(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def adaptive_mean(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def clahe_threshold(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.threshold(clahe.apply(img), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def preprocess_and_decode(stage_name, preprocess_func, cropped_dm_gray, dm_chars_sizes):
    """Helper function to apply preprocessing and decode the image."""
    try:
        processed_image = preprocess_func(cropped_dm_gray)
        decoded_dm_raw = pylibdmtx.decode(processed_image)
        if decoded_dm_raw:
            decoded_dm = decoded_dm_raw[0][0].decode('utf-8')
            if len(decoded_dm) in dm_chars_sizes:
                return decoded_dm, processed_image, stage_name
    except Exception as e:
        return None, None, stage_name
    return None, None, stage_name

stages = [
    ("otsu", otsu_threshold),
    ("adaptive_gaussian", adaptive_gaussian),
    ("adaptive_mean", adaptive_mean),
    ("clahe", clahe_threshold),
]

def decode_dm(frame_name, frame, yolo_object, iterator, dm_chars_sizes=[13,19,26]):
    try:
        # Pre-calculate bounding box coordinates
        ymin = max(0, int(yolo_object['ymin']) - 10)
        ymax = min(frame.shape[0], int(yolo_object['ymax']) + 10)
        xmin = max(0, int(yolo_object['xmin']) - 10)
        xmax = min(frame.shape[1], int(yolo_object['xmax']) + 10)
        cropped_dm = frame[ymin:ymax, xmin:xmax]

        # Convert to grayscale
        cropped_dm_gray = cv2.cvtColor(cropped_dm, cv2.COLOR_BGR2GRAY)

        # Start decoding
        with ProcessPoolExecutor(max_workers=4) as process_executor3:
            futures = [
                process_executor3.submit(preprocess_and_decode, stage_name, preprocess, cropped_dm_gray, dm_chars_sizes)
                for stage_name, preprocess in stages
            ]
            for future in futures:
                decoded_dm, processed_image, stage_name = future.result()
                if decoded_dm:

                    # Optionally save the image for debugging
                    name_dm = os.path.join("raw_images", f"{frame_name}_{decoded_dm}_{iterator}_{stage_name}_dm.jpg")
                    cv2.imwrite(name_dm, processed_image)
                    return decoded_dm

        return None

    except Exception as e:
        logger.error(f"Decoding failed: {frame_name} - {e}")
        return None

def nest_objects(yolo_results, custom_parent_list=None, overlap_threshold=0.6):
    """
    Nest objects detected by YOLO based on spatial hierarchy.

    Parameters:
        yolo_results (list): List of YOLO detection results, each containing:
            - 'name': The detected object's class name
            - 'xmin': The minimum x-coordinate of the bounding box
            - 'ymin': The minimum y-coordinate of the bounding box
            - 'xmax': The maximum x-coordinate of the bounding box
            - 'ymax': The maximum y-coordinate of the bounding box
            - 'confidence': The confidence score of the detection.

        custom_parent_list (list): List of object names considered as parents.
                                   If None, uses global PARENT_OBJECT_LIST.

    Returns:
        dict: Nested structure of objects with parents ("box", "pack") and their children.
               If no parents found, all objects are nested under a single virtual "_root" parent.
    """
    try:
        # Use global PARENT_OBJECT_LIST if no custom list provided
        parent_list = custom_parent_list if custom_parent_list is not None else PARENT_OBJECT_LIST
        # Filter out empty strings from parent list
        effective_parent_list = [p for p in parent_list if p.strip()]

        parents = [obj for obj in yolo_results if obj['name'] in effective_parent_list]
        children = [obj for obj in yolo_results if obj['name'] not in effective_parent_list]

        # Create a nested dictionary to store results
        nested_results = {'parents': []}

        # If no parents detected, nest all objects under virtual _root parent
        if not parents:
            if yolo_results:
                all_objects = list(yolo_results)
                nested_results['parents'].append({
                    'name': '_root',
                    'xmin': 0,
                    'ymin': 0,
                    'xmax': 0,
                    'ymax': 0,
                    'confidence': 1.0,
                    'children': all_objects
                })
            return nested_results

        # Normal parent-child nesting logic
        for parent in parents:
            parent_children = []

            # Check if children are within the parent bounding box
            for child in children:
                if calculate_iou_child(parent, child) > overlap_threshold:
                    parent_children.append(child)

            # Add parent and its children to the nested structure
            nested_results['parents'].append({
                'name': parent['name'],
                'xmin': parent['xmin'],
                'ymin': parent['ymin'],
                'xmax': parent['xmax'],
                'ymax': parent['ymax'],
                'confidence': parent['confidence'],
                'children': parent_children
            })

        return nested_results

    except Exception as e:
        logger.error(f"Nesting error: {e}")
        return None


def decode_objects(frame_name, frame, nested_object, iterator=0, query_data=None, confidence_threshold=0.8, overlap_threshold=0.2, dm_chars_sizes=[13,19,26]):
    """
    Decode multiple DataMatrix objects inside nested objects and handle fallback logic based on query data.

    Parameters:
        frame_name (str): Name of the current frame.
        frame (object): The current frame or image data.
        nested_object (dict): The nested structure containing parent-child object data.
        iterator (int): Iterator for frame processing.
        query_data (list): List of pre-defined query data for matching (uses global prepared_query_data if None).
        confidence_threshold (float): Minimum confidence required to consider a match.
        overlap_threshold (float): Minimum IOU required to consider overlap.
        dm_chars_sizes (list): Minimum character length required for a valid DataMatrix.

    Returns:
        str: The decoded DataMatrix or fallback value.
    """
    # Use global prepared_query_data if query_data is not provided
    if query_data is None:
        query_data = prepared_query_data

    try:
        # Helper function to decode a DataMatrix within a region of the frame

        # Step 1: Look for a DataMatrix inside the label region
        label_list = []
        datamatrix_list = []
        sticker_list = []

        for child in nested_object['children']:
            if child['name'] == 'label':  # Find label object
                label_list.append(child)
            elif child['name'] == 'datamatrix':  # Find DataMatrix objects
                datamatrix_list.append(child)
            elif child['name'] in ['mix_st', 'nrs_st' , 'sen_st' , 'chml_st' , 'lvn_st' , 'chml_lvn_st']:
                sticker_list.append(child)

        # If both label and datamatrix exist, check if the datamatrix is inside the label
        if label_list and datamatrix_list:
            for label in label_list:
                for datamatrix in datamatrix_list:
                    overlap_iou = calculate_iou_child(label, datamatrix)
                    if overlap_iou >= overlap_threshold:
                        decoded_dm = decode_dm(frame_name, frame, datamatrix, iterator, dm_chars_sizes=dm_chars_sizes)
                        if isinstance(decoded_dm, str) and decoded_dm.isalnum():
                            return 1 , decoded_dm  # Return the first valid DataMatrix inside the label

        # Step 2: If no DataMatrix inside label, check DataMatrix inside the parent (box/pack)
        for child in nested_object['children']:            
            if ( child['name'] == 'datamatrix' and sticker_list == []):
                decoded_dm = decode_dm(frame_name, frame, child, iterator, dm_chars_sizes=dm_chars_sizes)
                if isinstance(decoded_dm, str) and decoded_dm.isalnum():
                    return 2 , decoded_dm  # Return the first valid DataMatrix inside the parent

        # Step 3: Check if there is mark/unmark
        mark_list = [child for child in nested_object['children'] if child['name'] == "mark"]
        unmark_list = [child for child in nested_object['children'] if child['name'] == "unmark"]
        if mark_list and unmark_list:
            checkboxes_distances = [
                (obj, calculate_distance(nested_object, obj), "mark") for obj in mark_list
            ] + [
                (obj, calculate_distance(nested_object, obj), "unmark") for obj in unmark_list
            ]
            checkboxes_distances.sort(key=lambda x: x[1])  # Sort by distance
            mark_index = next((i for i, checkbox in enumerate(checkboxes_distances) if checkbox[2] == "mark"), None)
            nested_object['children'].append({
                'name': str(mark_index),
                'xmin': 0,
                'ymin': 0,
                'xmax': 0,
                'ymax': 0,
                'confidence': 0.9
            })

        # Step 4: Check for fallback
        best_match = None
        highest_avg_confidence = 0

        try:
            for obj in query_data:
                # Always include parent name in matching (including _root)
                labels_to_match = [nested_object['name']] + [child['name'] for child in nested_object['children']]
                confidences_to_avg = [nested_object['confidence']] + [child['confidence'] for child in nested_object['children']]

                match = all(
                    any(
                        label in sublist_item for sublist_item in sublist
                        for label in labels_to_match
                    ) for sublist in obj["chars"]
                )

                if match:
                    matched_confidences = confidences_to_avg
                    avg_confidence = sum(matched_confidences) / len(matched_confidences) if matched_confidences else 0

                    if avg_confidence >= confidence_threshold and avg_confidence > highest_avg_confidence:
                        best_match = obj
                        highest_avg_confidence = avg_confidence

            if best_match:
                name_dm = os.path.join("raw_images", f"{frame_name}_{best_match['dm']}_{iterator}_dm.jpg")
                os.makedirs(os.path.dirname(name_dm), exist_ok=True)

                for res in nested_object['children']:
                    cv2.rectangle(frame, (int(res['xmin']), int(res['ymin'])),(int(res['xmax']), int(res['ymax'])), (100, 150, 250), 4)
                    text = f"{res['name']} {res['confidence']:.2f}"
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 1
                    font_color = (255, 255, 255)
                    thickness = 1
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    x, y = int(res['xmin']), int(res['ymin'])
                    bottom_left = (x, y)
                    cv2.rectangle(frame, bottom_left, (x + text_w, y - text_h), (120, 120, 120), -1)
                    cv2.putText(frame, text, bottom_left, font, font_scale, font_color, thickness)

                # Save the annotated image
                # For _root objects (no bounding box), save the entire frame
                if nested_object['name'] == '_root':
                    cv2.imwrite(name_dm, frame)
                else:
                    cv2.imwrite(name_dm, frame[int(nested_object['ymin']):int(nested_object['ymax']), int(nested_object['xmin']):int(nested_object['xmax'])])

                if len(best_match["dm"]) in dm_chars_sizes:
                    if sticker_list != []:
                        return 0 , best_match["dm"]
                    else:
                        return 4 , best_match["dm"]
                else:
                    return 4 , None

            return 4 , None

        except Exception as e:
            logger.error(f"Error in decode_objects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 4 , None

    except Exception as e:
        logger.error(f"Error in decode_objects: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 4 , None


watcher = ArduinoSocket(
    camera_paths=DETECTED_CAMERAS if DETECTED_CAMERAS else None,  # Auto-detect or use legacy fallback
    serial_port=WATCHER_USB,
    serial_baudrate=SERIAL_BAUDRATE
)
# Set global reference for FastAPI endpoints
watcher_instance = watcher

# Initialize state manager with watcher
state_manager = StateManager(watcher)

# Initialize pipeline manager for inference
pipeline_manager = PipelineManager()
logger.info(f"Pipeline manager initialized: current pipeline={pipeline_manager.current_pipeline.name if pipeline_manager.current_pipeline else 'none'}")

# Store in app.state for FastAPI endpoint access (similar to fabriqc-local-server pattern)
app.state.watcher = watcher
app.state.cameras = watcher.cameras  # Use dynamic cameras dict
app.state.redis_conn = watcher.redis_connection
app.state.state_manager = state_manager
app.state.pipeline_manager = pipeline_manager

# Load and apply saved configuration at startup
apply_saved_config_at_startup(watcher)

# Load saved states configuration
apply_states_from_config(state_manager)

# Load saved pipeline configuration
apply_pipeline_config_at_startup(pipeline_manager)

# Start web server in a separate thread
web_server_thread = threading.Thread(target=start_web_server, daemon=True)
web_server_thread.start()
logger.info(f"Web server started on http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")

# Inference worker thread function - must be defined before being used
def inference_worker_thread():
    """
    Dedicated thread for processing frames through inference pipeline.
    Runs independently from capture and web server threads.
    """
    logger.info("Inference worker thread started")

    while True:
        try:
            # Fetch all frames from Redis queue
            st_ts = time.time()
            raw_frames_queue = watcher.get_queue_messages(stream_name="frames_queue")
            if not raw_frames_queue:
                time.sleep(0.01)
                continue

            try:
                frames_data = json.loads(raw_frames_queue)
            except json.JSONDecodeError:
                continue

            if not frames_data:
                continue

            # Handle new format with encoder value or legacy format
            if isinstance(frames_data, dict):
                frames = frames_data.get("frames", [])
                capture_encoder = frames_data.get("encoder", 0)
                capture_shipment = frames_data.get("shipment", "no_shipment")
            else:
                # Legacy format: frames_data is the frames list directly
                frames = frames_data
                capture_encoder = 0
                capture_shipment = "no_shipment"

            # Validate frame structure before processing
            valid_frames = []
            for frame in frames:
                if isinstance(frame, list) and len(frame) > 0:
                    valid_frames.append(frame)

            if not valid_frames:
                continue

            try:
                with ProcessPoolExecutor() as process_executor:
                    # Process only valid frames
                    results = list(process_executor.map(process_frame_helper, valid_frames))

                    # Filter out None results
                    valid_results = [r for r in results if r is not None]
                    if not valid_results:
                        # Retry the entire batch once
                        results = list(process_executor.map(process_frame_helper, valid_frames))
                        valid_results = [r for r in results if r is not None]
                        if not valid_results:
                            continue

                    queue_messages, processed_frames = zip(*valid_results)

                    # Filter and collect all valid messages in one go
                    valid_queue_messages = [
                        msg for msg in queue_messages
                        if isinstance(msg.get('dms'), list) and len(msg['dms']) > 0 and msg['dms'][0] not in [None, '']
                    ]

                    if valid_queue_messages:
                        all_dms = [msg['dms'][0] for msg in valid_queue_messages]
                        most_frequent = most_frequent_string(all_dms)

                        # Sort queue messages with safe handling of None values
                        queue_messages = sorted(
                            queue_messages,
                            key=lambda x: (
                                x.get('priority', 0),
                                not isinstance(x.get('dms'), list) or len(x.get('dms', [])) == 0 or x['dms'][0] is None,
                                -len(x['dms'][0]) if isinstance(x.get('dms'), list) and len(x.get('dms', [])) > 0 and x['dms'][0] is not None else float('-inf'),
                                x['dms'][0] != most_frequent if isinstance(x.get('dms'), list) and len(x.get('dms', [])) > 0 and x['dms'][0] is not None else True
                            )
                        )

                    # Add histogram data to the first queue message if available
                    if hasattr(watcher, 'stream_histogram_data') and watcher.stream_histogram_data:
                        if len(queue_messages) > 0:
                            queue_messages[0]["extra_info"] = {"data": watcher.stream_histogram_data}
                            watcher.stream_histogram_data = []

                    if queue_messages:
                        # Add encoder value to each queue message for ejector tracking
                        for msg in queue_messages:
                            msg['encoder'] = capture_encoder
                        watcher.send_queue_messages(json.dumps(queue_messages))

                    if processed_frames:
                        watcher.redis_connection.update_queue_messages_redis(
                            json.dumps(processed_frames),
                            stream_name="stream_queue"
                        )

                    try:
                        for msg in queue_messages:
                            ts = msg.get('ts', 'N/A')

                            # DMS extraction
                            dms_list = msg.get('dms', [])
                            first_dms = dms_list[0] if isinstance(dms_list, list) and len(dms_list) > 0 else None

                            timestamp = time.time() - st_ts

                            # Detection extraction
                            detection_info = msg.get('detection', [])
                            if not isinstance(detection_info, list):
                                detection_info = []

                            # Log processed frame result
                            classes = [d.get('name', '?') for d in detection_info if isinstance(d, dict)]
                            print(f"TS:{ts} | DM:{first_dms} | {','.join(classes)} | {timestamp:.2f}s")

                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(0.1)

# Start inference worker in a separate thread
inference_thread = threading.Thread(target=inference_worker_thread, daemon=True)
inference_thread.start()
logger.info("Inference worker thread started")

# scanner = Scanner(watcher)

ejector_start_ts = time.time()
# res = requests.post(set_model_url, data={"model_path":"best.pt"})
signal_counter = 0

blank_image, green_image, red_image = create_dynamic_images(720)

# Frame Processing Function
# priortize dm based on barcode on image processing
def check_class_counts(yolo_results, confidence_threshold=None):
    """
    Check if the number of specific classes in YOLO results are all equal.
    Uses CHECK_CLASS_COUNTS_CLASSES from config.
    Only counts detections above the confidence threshold.
    Returns True if all specified class counts are equal, False otherwise.
    """
    target_classes = CHECK_CLASS_COUNTS_CLASSES
    conf_threshold = confidence_threshold if confidence_threshold is not None else CHECK_CLASS_COUNTS_CONFIDENCE

    # Filter by confidence threshold and count classes
    class_counts = Counter(
        obj['name'] for obj in yolo_results
        if obj['name'] in target_classes and obj.get('confidence', 1.0) >= conf_threshold
    )

    # If no target classes found, return True (nothing to check)
    if not class_counts:
        return True

    # Ensure all target classes are present (missing = count 0)
    for cls in target_classes:
        if cls not in class_counts:
            class_counts[cls] = 0

    # Check if all counts are equal
    counts = list(class_counts.values())
    all_equal = len(set(counts)) == 1  # All counts are equal if set has only 1 unique value

    return all_equal

def process_frame(frame, capture_mode):
    try:
        global pipeline_manager

        frame_id , _ = frame
        frame_path = os.path.join("raw_images", f"{frame_id}.jpg")
        image = cv2.imread(frame_path)

        img_encoded = cv2.imencode('.jpg', image)[1]
        img_bytes = img_encoded.tobytes()

        # Use pipeline manager if available, otherwise fallback to legacy req_predict
        start_time = time.time()
        model_name_used = "unknown"

        if pipeline_manager:
            yolo_res, model_name_used = pipeline_manager.run_inference(img_bytes)
        else:
            # Fallback to legacy inference
            yolo_res = json.loads(req_predict(img_bytes))
            model_name_used = "legacy"

        processing_time = time.time() - start_time

        # Cache frame_id for video feed to display processed image
        global watcher_instance
        if watcher_instance:
            watcher_instance.latest_frame_id = frame_id

        # Save annotated image with bounding boxes
        if yolo_res and len(yolo_res) > 0:
            annotated_image = image.copy()
            for det in yolo_res:
                try:
                    x1 = int(det.get('xmin', 0))
                    y1 = int(det.get('ymin', 0))
                    x2 = int(det.get('xmax', 0))
                    y2 = int(det.get('ymax', 0))
                    confidence = det.get('confidence', 0)
                    name = det.get('name', f"Class {det.get('class', 0)}")

                    # Draw thick green rectangle
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Draw label with background
                    label = f"{name} {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                except Exception as e:
                    logger.error(f"Error drawing detection on {frame_id}: {e}")

            # Save annotated image with _DETECTED suffix
            annotated_path = os.path.join("raw_images", f"{frame_id}_DETECTED.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            logger.info(f"Saved annotated image: {annotated_path} with {len(yolo_res)} detections (model: {model_name_used})")

            # Write inference result to database with model name
            if watcher:
                inference_time_ms = int(processing_time * 1000)
                write_inference_to_db(
                    shipment=watcher.shipment or "no_shipment",
                    image_path=annotated_path,
                    detections=yolo_res,
                    inference_time_ms=inference_time_ms,
                    model_used=model_name_used
                )

        frame_data = [frame_id, image, yolo_res, 5]

        if yolo_res:
            # Check if specific class counts are equal (only if enabled)
            if CHECK_CLASS_COUNTS_ENABLED and not check_class_counts(yolo_res):
                frame_data.append(None)
                frame_data[1] = None  # make the frame light by deleting image part
                queue_message = {
                    "ts": frame_data[0],
                    "dms": frame_data[4:],
                    "priority": frame_data[3],
                    "detection": frame_data[2],
                    "shipment": watcher.shipment
                }
                return queue_message, frame_data

            nested = nest_objects(yolo_res)
            q = 0

            if capture_mode == "multiple":

                for n in nested["parents"]:
                    priority , dm = decode_objects(frame_id, frame_data[1], n, q)
                    q += 1
                    frame_data[3] = priority
                    frame_data.append(dm)

            elif capture_mode == "single":

                dm_polling_list = []
                for n in nested["parents"]:
                    priority , dm = decode_objects(frame_id, frame_data[1], n, q)
                    q = q + 1
                    dm_polling_list.append({ "dm" : dm, "priority" : priority , "x_center" : (n["xmin"]+n["xmax"])/2, "area" : (n["xmax"] - n["xmin"])*(n["ymax"] - n["ymin"]) } )
                filtered_dm_polling_list = [obj for obj in dm_polling_list if obj["dm"] is not None]

                if filtered_dm_polling_list:
                    overall_center = sum(obj["x_center"] for obj in filtered_dm_polling_list) / len(filtered_dm_polling_list)
                    filtered_dm_polling_list.sort(key=lambda obj: (abs(obj["x_center"] - overall_center), -obj["area"]))
                    largest_area_object = max(filtered_dm_polling_list, key=lambda obj: obj["area"], default=None)
                    # Determine the result
                    if largest_area_object["dm"] is not None:
                        frame_data.append(largest_area_object["dm"])  # Return the largest area object's "dm"
                        frame_data[3] = largest_area_object["priority"]
                    else:
                        # Fallback to the most centered object's "dm"
                        most_centered_object = filtered_dm_polling_list[0]
                        if most_centered_object["dm"] is not None:
                            frame_data.append(most_centered_object["dm"])
                            frame_data[3] = most_centered_object["priority"]
                        else:
                            # Fallback to the first non-None "dm" in the list
                            first_non_none_object = next((obj for obj in filtered_dm_polling_list if obj["dm"] is not None), None)
                            frame_data.append(first_non_none_object["dm"] if first_non_none_object else None)
                            frame_data[3] = first_non_none_object["priority"]
                else:
                    frame_data.append(None)
        frame_data[1] = None # make the frame light by deleteing image part
        queue_message = {
            "ts": frame_data[0],
            "dms": frame_data[4:],
            "priority": frame_data[3],
            "detection" : frame_data[2],
            "shipment": watcher.shipment
        }

        return queue_message, frame_data

    except Exception as e:
        logger.error(f"Error processing frame {frame_id}: {e}")
        return None


def most_frequent_string(dms_list):
    """Find the most frequent string in the list of 'dms[0]'."""
    counter = Counter(dms_list)
    return counter.most_common(1)[0][0] if counter else None


def process_frame_helper(frame):
    capture_mode = "single"  # or "multiple"
    return process_frame(frame, capture_mode)


# Main thread - keep application alive
# All work is done in separate threads (uvicorn, inference, capture, etc.)
logger.info('All worker threads started - main thread entering keep-alive loop')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info('Shutting down...')

