"""ArduinoSocket — central coordinator for serial, camera, and Redis communication.

Manages:
- Serial communication with Arduino/watcher hardware
- Camera capture and frame management
- Redis-based frame queue for inference pipeline
- Ejector control based on encoder position
- Stream results visualization
- Production metrics to TimescaleDB
- Barcode scanner input
"""

import cv2
import time
import json
import os
import queue
import numpy as np
import serial
import requests
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List

import config as cfg
from services.camera import (
    CameraBuffer, DETECTED_CAMERAS,
    CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH,
    apply_camera_config_from_saved,
)
from redis import Redis as DirectRedis
from services.redis_service import RedisConnection
from services.db import write_production_metrics_to_db
from services.detection import create_dynamic_images

logger = logging.getLogger(__name__)

# Module-level runtime references (set from main.py after initialization)
_state_manager = None
_app = None

# Background disk write queue — cv2.imwrite runs here to unblock capture loop
_disk_queue = queue.Queue(maxsize=2000)
_disk_writers_count = 0
_disk_lock = threading.Lock()

def _disk_writer_loop():
    """Background thread that saves images to disk without blocking capture."""
    while True:
        try:
            path, frame = _disk_queue.get()
            cv2.imwrite(path, frame)
        except Exception as e:
            logger.error(f"Disk write error: {e}")

def add_disk_writers(count: int):
    """Add disk writer threads. Thread-safe, callable at any time by autoscaler.

    Args:
        count: Number of NEW threads to add (not total target).
    """
    global _disk_writers_count
    if count <= 0:
        return
    with _disk_lock:
        start_idx = _disk_writers_count
        for i in range(count):
            threading.Thread(
                target=_disk_writer_loop, daemon=True,
                name=f"disk-writer-{start_idx + i + 1}"
            ).start()
        _disk_writers_count += count
        logger.info(f"[DiskWriters] +{count} threads, total now {_disk_writers_count}")


# Persistent Redis connection for capture FPS timestamps (db=0, reused across captures)
_cap_redis = None

def _get_cap_redis():
    """Get or create persistent Redis connection for capture timestamps."""
    global _cap_redis
    if _cap_redis is None:
        _cap_redis = DirectRedis("redis", 6379, db=0)
    return _cap_redis


def set_state_manager(sm):
    """Set the state manager reference (called from main.py after both are created)."""
    global _state_manager
    _state_manager = sm


def set_app(app):
    """Set the FastAPI app reference (called from main.py)."""
    global _app
    _app = app


class ArduinoSocket:
    def __init__(self, camera_paths: List[str] = None, serial_port=None, serial_baudrate=None):
        """Initialize ArduinoSocket with dynamic camera support.

        Args:
            camera_paths: List of video device paths to use. If None, uses auto-detected cameras.
            serial_port: Serial port path (default from env)
            serial_baudrate: Serial baud rate (default from env)
        """
        # Use configurable serial settings
        self.serial_port = serial_port or cfg.WATCHER_USB
        self.serial_baudrate = serial_baudrate or cfg.SERIAL_BAUDRATE
        self.serial_mode = cfg.SERIAL_MODE
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
        self.redis_connection = RedisConnection(cfg.REDIS_HOST, cfg.REDIS_PORT)
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

        # Start disk writer threads — conservative initial count, autoscaler ramps up
        add_disk_writers(max(2, len(self.cameras)))

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

        If cfg.LIGHT_STATUS_CHECK_ENABLED is True (closed-loop):
            Checks actual serial status before sending command.
            Only sends command if current status differs from requested mode.

        If cfg.LIGHT_STATUS_CHECK_ENABLED is False (open-loop):
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
        if not force and cfg.LIGHT_STATUS_CHECK_ENABLED and self.serial_available:
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
        current_state = _state_manager.current_state if _state_manager else None

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
        if not cfg.EJECTOR_ENABLED:
            logger.info(f"Ejector disabled, ignoring ejection request for dm={dm}")
            return
        target_encoder = self.encoder_value + cfg.EJECTOR_OFFSET
        self.ejection_queue.append({
            "target": target_encoder,
            "dm": dm,
            "queued_at": self.encoder_value
        })

    def run(self):
        """
        Read and parse serial data from the watcher device.

        Supports two modes (configured via cfg.SERIAL_MODE env variable):
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
                if not cfg.EJECTOR_ENABLED:
                    if self.ejector_running:
                        self._send_message('7\n')
                        self.ejector_running = False
                    # Clear any pending ejection requests
                    self.ejection_queue.clear()
                    time.sleep(cfg.EJECTOR_POLL_INTERVAL)
                    continue

                # Pop new ejection requests from Redis list and add to local queue
                raw_ejector_data = self.redis_connection.pop_queue_messages_redis(stream_name="ejector_queue")
                if raw_ejector_data:
                    try:
                        ejector_data = json.loads(raw_ejector_data.decode('utf-8'))
                        capture_encoder = ejector_data.get("encoder", self.encoder_value)
                        dm = ejector_data.get("dm", None)
                        target_encoder = capture_encoder + cfg.EJECTOR_OFFSET
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

                # Stop ejector after cfg.EJECTOR_DURATION
                if self.ejector_running and (time.time() - self.ejector_start_ts > cfg.EJECTOR_DURATION):
                    self._send_message('7\n')
                    self.ejector_running = False

                time.sleep(cfg.EJECTOR_POLL_INTERVAL)

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
                        os.makedirs("raw_images/" + self.shipment, exist_ok=True)
                        if self.shipment != self.old_shipment:
                            self.old_shipment = self.shipment
                    except Exception as e:
                        self.shipment = "no_shipment"

                    # Execute capture based on StateManager configuration
                    grabbed_frames = []
                    start_save = time.time()

                    # Get current state from _state_manager
                    current_state = _state_manager.current_state if _state_manager else None

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
                                    capture_ts = time.time()
                                    frame = cam.read()
                                    grabbed_frames.append((cam_id, frame))

                                    # Track capture FPS via persistent Redis connection (db=0)
                                    try:
                                        cr = _get_cap_redis()
                                        cr.lpush("capture_timestamps", str(capture_ts))
                                        cr.ltrim("capture_timestamps", 0, 9)
                                    except Exception:
                                        globals()['_cap_redis'] = None  # Reset, will reconnect
                    else:
                        # No fallback - StateManager must be properly configured
                        logger.error("StateManager not available or state disabled - no capture performed")

                    # Second loop - queue frames for async disk write (non-blocking)
                    for camera_index, grabbed in grabbed_frames:
                        d_path = f"{self.shipment}/{d}_{camera_index}"
                        name = os.path.join("raw_images", f"{d_path}.jpg")
                        try:
                            _disk_queue.put_nowait((name, grabbed))
                        except queue.Full:
                            logger.warning("Disk write queue full — writing synchronously")
                            cv2.imwrite(name, grabbed)
                        frames.append([d_path, None])

                    if (capture_timestamp - self.last_capture_ts > cfg.time_between_two_package):
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
                    remove_raw_image = cfg.remove_raw_image_when_dm_decoded
                    for stream_frame in self.stream_data:
                        frame_histogram_data = []
                        stream_path = os.path.join("raw_images", f"{stream_frame[0]}.jpg")
                        stream_frame[1] = cv2.imread(stream_path)
                        # Annotate frame_data[1] with bounding boxes and labels (filtered by object_filters)
                        _tl_cfg = getattr(_app.state, 'timeline_config', {})
                        _obj_filters = _tl_cfg.get('object_filters', {})
                        _skip_bbox = not _tl_cfg.get('show_bounding_boxes', True)
                        for idx, res in enumerate(stream_frame[2]):
                            if _skip_bbox:
                                break
                            _of = _obj_filters.get(res.get('name', ''), {})
                            if _of.get('show') is False:
                                continue
                            if res.get('confidence', 0) < _of.get('min_confidence', 0.01):
                                continue
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
                            if cfg.HISTOGRAM_ENABLED:
                                roi = stream_frame[1][int(res['ymin']):int(res['ymax']), int(res['xmin']):int(res['xmax'])]
                                if roi.size > 0:
                                    hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])
                                    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
                                    hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])

                                    # Create and save histogram visualization image (if enabled)
                                    if cfg.HISTOGRAM_SAVE_IMAGE:
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
                                remove_raw_image = cfg.remove_raw_image_when_dm_decoded
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
