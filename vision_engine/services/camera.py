import cv2
import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Force RTSP over TCP to prevent half-grey frames from UDP packet loss
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Camera configuration (from environment)
IP_CAMERA_USER = os.environ.get("IP_CAMERA_USER", "admin")
IP_CAMERA_PASS = os.environ.get("IP_CAMERA_PASS", "")
IP_CAMERA_SUBNET = os.environ.get("IP_CAMERA_SUBNET", "")
IP_CAMERA_BRIGHTNESS = int(os.environ.get("IP_CAMERA_BRIGHTNESS", 128))
IP_CAMERA_CONTRAST = int(os.environ.get("IP_CAMERA_CONTRAST", 128))
IP_CAMERA_SATURATION = int(os.environ.get("IP_CAMERA_SATURATION", 128))
CAM_1_PATH = os.environ.get("CAM_1_PATH", "")
CAM_2_PATH = os.environ.get("CAM_2_PATH", "")
CAM_3_PATH = os.environ.get("CAM_3_PATH", "")
CAM_4_PATH = os.environ.get("CAM_4_PATH", "")


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
    """Apply saved configuration to camera and store for reconnect persistence."""
    if cam is None or saved_config is None:
        return

    # Apply ROI settings
    cam.roi_enabled = saved_config.get('roi_enabled', False)
    cam.roi_xmin = saved_config.get('roi_xmin', 0)
    cam.roi_ymin = saved_config.get('roi_ymin', 0)
    cam.roi_xmax = saved_config.get('roi_xmax', 1280)
    cam.roi_ymax = saved_config.get('roi_ymax', 720)

    # Apply OpenCV settings and store in _saved_props for reconnect persistence
    prop_map = {
        'exposure': cv2.CAP_PROP_EXPOSURE,
        'gain': cv2.CAP_PROP_GAIN,
        'brightness': cv2.CAP_PROP_BRIGHTNESS,
        'contrast': cv2.CAP_PROP_CONTRAST,
        'saturation': cv2.CAP_PROP_SATURATION,
        'fps': cv2.CAP_PROP_FPS,
    }
    if hasattr(cam, 'camera'):
        try:
            for key, prop in prop_map.items():
                if key in saved_config:
                    cam.camera.set(prop, saved_config[key])
                    if hasattr(cam, '_saved_props'):
                        cam._saved_props[prop] = saved_config[key]
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
        # Saved camera properties — re-applied after reconnect
        self._saved_props = {}
        self._save_current_props()
        threading.Thread(target=self.buffer).start()

    def _save_current_props(self):
        """Snapshot current camera properties so they can be re-applied after reconnect."""
        try:
            self._saved_props = {
                cv2.CAP_PROP_EXPOSURE: int(self.camera.get(cv2.CAP_PROP_EXPOSURE)),
                cv2.CAP_PROP_GAIN: int(self.camera.get(cv2.CAP_PROP_GAIN)),
                cv2.CAP_PROP_BRIGHTNESS: int(self.camera.get(cv2.CAP_PROP_BRIGHTNESS)),
                cv2.CAP_PROP_CONTRAST: int(self.camera.get(cv2.CAP_PROP_CONTRAST)),
                cv2.CAP_PROP_SATURATION: int(self.camera.get(cv2.CAP_PROP_SATURATION)),
                cv2.CAP_PROP_FPS: int(self.camera.get(cv2.CAP_PROP_FPS)),
            }
        except Exception:
            pass

    def _apply_saved_props(self):
        """Re-apply saved camera properties after reconnect."""
        if not self._saved_props:
            return
        for prop, val in self._saved_props.items():
            try:
                self.camera.set(prop, val)
            except Exception:
                pass
        logger.info(f"Re-applied saved camera config after reconnect: {self.source}")

    def update_prop(self, prop, value):
        """Update a camera property and save it for reconnect persistence."""
        self.camera.set(prop, value)
        self._saved_props[prop] = value

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
                    # Reject half-grey frames (incomplete decode)
                    # Check bottom 25% of frame for uniform grey (128) pixels
                    h = frame.shape[0]
                    bottom = frame[h * 3 // 4:, :]
                    if np.mean(bottom == 128) > 0.9:
                        success = False
                        logger.debug("Rejected grey frame (incomplete decode)")
                    else:
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

                            # Re-apply saved camera properties
                            self._apply_saved_props()

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
        # Re-apply saved camera properties
        self._apply_saved_props()
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
