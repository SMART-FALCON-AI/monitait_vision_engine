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


# ---------------------------------------------------------------------------
# 4.0.54 — Self-healing V4L2 path recovery.
#
# Problem observed on kiancord: MVE addresses cameras by /dev/videoN which
# is NOT stable across USB re-enumeration. When one physical camera drops
# and re-attaches (bad cable / hub renegotiation / firmware watchdog), the
# kernel reassigns node numbers — the surviving cameras keep their old
# numbers, the flaky one lands on a NEW pair (e.g. video10/11 → video11/12).
# The saved config still points at the OLD path, so cv2.VideoCapture opens
# a metadata sibling or a vanished node, grab() returns nothing forever,
# and the existing reconnect loop at buffer() line ~766 spins on the same
# broken path indefinitely.
#
# Recovery strategy (kept minimal, ships with no image rebuild required):
#   1. Track which V4L2 paths are currently bound by a live CameraBuffer
#      in a module-level set so the resolver never steals another cam's
#      device node.
#   2. After N consecutive reconnect cycles yield zero valid frames, probe
#      the unclaimed EVEN /dev/videoN nodes — the first one that opens AND
#      delivers a frame with non-zero geometry within ~2s replaces the
#      stale source. Metadata siblings (odd nodes by convention, or nodes
#      that open but never yield a valid frame) are filtered by the probe.
#   3. On self.stop / release, the path is unregistered so a future rebind
#      can reclaim it.
#
# This is a per-instance heal, not a global rescan — each broken camera
# fixes itself independently and other cams keep serving frames the whole
# time. If no replacement is found, the existing reconnect loop continues
# unchanged, so the failure mode is at worst identical to today's.
# ---------------------------------------------------------------------------
_V4L_BINDINGS_LOCK = threading.Lock()
_V4L_BOUND_PATHS: set = set()

# v4.0.138 — deterministic USB-renumber recovery via /dev/v4l/by-path.
# Every time we open a /dev/videoN successfully, we snapshot the current
# `/dev/v4l/by-path/…` symlink that resolves to that node. That symlink is
# tied to the physical USB bus:port, not the /dev/videoN index. When the
# same physical camera later comes back on a DIFFERENT /dev/videoN because
# a hub or another device renumbered the bus, following the SAME by-path
# lands us on the same physical camera again — no fuzzy "grab the next
# unclaimed videoN" logic, no risk of swapping cam-4's ROI onto cam-6's
# frames. Memo lives in-process only; on MVE restart it's rebuilt on the
# first successful open per camera (usually within the first frame poll).
_V4L_BY_PATH_MEMO: Dict[str, str] = {}          # last-known-good /dev/videoN → its by-path
_V4L_BY_PATH_MEMO_LOCK = threading.Lock()


def _by_path_for_video_node(video_node: str) -> Optional[str]:
    """Return the currently-active /dev/v4l/by-path/… symlink whose target
    resolves to ``video_node`` (e.g. `/dev/video7`). None when the machine
    doesn't have v4l by-path symlinks (very minimal container base) or when
    the node isn't reachable through any."""
    if not video_node or not video_node.startswith('/dev/video'):
        return None
    import glob
    try:
        target = os.path.realpath(video_node)
    except Exception:
        return None
    for link in glob.glob('/dev/v4l/by-path/*'):
        try:
            if os.path.realpath(link) == target:
                return link
        except Exception:
            continue
    return None


def _remember_by_path(video_node: str) -> Optional[str]:
    """After a successful open on ``video_node``, snapshot the by-path so
    a later USB renumber can be recovered deterministically. Returns the
    resolved by-path for the caller's log line; None when no symlink exists.
    Safe to call with any source — no-op for RTSP / basler / by-path itself.
    """
    if not video_node or not video_node.startswith('/dev/video'):
        return None
    bp = _by_path_for_video_node(video_node)
    if bp:
        with _V4L_BY_PATH_MEMO_LOCK:
            _V4L_BY_PATH_MEMO[video_node] = bp
    return bp


def _recover_via_by_path(original_video_node: str) -> Optional[str]:
    """If ``original_video_node`` has vanished and we previously remembered
    its by-path, return whatever /dev/videoN that by-path resolves to now.
    Only returns a NEW node (different from original) that:
      - exists on disk (by-path symlink still resolves),
      - is not already claimed by another live CameraBuffer.
    Callers still probe the returned node with _probe_v4l_capture_capable
    before adopting it, so a metadata-sibling swap can't sneak through."""
    with _V4L_BY_PATH_MEMO_LOCK:
        bp = _V4L_BY_PATH_MEMO.get(original_video_node)
    if not bp or not os.path.exists(bp):
        return None
    try:
        target = os.path.realpath(bp)
    except Exception:
        return None
    if not target.startswith('/dev/video') or target == original_video_node:
        return None
    with _V4L_BINDINGS_LOCK:
        if target in _V4L_BOUND_PATHS:
            return None
    return target


def _register_v4l_binding(path: str) -> None:
    """Record that ``path`` is currently held by a CameraBuffer.
    The resolver skips bound paths so it never steals another cam's node.
    Safe to call with any path — no-op for non-/dev/video sources."""
    if not path or not path.startswith('/dev/video'):
        return
    with _V4L_BINDINGS_LOCK:
        _V4L_BOUND_PATHS.add(path)


def _unregister_v4l_binding(path: str) -> None:
    """Drop ``path`` from the bound set (e.g. on release, restart, self-heal)."""
    if not path:
        return
    with _V4L_BINDINGS_LOCK:
        _V4L_BOUND_PATHS.discard(path)


def _probe_v4l_capture_capable(path: str, timeout_s: float = 2.0) -> bool:
    """Return True iff ``path`` opens and delivers a frame with non-zero
    geometry within ``timeout_s``. Filters out metadata siblings (open OK
    but read() returns success=False forever) and unpowered / wedged nodes.
    Never raises; always releases the temporary handle."""
    cap = None
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, 'size', 0) > 0:
                shape = getattr(frame, 'shape', None)
                if shape and len(shape) >= 2 and shape[0] > 0 and shape[1] > 0:
                    return True
            time.sleep(0.05)
        return False
    except Exception:
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _find_replacement_v4l_path(current_path: str) -> Optional[str]:
    """Look for a replacement /dev/videoN for a stale source.

    v4.0.138 — DETERMINISTIC FIRST: consult the by-path memo. If we opened
    ``current_path`` successfully before, we know which USB bus:port it was
    on. Following the SAME by-path lands us on the same physical camera even
    when the /dev/videoN index changed. Then AND ONLY THEN fall back to
    probing unclaimed nodes — which is the old behaviour, kept for the
    fresh-boot case where the memo is empty.

    Prefers EVEN-numbered nodes in the fallback (metadata siblings are
    conventionally odd), then falls back to odd nodes if no even candidate
    probes green.

    Returns the replacement path, or None if nothing works.
    """
    # v4.0.138 — deterministic by-path recovery. Zero risk of grabbing the
    # wrong physical camera because the bus:port is baked into the symlink.
    recovered = _recover_via_by_path(current_path)
    if recovered and _probe_v4l_capture_capable(recovered, timeout_s=2.0):
        logger.info(
            f"v4l-recover: {current_path} → {recovered} via by-path memo "
            f"(bus:port preserved, no fuzzy hunt needed)"
        )
        return recovered
    import glob
    even_candidates: List[tuple] = []
    odd_candidates: List[tuple] = []
    with _V4L_BINDINGS_LOCK:
        bound_snapshot = set(_V4L_BOUND_PATHS)
    for p in glob.glob('/dev/video*'):
        try:
            n = int(p.replace('/dev/video', ''))
        except ValueError:
            continue
        if p == current_path:
            continue
        if p in bound_snapshot:
            continue
        (even_candidates if n % 2 == 0 else odd_candidates).append((n, p))
    even_candidates.sort()
    odd_candidates.sort()
    for _, p in even_candidates + odd_candidates:
        if _probe_v4l_capture_capable(p, timeout_s=2.0):
            return p
    return None


def _load_persisted_usb_sources_ordered() -> List[str]:
    """v4.0.143 — return the list of USB camera sources from persistence,
    ordered by cam ID (1, 2, 3, ...). Each source is:
      - resolved from `/dev/v4l/by-path/…` symlink to the current /dev/videoN
        target (so downstream code that does `os.path.exists` / registration
        by raw path still works), OR
      - kept as raw /dev/videoN when persistence still has one.
    Skips cams whose source doesn't exist on disk (physically absent).
    Returns an empty list when persistence is empty / unreadable, so callers
    fall back to auto-detection.
    """
    try:
        # Lazy import: config module isn't ready at services.camera import time
        # in every entry-point (some tools import services.camera standalone).
        from config import load_service_config
        svc = load_service_config() or {}
        cams = svc.get("cameras") or {}
        if not cams:
            return []
        try:
            ordered_ids = sorted(cams.keys(), key=lambda k: int(k))
        except (ValueError, TypeError):
            ordered_ids = sorted(cams.keys())
        result = []
        for cid in ordered_ids:
            c = cams.get(cid) or {}
            ct = str(c.get("type") or "usb").lower()
            if ct != "usb":
                continue  # skip IP and pro (basler) — those are opened by other paths
            src = c.get("source")
            if not isinstance(src, str) or not src:
                continue
            if src.startswith("/dev/v4l/by-path/"):
                if not os.path.exists(src):
                    logger.warning(f"persisted USB cam {cid}: by-path missing on disk, skip: {src}")
                    continue
                # v4.0.143 — resolve the by-path symlink to its current
                # /dev/videoN target so downstream code (CameraBuffer init,
                # _register_v4l_binding, self-heal fuzzy scan's bound-set)
                # sees a normal /dev/video* path and its existing invariants
                # hold. The physical camera identity is preserved because
                # the by-path always points at the same USB bus:port — its
                # target is whatever kernel-node number that port has right
                # now. On the next save, the memo (populated during
                # __init__'s _remember_by_path) converts the /dev/videoN
                # back to a by-path so persistence stays stable.
                try:
                    resolved = os.path.realpath(src)
                except Exception:
                    resolved = None
                if not resolved or not resolved.startswith("/dev/video") or not os.path.exists(resolved):
                    logger.warning(f"persisted USB cam {cid}: by-path resolves to bad target ({resolved}), skip")
                    continue
                logger.info(f"persisted USB cam {cid}: by-path {src} -> {resolved}")
                result.append(resolved)
            elif src.startswith("/dev/video"):
                if not os.path.exists(src):
                    logger.warning(f"persisted USB cam {cid}: raw path missing on disk, skip: {src}")
                    continue
                result.append(src)
            # anything else (rtsp/http/basler/...) doesn't belong in the USB
            # boot list — it's added elsewhere.
        return result
    except Exception as e:
        logger.warning(f"_load_persisted_usb_sources_ordered failed, falling back to auto-detect: {e}")
        return []


def detect_video_devices() -> List[str]:
    """Auto-detect available video devices from /dev/video* (even numbers only).

    Returns a sorted list of video device paths like ['/dev/video0', '/dev/video2', '/dev/video4'].
    Only even-numbered devices are returned as odd numbers are typically metadata devices.

    NOTE: This function only sees UVC / V4L2 cameras. Basler USB3 Vision
    cameras (and other industrial "pro" cameras) do NOT register as
    /dev/video* — they need their own SDK to enumerate. See
    detect_pro_cameras() below.
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


def detect_pro_cameras() -> List[Dict[str, Any]]:
    """Auto-detect all industrial ("pro") cameras attached to the host.

    Currently only Basler USB3 Vision (via services.basler_camera). As new
    industrial vendors are added (Allied Vision, IDS, FLIR/Teledyne), each
    ships an enumeration function and this aggregator concatenates them so
    the UI sees a single unified "pro" camera list.

    Each entry is: {source, serial, model, vendor, device_class, type: 'pro'}.
    Returns [] when no drivers are installed or no devices are present —
    never raises, so it's safe to call at module load time.
    """
    results: List[Dict[str, Any]] = []
    try:
        from services.basler_camera import detect_basler_cameras
        results.extend(detect_basler_cameras())
    except Exception as e:
        logger.debug(f"Basler enumeration skipped: {e}")
    # Future: Allied Vision (VimbaX), IDS (peak), FLIR/Teledyne (Spinnaker)
    # would append here.
    return results


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
    """Get all available auto-discoverable camera sources (USB + pro).

    Returns a list of camera source strings — /dev/videoN for UVC + basler://SN
    for Basler etc. IP cameras are still stored in the "cameras" config table
    and re-added by the operator, not auto-detected on the local host.

    Priority for UVC (V4L2) sources:
      1. If environment variables (CAM_X_PATH) are set, use those
      2. Otherwise, auto-detect from /dev/video*

    Pro cameras (Basler etc.) are always auto-enumerated via their vendor
    SDKs (Pylon in Basler's case) and appended at the end so operator's
    UVC ordering is preserved.
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
        # v4.0.143 — PREFER persistence sources over raw /dev/video* scan.
        # If the operator has persisted USB cams (typically with by-path
        # sources written by the save-time normaliser), boot in the SAME
        # cam-ID → physical-device order every time, even after a USB
        # re-enumeration renumbered /dev/videoN. Falls back to raw scan
        # when persistence has no USB cams (fresh install, first boot).
        persisted = _load_persisted_usb_sources_ordered()
        if persisted:
            for cam_path in persisted:
                all_cameras.append(cam_path)
                logger.info(f"Boot: using persisted USB source: {cam_path}")
        else:
            # Auto-detect USB cameras from /dev/video*
            detected = detect_video_devices()
            for cam_path in detected:
                if os.path.exists(cam_path):
                    all_cameras.append(cam_path)
                    logger.info(f"Auto-detected USB camera: {cam_path}")

    # 4.0.50 — append auto-detected "pro" cameras (Basler etc.) after UVC.
    # `type: "pro"` is a first-class camera category alongside "usb" and "ip"
    # so future non-Basler industrial vendors slot in with no schema change.
    for pro in detect_pro_cameras():
        src = pro.get("source")
        if src:
            all_cameras.append(src)
            logger.info(
                f"Auto-detected pro camera: {pro.get('model')} "
                f"(serial={pro.get('serial')}) → {src}"
            )

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
        # Per-camera auto-exposure opt-in (USB only). Persisted so it survives
        # restart; watcher.py reads it back on next init via
        # CameraBuffer(..., auto_exposure=cc.get('auto_exposure', False))
        "auto_exposure": bool(getattr(cam, 'auto_exposure', False)),
        # 4.0.15 — Per-camera pixel-to-millimetre calibration. None until
        # the operator enters it on the Cameras tab. Consumers convert
        # px → mm only when present.
        "px_per_mm": getattr(cam, 'px_per_mm', None),
    }

    # Include name from metadata if available
    if camera_metadata and cam_id in camera_metadata:
        meta = camera_metadata[cam_id]
        config["name"] = meta.get("name", f"Camera {cam_id}")
        # Also include type from metadata if available
        if "type" in meta:
            config["type"] = meta["type"]
        # 4.0.50 — pro-camera identity (model + serial). Without this, every
        # auto-save (Save All Configuration, camera edit, AI-model change,
        # state change) would strip model/serial from the persisted config
        # and next boot would show `Basler Camera 3` instead of the friendly
        # `Basler daA1280-54um (24613703)`.
        if meta.get("model"):
            config["model"] = meta["model"]
        if meta.get("serial"):
            config["serial"] = meta["serial"]

    # Add camera source (URL/path)
    if hasattr(cam, 'source'):
        # v4.0.143 — normalise raw /dev/videoN → its /dev/v4l/by-path/…
        # symlink at save time (using the by-path memo captured on the last
        # successful open). Persistence therefore records the physical USB
        # bus:port instead of the kernel-assigned videoN index. On next boot
        # or USB re-enumeration, resolving the same by-path lands on the
        # same physical camera — no fuzzy hunt, no silent swap between cams.
        # Never fails the save: any exception falls back to the raw source.
        _src_to_save = cam.source
        try:
            if isinstance(_src_to_save, str) and _src_to_save.startswith('/dev/video'):
                with _V4L_BY_PATH_MEMO_LOCK:
                    _bp = _V4L_BY_PATH_MEMO.get(_src_to_save)
                if _bp and os.path.exists(_bp):
                    _src_to_save = _bp
        except Exception:
            pass
        config["source"] = _src_to_save
        # Determine camera type if not already set from metadata
        if "type" not in config:
            if isinstance(cam.source, str) and cam.source.startswith(("rtsp://", "http://", "https://")):
                config["type"] = "ip"
            elif isinstance(cam.source, str) and cam.source.startswith("basler://"):
                # 4.0.50 — recognise Basler / pro-camera URIs so a save without
                # metadata (rare edge case) still tags the entry correctly and
                # boot-restore picks the "pro" branch.
                config["type"] = "pro"
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

    # Prefer _saved_props (user-configured values) over live camera.get() which
    # may return auto-exposure defaults if the camera reset itself
    saved = getattr(cam, '_saved_props', {})
    prop_to_key = {
        cv2.CAP_PROP_EXPOSURE: "exposure",
        cv2.CAP_PROP_GAIN: "gain",
        cv2.CAP_PROP_BRIGHTNESS: "brightness",
        cv2.CAP_PROP_CONTRAST: "contrast",
        cv2.CAP_PROP_SATURATION: "saturation",
        cv2.CAP_PROP_FPS: "fps",
    }
    if saved:
        for prop, key in prop_to_key.items():
            if prop in saved:
                config[key] = int(saved[prop])
    elif hasattr(cam, 'camera'):
        # Fallback: read live values only if no saved props exist
        try:
            for prop, key in prop_to_key.items():
                config[key] = int(cam.camera.get(prop))
        except:
            pass

    return config

def apply_camera_config_from_saved(cam, saved_config):
    """Apply configuration to a running camera (runtime changes from UI/API).

    Also used at startup for IP cameras added after initial USB camera creation.
    """
    if cam is None or saved_config is None:
        return

    # Apply ROI settings
    cam.roi_enabled = saved_config.get('roi_enabled', False)
    cam.roi_xmin = saved_config.get('roi_xmin', 0)
    cam.roi_ymin = saved_config.get('roi_ymin', 0)
    cam.roi_xmax = saved_config.get('roi_xmax', 1280)
    cam.roi_ymax = saved_config.get('roi_ymax', 720)

    # Apply OpenCV settings and update _saved_props for reconnect persistence
    prop_map = {
        'exposure': cv2.CAP_PROP_EXPOSURE,
        'gain': cv2.CAP_PROP_GAIN,
        'brightness': cv2.CAP_PROP_BRIGHTNESS,
        'contrast': cv2.CAP_PROP_CONTRAST,
        'saturation': cv2.CAP_PROP_SATURATION,
        'fps': cv2.CAP_PROP_FPS,
    }
    # If the saved config flips the auto-exposure preference, mirror it onto the
    # CameraBuffer object so subsequent reconnects pick up the new mode.
    if 'auto_exposure' in saved_config:
        cam.auto_exposure = bool(saved_config['auto_exposure'])

    if hasattr(cam, 'camera'):
        try:
            auto_exp = not getattr(cam, 'is_ip_camera', False) and getattr(cam, 'auto_exposure', False)
            if not getattr(cam, 'is_ip_camera', False):
                if auto_exp:
                    # Camera-firmware AE. Skip toggling to manual (mode 1) and also
                    # skip the manual EXPOSURE write below.
                    cam.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                else:
                    cam.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    cam.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            for key, prop in prop_map.items():
                if key in saved_config:
                    if auto_exp and key == 'exposure':
                        continue  # don't clobber camera-side AE
                    cam.camera.set(prop, saved_config[key])
                    if hasattr(cam, '_saved_props'):
                        cam._saved_props[prop] = saved_config[key]
            logger.info(
                f"Applied camera config: { {k: saved_config[k] for k in prop_map if k in saved_config} }"
                + (" | auto_exposure=ON (manual exposure skipped)" if auto_exp else "")
            )
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


def open_camera_source(source, **kwargs):
    """Factory: return the right buffer class for this source URI.

    Preserves the existing `CameraBuffer(...)` call convention that watcher.py
    and routers/cameras.py already use. The change: if `source` starts with
    `basler://` we hand off to services.basler_camera.BaslerBuffer, which
    exposes the same public interface (frame/success/stop/_saved_props/roi_*
    /read/release). Callers cannot tell which backend they got.

    Rest of the codebase should migrate to calling this factory instead of
    instantiating CameraBuffer directly, but CameraBuffer(source=...) is kept
    fully backwards-compatible via the dispatch in __new__ below.
    """
    # 4.0.50 — "pro" camera dispatch. Anything that is not UVC (/dev/videoX)
    # or IP (rtsp://, http(s)://) is routed to its vendor-specific backend.
    from services import basler_camera as _basler
    if _basler.is_basler_source(source):
        return _basler.BaslerBuffer(source, **kwargs)
    # Otherwise fall through to the classic UVC/IP CameraBuffer.
    return CameraBuffer(source, **kwargs)


class CameraBuffer:
    def __new__(cls, source, *args, **kwargs):
        """Route to BaslerBuffer when the URI says so.

        Preserves the existing `CameraBuffer(source, exposure=...)` call
        sites transparently — no need to change every caller to use the
        factory. If the source is a basler:// URI, __new__ returns a
        BaslerBuffer instance and __init__ below is skipped by Python
        because the returned instance's type is not a CameraBuffer subclass.
        """
        # 4.0.50 — transparent Basler dispatch. See open_camera_source() docstring.
        from services import basler_camera as _basler
        if _basler.is_basler_source(source):
            return _basler.BaslerBuffer(source, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, source, exposure=100, gain=100, brightness=100,
                 contrast=0, saturation=50, fps=10, roi_config=None,
                 auto_exposure=False) -> None:
        """Open a video source (USB /dev/videoX or RTSP URL) and start the buffer thread.

        auto_exposure (USB only): when True, MVE does NOT force the camera into manual
        exposure mode — the camera firmware's own AE algorithm runs and the user-set
        `exposure` value is ignored. `gain`, `brightness`, `contrast`, `saturation`,
        `fps` still apply. Use this for venues where lighting changes a lot and a
        per-state exposure override (state_machine.State.exposure) isn't enough.
        """
        # If we somehow ended up here for a basler:// source (shouldn't
        # happen thanks to __new__), reroute proactively so we don't crash.
        try:
            from services.basler_camera import is_basler_source
            if is_basler_source(source):
                logger.warning("CameraBuffer.__init__ reached with basler:// source; expected __new__ to route. Ignoring init.")
                return
        except Exception:
            pass

        self.source = source
        self.auto_exposure = bool(auto_exposure)
        self.is_ip_camera = isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://"))
        # 4.0.54 — register this V4L2 path so the self-heal resolver never
        # picks it as a "replacement" for another cam's stuck source.
        # No-op for RTSP/HTTP sources.
        _register_v4l_binding(source)
        # 4.0.50 — expose is_pro_camera on classic UVC/IP buffers too so
        # downstream code can uniformly check `if getattr(cam, 'is_pro_camera', False)`
        # without needing to import BaslerBuffer to distinguish types.
        self.is_pro_camera = False

        # Initialize camera with appropriate backend
        if self.is_ip_camera:
            self.camera = cv2.VideoCapture(source)
            logger.info(f"Initializing IP camera: {source.split('@')[-1] if '@' in source else source}")
        else:
            self.camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
            logger.info(f"Initializing USB camera: {source}")
            # v4.0.138 — snapshot the current /dev/v4l/by-path symlink that
            # points at this /dev/videoN so a later USB renumber can be
            # recovered deterministically (see _recover_via_by_path).
            _bp = _remember_by_path(source)
            if _bp:
                logger.info(f"USB camera {source}: by-path memo → {_bp}")

        # Set resolution
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if self.is_ip_camera:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, IP_CAMERA_BRIGHTNESS)
            self.camera.set(cv2.CAP_PROP_CONTRAST, IP_CAMERA_CONTRAST)
            self.camera.set(cv2.CAP_PROP_SATURATION, IP_CAMERA_SATURATION)
        else:
            # USB camera: set format, then either force manual exposure (default) or
            # leave auto-exposure on if the user opted in via camera config.
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J', 'P', 'G'))
            if self.auto_exposure:
                # Mode 3 = aperture priority / camera-side AE. We deliberately do NOT
                # toggle back to mode 1 (manual). Skip the `set(CAP_PROP_EXPOSURE, …)`
                # call too — writing a value would re-enable manual mode on some
                # firmwares and override the AE we just asked for.
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                logger.info(f"Camera {source}: auto_exposure=True — manual exposure override skipped")
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 6000)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 0)
            self.camera.set(cv2.CAP_PROP_GAMMA, 1)
            if not self.auto_exposure:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.camera.set(cv2.CAP_PROP_GAIN, gain)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            self.camera.set(cv2.CAP_PROP_CONTRAST, contrast)
            self.camera.set(cv2.CAP_PROP_SATURATION, saturation)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

        success, frame = self.camera.retrieve(0)
        self.success = success
        self.frame = frame
        self.stop = False

        # ROI (Region of Interest) settings
        if roi_config:
            self.roi_enabled = roi_config.get('roi_enabled', False)
            self.roi_xmin = roi_config.get('roi_xmin', 0)
            self.roi_ymin = roi_config.get('roi_ymin', 0)
            self.roi_xmax = roi_config.get('roi_xmax', 1280)
            self.roi_ymax = roi_config.get('roi_ymax', 720)
        else:
            self.roi_enabled = False
            self.roi_xmin = 0
            self.roi_ymin = 0
            self.roi_xmax = 1280
            self.roi_ymax = 720

        # Store user-configured values for reconnect persistence and save-to-disk.
        # These are the SOURCE OF TRUTH — not camera.get() which may differ.
        self._saved_props = {
            cv2.CAP_PROP_EXPOSURE: exposure,
            cv2.CAP_PROP_GAIN: gain,
            cv2.CAP_PROP_BRIGHTNESS: brightness,
            cv2.CAP_PROP_CONTRAST: contrast,
            cv2.CAP_PROP_SATURATION: saturation,
            cv2.CAP_PROP_FPS: fps,
        }
        logger.info(f"Camera {source}: exposure={exposure}, gain={gain}, brightness={brightness}, contrast={contrast}, saturation={saturation}, fps={fps}")
        threading.Thread(target=self.buffer).start()

    def _apply_saved_props(self):
        """Re-apply saved camera properties after reconnect.

        Mirrors the auto_exposure logic from __init__: if the user opted into
        camera-firmware AE via the Auto-Exposure checkbox, leave AE mode on and
        skip the manual CAP_PROP_EXPOSURE write — otherwise the reconnect path
        would clobber the AE every time the camera dropped and re-attached.
        """
        # Re-apply base camera settings (resolution, format)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.is_ip_camera:
            try:
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                if getattr(self, 'auto_exposure', False):
                    # Mode 3 = aperture priority / camera-side AE. Do NOT toggle back
                    # to manual (mode 1). Below we also skip the EXPOSURE prop write.
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                else:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
                self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 6000)
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
        if not self._saved_props:
            return
        skip_props = set()
        if not self.is_ip_camera and getattr(self, 'auto_exposure', False):
            # Writing CAP_PROP_EXPOSURE re-enables manual mode on most firmwares,
            # negating the AE we just asked for. Skip just this one prop.
            skip_props.add(cv2.CAP_PROP_EXPOSURE)
        for prop, val in self._saved_props.items():
            if prop in skip_props:
                continue
            try:
                self.camera.set(prop, val)
            except Exception:
                pass
        logger.info(
            f"Re-applied saved camera config after reconnect: {self.source} | "
            f"props={ {p: v for p, v in self._saved_props.items() if p not in skip_props} }"
            + (" | auto_exposure=ON (manual exposure skipped)" if cv2.CAP_PROP_EXPOSURE in skip_props else "")
        )

    def update_prop(self, prop, value):
        """Update a camera property and save it for reconnect persistence."""
        self.camera.set(prop, value)
        self._saved_props[prop] = value

    def buffer(self):
        failure_count = 0
        max_failures = 100  # Reconnect after 100 consecutive failures
        # 4.0.54 — after this many blind reconnect cycles (each = max_failures
        # blank reads + one release/reopen on the same path) we assume the
        # source path itself is stale (e.g. UVC re-enumeration renumbered the
        # device or MVE captured a metadata sibling). At that point, try to
        # resolve a working alternate node instead of spinning forever.
        stale_reconnect_cycles = 0
        STALE_CYCLES_BEFORE_HEAL = 3

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
                        # 4.0.54 — a run of successful reads means the current
                        # source is delivering; clear the stale-cycle budget so
                        # a future drift episode gets the full 3-cycle grace
                        # before the resolver kicks in.
                        stale_reconnect_cycles = 0
                else:
                    self.success = success
                    failure_count += 1

                    # Auto-reconnect after consecutive failures
                    if failure_count >= max_failures:
                        logger.warning(f"Camera connection lost after {failure_count} failures. Attempting reconnect...")
                        try:
                            self.camera.release()
                            time.sleep(1)  # Brief pause before reconnecting

                            # 4.0.54 — self-heal stale /dev/videoN. If reopening
                            # the same USB path hasn't helped after
                            # STALE_CYCLES_BEFORE_HEAL rounds, probe unclaimed
                            # video nodes for one that actually delivers frames
                            # and switch self.source. Scoped to USB — RTSP/HTTP
                            # keeps the old blind-retry behaviour.
                            if not self.is_ip_camera:
                                stale_reconnect_cycles += 1
                                if stale_reconnect_cycles >= STALE_CYCLES_BEFORE_HEAL:
                                    replacement = _find_replacement_v4l_path(self.source)
                                    if replacement:
                                        logger.warning(
                                            f"Stale V4L2 path detected — swapping {self.source} → {replacement} "
                                            f"(after {stale_reconnect_cycles} blind reconnect cycles)"
                                        )
                                        _unregister_v4l_binding(self.source)
                                        self.source = replacement
                                        _register_v4l_binding(self.source)
                                        stale_reconnect_cycles = 0

                            # Reinitialize camera
                            if self.is_ip_camera:
                                self.camera = cv2.VideoCapture(self.source)
                                logger.info(f"Reconnecting IP camera: {self.source.split('@')[-1] if '@' in self.source else self.source}")
                            else:
                                self.camera = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                                logger.info(f"Reconnecting USB camera: {self.source}")
                                # v4.0.138 — refresh by-path memo on every reconnect
                                # so a subsequent USB renumber can be recovered even
                                # if it happens minutes/hours after the initial open.
                                _remember_by_path(self.source)

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
