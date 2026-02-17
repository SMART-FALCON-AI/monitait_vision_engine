from fastapi import APIRouter, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from config import (load_data_file, save_data_file, load_service_config, save_service_config,
                    REDIS_HOST, REDIS_PORT, DATA_FILE)
from routers.config_routes import build_current_service_config
from services.camera import (CameraBuffer, scan_network_for_cameras, scan_network_for_camera_devices,
                             test_camera_stream, get_camera_config_for_save, apply_camera_config_from_saved,
                             format_relative_time, IP_CAMERA_USER, IP_CAMERA_PASS)
from services.state_machine import State
import asyncio
import base64
import cv2
import numpy as np
import os
import time
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class CameraDiscoveryRequest(BaseModel):
    subnet: str = "192.168.0"


# =============================================================================
# VIDEO FEED ENDPOINTS
# =============================================================================

@router.get("/video_feed")
async def video_feed(request: Request):
    """Lightweight MJPEG stream from all cameras stacked vertically with detection overlays."""
    app_state = request.app.state

    def generate():
        frame_count = 0
        last_log_time = time.time()
        while True:
            try:
                watcher = app_state.watcher_instance
                if watcher and watcher.cameras:
                    # Get rotation from config
                    try:
                        _rot = app_state.timeline_config.get('image_rotation', 0)
                    except:
                        _rot = 0
                    # Collect frames from all cameras (sorted by camera ID)
                    camera_frames = []
                    for cam_id in sorted(watcher.cameras.keys()):
                        cam = watcher.cameras[cam_id]
                        if hasattr(cam, 'frame') and cam.frame is not None:
                            f = cam.frame.copy()
                            if _rot == 90:
                                f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                            elif _rot == 180:
                                f = cv2.rotate(f, cv2.ROTATE_180)
                            elif _rot == 270:
                                f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            camera_frames.append(f)

                    if camera_frames:
                        frame_count += 1

                        # Draw bounding boxes on the first camera frame (detections are from the capture pipeline)
                        detections = getattr(app_state, 'latest_detections', [])
                        timestamp = getattr(app_state, 'latest_detections_timestamp', None)

                        # Log periodically
                        if time.time() - last_log_time > 5.0:
                            logger.info(f"[VIDEO_FEED] Frame #{frame_count}, {len(camera_frames)} cam(s), latest_detections={len(detections) if detections else 0}, age={time.time() - timestamp if timestamp else 'N/A'}s")
                            last_log_time = time.time()
                        if detections and timestamp:
                            detection_age = time.time() - timestamp
                            if detection_age < 10.0 and getattr(app_state, 'timeline_config', {}).get('show_bounding_boxes', True):
                                _obj_filters = getattr(app_state, 'timeline_config', {}).get('object_filters', {})
                                for det in detections:
                                    try:
                                        confidence = det.get('confidence', 0)
                                        name = det.get('name', f"Class {det.get('class', 0)}")
                                        # Apply per-object filter: check show flag and min confidence
                                        obj_f = _obj_filters.get(name, {})
                                        if obj_f.get('show') is False:
                                            continue
                                        min_conf = obj_f.get('min_confidence', 0.01)
                                        if confidence < min_conf:
                                            continue
                                        x1 = int(det.get('xmin', 0))
                                        y1 = int(det.get('ymin', 0))
                                        x2 = int(det.get('xmax', 0))
                                        y2 = int(det.get('ymax', 0))
                                        cv2.rectangle(camera_frames[0], (x1, y1), (x2, y2), (0, 255, 0), 3)
                                        label = f"{name} {confidence:.2f}"
                                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        cv2.rectangle(camera_frames[0], (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                                        cv2.putText(camera_frames[0], label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    except Exception as e:
                                        logger.error(f"Error drawing detection box: {e}")

                        # Stack all camera frames vertically
                        if len(camera_frames) > 1:
                            max_width = max(f.shape[1] for f in camera_frames)
                            resized = []
                            for f in camera_frames:
                                if f.shape[1] != max_width:
                                    scale = max_width / f.shape[1]
                                    f = cv2.resize(f, (max_width, int(f.shape[0] * scale)))
                                resized.append(f)
                            frame = np.vstack(resized)
                        else:
                            frame = camera_frames[0]

                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)  # 10 FPS max
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/video_feed_detections")
async def video_feed_detections(request: Request):
    """Video feed showing the LAST PROCESSED IMAGE with bounding boxes (Gradio inference results).

    This shows the actual processed images from raw_images/ directory with detections drawn on them,
    NOT the live camera stream. Updates when new images are processed by the state machine.
    """
    app_state = request.app.state

    def generate():
        logger.info("[DETECTION_FEED] Stream started - showing processed images with detections")
        last_frame_id = None
        while True:
            try:
                watcher = app_state.watcher_instance
                detections = getattr(app_state, 'latest_detections', [])
                timestamp = getattr(app_state, 'latest_detections_timestamp', None)
                latest_frame_id = getattr(watcher, 'latest_frame_id', None) if watcher else None

                # Only update when a new frame has been processed
                if latest_frame_id and latest_frame_id != last_frame_id and detections:
                    last_frame_id = latest_frame_id

                    # Try to load the processed image from raw_images directory
                    frame_path = os.path.join("raw_images", f"{latest_frame_id}.jpg")
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None and getattr(app_state, 'timeline_config', {}).get('show_bounding_boxes', True):
                            # Draw detections with bounding boxes (filtered by object_filters)
                            _obj_filters = getattr(app_state, 'timeline_config', {}).get('object_filters', {})
                            for det in detections:
                                try:
                                    confidence = det.get('confidence', 0)
                                    name = det.get('name', f"Class {det.get('class', 0)}")
                                    obj_f = _obj_filters.get(name, {})
                                    if obj_f.get('show') is False:
                                        continue
                                    min_conf = obj_f.get('min_confidence', 0.01)
                                    if confidence < min_conf:
                                        continue
                                    x1 = int(det.get('xmin', 0))
                                    y1 = int(det.get('ymin', 0))
                                    x2 = int(det.get('xmax', 0))
                                    y2 = int(det.get('ymax', 0))
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
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
                                logger.info(f"[DETECTION_FEED] Displayed frame {latest_frame_id} with {len(detections)} detections")

                time.sleep(0.5)  # 2 FPS - only updates when new images are processed
            except Exception as e:
                logger.error(f"Detection stream error: {e}")
                time.sleep(1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# =============================================================================
# CAMERA STATUS & CONTROL ENDPOINTS
# =============================================================================

@router.get("/api/cameras")
async def get_cameras_status(request: Request):
    """Get status of all cameras (dynamically detected)."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    cameras = []
    # Use dynamic cameras dict
    for cam_id, cam in watcher.cameras.items():
        # Get camera path from stored paths
        cam_path = watcher.camera_paths[cam_id - 1] if cam_id <= len(watcher.camera_paths) else "unknown"

        # Get camera metadata if available
        metadata = {}
        if hasattr(watcher, 'camera_metadata') and cam_id in watcher.camera_metadata:
            metadata = watcher.camera_metadata[cam_id]

        # For USB cameras, verify device node still exists
        is_usb = metadata.get("type", "usb") == "usb"
        device_exists = os.path.exists(cam_path) if is_usb and cam_path != "unknown" else True
        is_connected = cam is not None and getattr(cam, 'success', False) and device_exists

        cam_info = {
            "id": cam_id,
            "path": cam_path,
            "name": metadata.get("name", f"Camera {cam_id}"),
            "type": metadata.get("type", "usb"),
            "connected": is_connected,
            "running": cam is not None and not getattr(cam, 'stop', True),
            "device_exists": device_exists,
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


@router.post("/api/cameras/rescan")
async def rescan_cameras(request: Request):
    """Re-scan /dev/video* for USB cameras (hot-plug support).

    Detects newly plugged cameras and removes disconnected ones.
    """
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        result = watcher.rescan_cameras()
        return JSONResponse(content={"success": True, **result})
    except Exception as e:
        logger.error(f"Camera rescan failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/camera/{camera_id}/restart")
async def restart_camera(camera_id: int, request: Request):
    """Restart a specific camera."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher.cameras.get(camera_id)
    if cam is None:
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    try:
        cam.restart_camera()
        return JSONResponse(content={"success": True, "message": f"Camera {camera_id} restarted"})
    except Exception as e:
        logger.error(f"Error restarting camera {camera_id}: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/camera/{camera_id}/config")
async def update_camera_config(camera_id: int, request: Request):
    """Update camera configuration."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher.cameras.get(camera_id)
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


@router.get("/api/camera/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: int, request: Request):
    """Get current frame from a camera as JPEG."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher.cameras.get(camera_id)
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


@router.get("/api/camera/{camera_id}/stream")
async def get_camera_stream(camera_id: int, request: Request):
    """Get live MJPEG stream from a camera."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        logger.error("Stream request but watcher not initialized")
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher.cameras.get(camera_id)
    if cam is None:
        logger.error(f"Stream request for camera {camera_id} but camera not found")
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    # Check if camera has success attribute and is working
    if hasattr(cam, 'success') and not cam.success:
        logger.error(f"Camera {camera_id} is not successfully initialized")
        return JSONResponse(content={"error": f"Camera {camera_id} not initialized"}, status_code=503)

    async def generate():
        frame_count = 0
        try:
            while True:
                frame = cam.read()
                if frame is not None:
                    # Encode frame as JPEG
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        frame_count += 1
                        if frame_count % 100 == 0:
                            logger.debug(f"Camera {camera_id} stream: {frame_count} frames sent")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        logger.warning(f"Failed to encode frame for camera {camera_id}")
                else:
                    if frame_count == 0:
                        logger.warning(f"Camera {camera_id} returning None frames")
                await asyncio.sleep(0.033)  # ~30 FPS
        except GeneratorExit:
            logger.info(f"Camera {camera_id} stream closed by client (sent {frame_count} frames)")
        except Exception as e:
            logger.error(f"Error streaming from camera {camera_id}: {e}", exc_info=True)

    logger.info(f"Starting MJPEG stream for camera {camera_id}")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# =============================================================================
# CAMERA CONFIGURATION SAVE/LOAD ENDPOINTS
# =============================================================================

@router.post("/api/cameras/config/save")
async def save_all_config(request: Request):
    """Save all service configurations (cameras + settings) to data file."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        config = build_current_service_config(request.app.state)
        if config is None:
            return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

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


@router.post("/api/cameras/config/load")
async def load_all_config(request: Request):
    """Load all service configurations from data file and apply them."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        config = load_service_config()
        if not config:
            return JSONResponse(content={"error": "No saved configuration found"}, status_code=404)

        # Use the helper function to apply all settings
        settings_applied, cameras_loaded = request.app.state.apply_config_settings(config, watcher)

        # Apply states configuration
        states_loaded = 0
        state_manager = request.app.state.state_manager
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


@router.get("/api/cameras/config")
async def get_saved_config(request: Request):
    """Get the saved service configuration from data file."""
    try:
        config = load_service_config()
        if not config:
            return JSONResponse(content={"exists": False, "config": None})
        return JSONResponse(content={"exists": True, "config": config})
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# CAMERA CONFIGURATION UPLOAD ENDPOINT
# =============================================================================

@router.post("/api/cameras/config/upload")
async def upload_config(request: Request):
    """Upload and apply a service configuration (for restoring from backup)."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
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
        settings_applied, cameras_loaded = request.app.state.apply_config_settings(config, watcher)

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

@router.post("/api/cameras/discover")
async def discover_cameras(request: Request, body: CameraDiscoveryRequest):
    """Quick scan to discover IP cameras on the network (no credentials needed).

    This scans for devices with camera ports open and returns all possible camera paths.
    Users can then test each path with their own credentials.
    """
    try:
        logger.info(f"Starting quick camera discovery on subnet {body.subnet}.0/24")

        # Perform port scan (no authentication needed)
        discovered_devices = scan_network_for_camera_devices(subnet=body.subnet)

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
            "subnet": body.subnet
        })

    except Exception as e:
        logger.error(f"Error during camera discovery: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "cameras": []
        }, status_code=500)


@router.post("/api/cameras/test")
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


@router.post("/api/cameras/save")
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

        # Build current service config from runtime state (preserves unsaved changes)
        config = build_current_service_config(request.app.state)
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
            # Hot-add camera to running watcher instance (no restart needed)
            watcher = request.app.state.watcher_instance
            cam_id_int = int(camera_id)
            if watcher is not None and cam_id_int not in watcher.cameras:
                try:
                    cam = CameraBuffer(camera_url, exposure=0, gain=0)
                    watcher.cameras[cam_id_int] = cam
                    watcher.camera_metadata[cam_id_int] = {
                        "name": camera_name,
                        "type": "ip",
                        "ip": camera_ip,
                        "path": camera_path,
                        "source": camera_url
                    }
                    # Update camera_paths list
                    while len(watcher.camera_paths) < cam_id_int:
                        watcher.camera_paths.append("")
                    if cam_id_int <= len(watcher.camera_paths):
                        watcher.camera_paths[cam_id_int - 1] = camera_url
                    else:
                        watcher.camera_paths.append(camera_url)
                    logger.info(f"Hot-added camera {cam_id_int} to running watcher: {camera_url}")
                except Exception as e:
                    logger.warning(f"Camera saved but failed to hot-add to watcher: {e}. Restart to apply.")

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
