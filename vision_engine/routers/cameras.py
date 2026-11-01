from fastapi import APIRouter, Request, HTTPException
from fastapi import Depends as Depends
from routers._body import json_body as _json_body_dep, body as _req_body  # 4.0.425
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from config import load_service_config, save_service_config, DATA_FILE
from routers.config_routes import build_current_service_config
from services.camera import (CameraBuffer, safe_source, mask_url_password, scan_network_for_camera_devices,
                             get_camera_config_for_save, detect_pro_cameras, open_ip_capture)
import base64
import cv2
import os
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class CameraDiscoveryRequest(BaseModel):
    subnet: str = "192.168.0"


# =============================================================================
# VIDEO FEED ENDPOINTS
# =============================================================================



# =============================================================================
# CAMERA STATUS & CONTROL ENDPOINTS
# =============================================================================

def _live_cam_path(watcher, cam_id, cam):
    """4.0.452 — the camera's CURRENT source. The boot-time camera_paths entry is never
    updated when a USB camera re-enumerates (by-path re-resolve / drift switch), so a
    streaming camera read 'disconnected' once its old /dev/videoN was gone."""
    boot = watcher.camera_paths[cam_id - 1] if cam_id <= len(watcher.camera_paths) else None
    src = getattr(cam, "source", None) if cam is not None else None
    if isinstance(src, str) and src.startswith("/dev/video"):
        return src
    return boot


@router.get("/api/cameras")
def get_cameras_status(request: Request):
    """Get status of all cameras (dynamically detected)."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    cameras = []
    # Use dynamic cameras dict
    for cam_id, cam in list(watcher.cameras.items()):   # 4.0.442 — a copy (hot-add / reconciler mutate it)
        # Get camera path from stored paths
        cam_path = _live_cam_path(watcher, cam_id, cam) or "unknown"

        # Get camera metadata if available
        metadata = {}
        if hasattr(watcher, 'camera_metadata') and cam_id in watcher.camera_metadata:
            metadata = watcher.camera_metadata[cam_id]

        # 4.0.50 — camera type dispatch. USB = /dev/video* (V4L2), IP = rtsp://
        # or http(s)://, pro = basler://<serial> (industrial). USB is the only
        # type that has a filesystem-visible device node, so device_exists
        # applies to USB alone. IP and pro use their own reconnect + health
        # signals via `cam.success`.
        cam_type = metadata.get("type", "usb")
        is_usb = cam_type == "usb"
        is_pro = cam_type == "pro" or bool(getattr(cam, 'is_pro_camera', False))
        device_exists = os.path.exists(cam_path) if is_usb and cam_path != "unknown" else True
        is_connected = cam is not None and getattr(cam, 'success', False) and device_exists

        # v4.0.139 — surface the current /dev/v4l/by-path/… symlink so
        # operators see the STABLE physical-USB-port identifier next to (or
        # instead of) the fragile /dev/videoN index. Only relevant for USB.
        by_path = None
        if is_usb and cam_path and cam_path.startswith('/dev/video'):
            try:
                from services.camera import _by_path_for_video_node
                by_path = _by_path_for_video_node(cam_path)
            except Exception:
                by_path = None

        cam_info = {
            "id": cam_id,
            # 4.0.452 — masked: for IP cameras this is the full rtsp://user:PASSWORD@… source,
            # and GET requests are never role-gated (the card shows it when camera_path is empty)
            "path": mask_url_password(cam_path),
            "by_path": by_path,   # v4.0.139 — None when not a USB or no symlink present
            "name": metadata.get("name", f"Camera {cam_id}"),
            "type": cam_type,
            "connected": is_connected,
            "running": cam is not None and not getattr(cam, 'stop', True),
            "device_exists": device_exists,
            # 4.0.415 — freeze watchdog visibility: seconds since the last good frame
            # (None = no frame yet) and how often this stream was found frozen.
            "last_frame_age_s": (round(time.time() - cam.last_frame_time, 1)
                                 if getattr(cam, 'last_frame_time', None) else None),
            "frozen_reconnects": getattr(cam, 'frozen_reconnects', 0),
        }

        # Add IP camera specific info
        if metadata.get("type") == "ip":
            cam_info["ip"] = metadata.get("ip")
            cam_info["camera_path"] = metadata.get("path")

        # 4.0.50 — pro-camera identity (model + serial) surfaces in the UI so
        # operators know which Basler / vendor unit is which without opening
        # each card.
        if is_pro:
            cam_info["model"]  = metadata.get("model")
            cam_info["serial"] = metadata.get("serial")

        if cam is not None and hasattr(cam, 'camera'):
            try:
                saved = getattr(cam, '_saved_props', {})
                if is_pro:
                    # 4.0.50 — pro-camera config comes ENTIRELY from _saved_props.
                    # cam.camera is a pylon.InstantCamera which has no cv2.get()
                    # method, so the UVC path below would crash. Read directly
                    # from _saved_props which BaslerBuffer keeps in sync with the
                    # actual NodeMap values.
                    _fps  = int(saved.get(cv2.CAP_PROP_FPS, 30))
                    _exp  = int(saved.get(cv2.CAP_PROP_EXPOSURE, 5000))
                    _gain = int(saved.get(cv2.CAP_PROP_GAIN, 0))
                    _br   = int(saved.get(cv2.CAP_PROP_BRIGHTNESS, 0))
                    _con  = int(saved.get(cv2.CAP_PROP_CONTRAST, 0))
                    _sat  = int(saved.get(cv2.CAP_PROP_SATURATION, 0))
                    _frame = getattr(cam, 'frame', None)
                    if _frame is not None and hasattr(_frame, 'shape'):
                        _h, _w = _frame.shape[:2]
                    else:
                        _h, _w = 720, 1280
                    cam_info["config"] = {
                        "fps": _fps,
                        "width": _w,
                        "height": _h,
                        "exposure": _exp,
                        "gain": _gain,
                        "brightness": _br,
                        "contrast": _con,
                        "saturation": _sat,
                        "roi_enabled": getattr(cam, 'roi_enabled', False),
                        "roi_xmin": getattr(cam, 'roi_xmin', 0),
                        "roi_ymin": getattr(cam, 'roi_ymin', 0),
                        "roi_xmax": getattr(cam, 'roi_xmax', 1280),
                        "roi_ymax": getattr(cam, 'roi_ymax', 720),
                        "pixel_format": saved.get("basler.pixel_format"),
                    }
                else:
                    # Existing UVC / IP path — untouched.
                    def _get(prop):
                        if prop in saved:
                            return int(saved[prop])
                        return int(cam.camera.get(prop))
                    cam_info["config"] = {
                        "fps": _get(cv2.CAP_PROP_FPS),
                        "width": int(cam.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cam.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "exposure": _get(cv2.CAP_PROP_EXPOSURE),
                        "gain": _get(cv2.CAP_PROP_GAIN),
                        "brightness": _get(cv2.CAP_PROP_BRIGHTNESS),
                        "contrast": _get(cv2.CAP_PROP_CONTRAST),
                        "saturation": _get(cv2.CAP_PROP_SATURATION),
                        "roi_enabled": getattr(cam, 'roi_enabled', False),
                        "roi_xmin": getattr(cam, 'roi_xmin', 0),
                        "roi_ymin": getattr(cam, 'roi_ymin', 0),
                        "roi_xmax": getattr(cam, 'roi_xmax', 1280),
                        "roi_ymax": getattr(cam, 'roi_ymax', 720),
                        # 4.0.416 — the card's checkboxes read these back; without
                        # them Auto-Exposure always rendered unchecked after a refresh.
                        "auto_exposure": bool(getattr(cam, 'auto_exposure', False)),
                        "auto": bool(getattr(cam, 'auto', False)),
                    }
            except:
                cam_info["config"] = {}
        cameras.append(cam_info)

    # 4.0.370 — also surface DETECTED-but-unconfigured USB cameras, so a newly
    # plugged-in camera is VISIBLE in the list (configured=False) instead of being
    # silently ignored — the operator can then add it. Enumerate the stable
    # /dev/v4l/by-path capture symlinks and skip any already bound to a config
    # camera (whether it's currently up OR down).
    for c in cameras:
        c["configured"] = True
    try:
        import glob as _glob
        from services.camera import _by_path_for_video_node as _bp_for
        used = set()
        for c in cameras:
            if c.get("by_path"):
                used.add(c["by_path"])
            p = c.get("path") or ""
            if isinstance(p, str) and p.startswith("/dev/video"):
                b = _bp_for(p)
                if b:
                    used.add(b)
        try:  # also cover configured-but-DOWN cameras via their persisted by-path source
            for _c in (load_service_config() or {}).get("cameras", {}).values():
                s = str((_c or {}).get("source") or "")
                if s.startswith("/dev/v4l/by-path/"):
                    used.add(s)
        except Exception:
            pass
        next_id = max([c["id"] for c in cameras], default=0) + 1
        for link in sorted(_glob.glob('/dev/v4l/by-path/*-video-index0')):
            if link in used:
                continue
            node = os.path.realpath(link)
            cameras.append({
                "id": next_id, "path": node, "by_path": link,
                "name": "Detected camera (not configured)", "type": "usb",
                "connected": False, "running": False,
                "device_exists": os.path.exists(node),
                "configured": False, "detected": True,
            })
            next_id += 1
    except Exception as e:
        logger.warning("detected-camera enumeration failed: %s", e)

    return JSONResponse(content=cameras)


@router.post("/api/cameras/add-detected")
def add_detected_camera(request: Request, _mve_body=Depends(_json_body_dep)):
    """4.0.370 — add a DETECTED-but-unconfigured USB camera (by its stable by-path)
    to the config as a new camera. Persists; the camera opens on the next restart."""
    try:
        body = _req_body(_mve_body)
        by_path = str(body.get("by_path") or "").strip()
        if not by_path.startswith("/dev/v4l/by-path/"):
            return JSONResponse({"error": "by_path (a /dev/v4l/by-path/… source) required"}, status_code=400)
        svc = load_service_config() or {}
        cams = dict(svc.get("cameras") or {})
        for _c in cams.values():
            if str((_c or {}).get("source") or "") == by_path:
                return JSONResponse({"error": "camera already configured", "source": by_path}, status_code=409)
        ids = [int(k) for k in cams.keys() if str(k).isdigit()]
        new_id = (max(ids) + 1) if ids else 1
        tmpl = cams[str(min(ids))] if ids else {}
        def _g(k, dflt): return (tmpl or {}).get(k, dflt)
        cams[str(new_id)] = {
            "name": str(body.get("name") or f"USB Camera {new_id}"), "type": "usb",
            "source": by_path,
            "fps": _g("fps", 10), "gain": _g("gain", 100), "exposure": _g("exposure", 10),
            "brightness": _g("brightness", 100), "contrast": _g("contrast", 0), "saturation": _g("saturation", 50),
            "auto_exposure": _g("auto_exposure", False), "auto": _g("auto", False), "px_per_mm": None,
            "roi_enabled": False, "roi_xmin": 0, "roi_ymin": 0, "roi_xmax": 1280, "roi_ymax": 720,
        }
        svc["cameras"] = cams
        save_service_config(svc)
        return {"ok": True, "id": new_id, "source": by_path,
                "note": "Added. Restart the vision engine to open it."}
    except Exception as e:
        logger.warning("add-detected camera failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/cameras/rescan")
def rescan_cameras(request: Request):
    """Hot-plug check for the Cameras tab's Refresh — REPORT ONLY since 4.0.428.

    Lists configured cameras whose device is gone (`missing`) and unbound by-path
    devices (`detected_new`); never removes, restarts or adds a camera at runtime.
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
def restart_camera(camera_id: int, request: Request):
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
        # 4.0.431 — the camera's own capture thread re-opens it (within about one read)
        return JSONResponse(content={"success": True, "message": f"Camera {camera_id} restart requested — it re-opens within a second"})
    except Exception as e:
        logger.error(f"Error restarting camera {camera_id}: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/camera/{camera_id}/config")
def update_camera_config(camera_id: int, request: Request, _mve_body=Depends(_json_body_dep)):
    """Update camera configuration."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    # Use dynamic cameras dict
    cam = watcher.cameras.get(camera_id)
    if cam is None or not hasattr(cam, 'camera'):
        return JSONResponse(content={"error": f"Camera {camera_id} not available"}, status_code=404)

    try:
        body = _req_body(_mve_body)

        # Apply configuration via update_prop (persists across reconnects)
        # 4.0.452 — width / height are not accepted: the resolution is fixed (1280×720) and
        # setting it on a streaming V4L2 camera from this thread could crash MVE (exit 139)
        prop_map = {
            'fps': cv2.CAP_PROP_FPS,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
        }

        updated = []
        ignored = [k for k in ('width', 'height') if k in body]
        # 4.0.416 — "Auto" (USB) first, so the property writes below already respect
        # it (in Auto they are remembered, not written). Applies live, no restart.
        if 'auto' in body and hasattr(cam, 'set_auto'):
            cam.set_auto(bool(body['auto']))
            updated.append('auto')
        for key, value in body.items():
            if key in prop_map:
                if hasattr(cam, 'update_prop'):
                    cam.update_prop(prop_map[key], value)
                    updated.append(key)
                else:
                    ignored.append(key)    # no settable capture (never cam.camera.set from here)

        # Handle ROI settings (stored on CameraBuffer object, not OpenCV)
        roi_props = ['roi_enabled', 'roi_xmin', 'roi_ymin', 'roi_xmax', 'roi_ymax']
        for key in roi_props:
            if key in body:
                setattr(cam, key, body[key])
                updated.append(key)

        # Auto-exposure flag: stored on the CameraBuffer object so it persists into
        # `get_camera_config_for_save()`. NOTE: flipping this at runtime does NOT
        # currently re-initialise the camera — the V4L2 mode change (manual ↔ auto)
        # only takes effect on next camera open (restart or reconnect). To make the
        # change visible immediately, click Restart on this camera after Save.
        if 'auto_exposure' in body:
            cam.auto_exposure = bool(body['auto_exposure'])
            updated.append('auto_exposure')

        # 4.0.15 — Per-camera pixel-to-millimetre calibration. Accepts a
        # positive float (e.g. 6.4 means 6.4 image pixels = 1 mm IRL) or
        # null/empty to clear. Used by the LSF modal + defect drawer + charts
        # to display bbox dimensions and areas in mm/mm² instead of px/px².
        if 'px_per_mm' in body:
            try:
                v = body['px_per_mm']
                if v in (None, "", "null"):
                    cam.px_per_mm = None
                else:
                    f = float(v)
                    cam.px_per_mm = f if f > 0 else None
                updated.append('px_per_mm')
            except (TypeError, ValueError):
                pass

        # Auto-save to disk so settings persist across restarts
        if updated:
            try:
                svc_config = build_current_service_config(request.app.state)
                if svc_config:
                    save_service_config(svc_config)
                    logger.info(f"Camera {camera_id} config auto-saved: {updated}")
            except Exception as save_err:
                logger.warning(f"Camera config auto-save failed: {save_err}")

        out = {"success": True, "updated": updated}
        if ignored:
            out["ignored"] = ignored
        return JSONResponse(content=out)
    except Exception as e:
        logger.error(f"Error updating camera {camera_id} config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/camera/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: int, request: Request):
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



# =============================================================================
# CAMERA CONFIGURATION SAVE/LOAD ENDPOINTS
# =============================================================================

@router.post("/api/cameras/config/save")
def save_all_config(request: Request):
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
                # 4.0.430 — the UI shows this as "Last saved" (it fell back to the browser's
                # UTC clock — 3.5 h off in Iran — because nothing was returned)
                "saved_at": config.get("saved_at"),
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



import re as _re
# 4.0.424 — secret-named fields + the password inside stream URLs (rtsp://user:PASS@…)
_SECRET_KEY = _re.compile(r"(api_?key|password|passwd|secret|access_token|refresh_token|bot_token|^token)$", _re.I)
# 4.0.452 — greedy to the LAST '@' of the address: a password containing '@' (e.g. the
# standard 'abc!@#xyz') was masked only up to its own '@' and the rest leaked
_URL_PASSWORD = _re.compile(r"(://[^:/@\s]+:)\S*@")


def _mask_secrets(obj):
    """Copy of `obj` with every secret value replaced by '***'."""
    if isinstance(obj, dict):
        return {k: ("***" if isinstance(v, str) and v and _SECRET_KEY.search(str(k))
                    else _mask_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(v) for v in obj]
    if isinstance(obj, str) and "://" in obj and "@" in obj:
        return _URL_PASSWORD.sub(r"\1***@", obj)
    return obj


@router.get("/api/cameras/config")
def get_saved_config(request: Request):
    """Get the saved service configuration from data file.

    4.0.424 — secrets are masked. This answered every browser (and the AI assistant's
    tools, which list this path) with the raw config: AI-provider API keys, the AI
    Trainer password + tokens, the Telegram bot token and camera passwords — while the
    dedicated AI / trainer / notification config endpoints carefully masked the same
    values. The UI only reads `saved_at` from here; the full backup is
    /api/export_service_config (an explicit operator download)."""
    try:
        config = load_service_config()
        if not config:
            return JSONResponse(content={"exists": False, "config": None})
        return JSONResponse(content={"exists": True, "config": _mask_secrets(config)})
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# CAMERA CONFIGURATION UPLOAD ENDPOINT
# =============================================================================

@router.post("/api/cameras/config/upload")
def upload_config(request: Request, _mve_body=Depends(_json_body_dep)):
    """Upload and apply a service configuration (for restoring from backup)."""
    watcher = request.app.state.watcher_instance
    if watcher is None:
        return JSONResponse(content={"error": "Watcher not initialized"}, status_code=503)

    try:
        from config import save_data_file, load_data_file
        body = _req_body(_mve_body)
        config = body.get("config")

        if not config:
            return JSONResponse(content={"error": "Invalid configuration format"}, status_code=400)

        # Two accepted shapes:
        #  (a) FULL bundle (3.15.7+ export): top-level has `service_config` plus
        #      sibling keys like `timeline_config`. Persist the whole file and
        #      apply service_config + timeline_config so procedures/store/audio
        #      all restore.
        #  (b) LEGACY (pre-3.15.7 export): the flat service_config contents
        #      (cameras/states/… at top level, no `service_config` wrapper).
        #      Persist under service_config only.
        # 4.0.439 — a backup carries the shipment that was running when it was taken. Keep
        # the LIVE shipment in what gets saved: the backup's current_shipment was persisted,
        # so the next restart switched the line to a days-old shipment.
        def _keep_live_shipment(_svc):
            _live = str(getattr(watcher, "shipment", "") or "").strip()
            if isinstance(_svc, dict) and _live:
                _svc["current_shipment"] = _live
                _svc["shipment_start_encoder"] = int(getattr(watcher, "shipment_start_encoder", 0) or 0)

        if isinstance(config, dict) and "service_config" in config:
            full_data = config
            svc = config.get("service_config", {}) or {}
            _keep_live_shipment(svc)
            full_data["service_config"] = svc
            if not save_data_file(full_data):
                return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)
        else:
            svc = config
            _keep_live_shipment(svc)
            if not save_service_config(svc):
                return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)
            full_data = load_data_file()  # reload merged file so timeline_config (if any) is present

        # Apply service_config AND root-level configs (timeline_config/procedures).
        settings_applied, cameras_loaded = request.app.state.apply_config_settings(
            svc, watcher, full_data=full_data
        )

        # Invalidate the per-class caches in services.detection so the freshly
        # imported store_objects / audio_settings take effect immediately instead
        # of serving stale values for up to the 5s TTL.
        try:
            import services.detection as _det
            _det._store_objects_loaded_at = 0.0
            _det._audio_settings_loaded_at = 0.0
        except Exception:
            pass

        return JSONResponse(content={
            "success": True,
            "message": "Configuration uploaded and applied",
            "cameras_loaded": cameras_loaded,
            "applied": list(settings_applied.keys()) if isinstance(settings_applied, dict) else None
        })
    except Exception as e:
        logger.error(f"Error uploading config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# CAMERA DISCOVERY API ENDPOINTS
# =============================================================================

@router.post("/api/cameras/discover")
def discover_cameras(request: Request, body: CameraDiscoveryRequest):
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


# =============================================================================
# 4.0.50 — Pro-camera (Basler USB3 Vision, etc.) discovery + save
#
# Mirrors the IP-camera discovery / save pattern. `type: "pro"` is a first-
# class camera category alongside "usb" and "ip" so future non-Basler
# industrial vendors slot in without any schema change.
# =============================================================================

@router.get("/api/cameras/discover-pro")
def discover_pro_cameras(request: Request):
    """Enumerate industrial ("pro") cameras attached to the host.

    Currently returns Basler USB3 Vision cameras (via pypylon). The endpoint
    is safe to call even when Pylon isn't installed or no Basler is
    attached — you get an empty list, not an error, so the UI can just say
    "no pro cameras detected".

    Response:
      {
        "success": true,
        "cameras": [
          {
            "source": "basler://12345678",
            "serial": "12345678",
            "model":  "acA640-90uc",
            "vendor": "Basler AG",
            "device_class": "BaslerUsb",
            "type":   "pro"
          },
          ...
        ],
        "count": N
      }
    """
    try:
        found = detect_pro_cameras()
        # Annotate whether each device is already saved so the UI can grey
        # out "add" for cameras that are already configured.
        try:
            svc_cfg = load_service_config() or {}
            saved_sources = {
                (cam or {}).get("source")
                for cam in (svc_cfg.get("cameras") or {}).values()
            }
        except Exception:
            saved_sources = set()
        for f in found:
            f["already_saved"] = f.get("source") in saved_sources
        return JSONResponse(content={
            "success": True,
            "cameras": found,
            "count":   len(found),
        })
    except Exception as e:
        logger.error(f"Pro camera discovery failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error":   str(e),
            "cameras": [],
        }, status_code=500)


@router.post("/api/cameras/save-pro")
def save_pro_camera(request: Request, _mve_body=Depends(_json_body_dep)):
    """Persist a discovered pro camera into service_config.cameras.

    Body:
      {
        "source": "basler://12345678",   # required
        "name":   "Basler acA640-90uc",  # optional; defaults to model+serial
        "model":  "acA640-90uc",         # optional; passed through for display
        "serial": "12345678",            # optional; parsed from source if missing
        "exposure": 5000,                # µs, Basler-native units (V1 default)
        "gain":     0,                   # dB
        "fps":      30
      }

    Returns the same shape as /api/cameras/save so the existing UI code can
    reuse its success handling.
    """
    from services.basler_camera import BASLER_SOURCE_PREFIX, is_basler_source
    try:
        body = _req_body(_mve_body)
        source = str(body.get("source") or "").strip()
        if not source:
            return JSONResponse(content={"success": False, "error": "source is required"}, status_code=400)
        # For now the only pro backend is Basler. Accept any basler:// URI;
        # future backends (allied://, ids://, flir://) route to their own
        # save-* endpoints so payload shapes stay explicit.
        if not is_basler_source(source):
            return JSONResponse(content={
                "success": False,
                "error":   f"Unsupported pro-camera source: {source}",
            }, status_code=400)

        serial = body.get("serial") or source.replace(BASLER_SOURCE_PREFIX, "")
        model  = body.get("model") or "Basler USB3"
        camera_name = body.get("name") or f"{model} ({serial})"

        # Same runtime-preserving config load pattern the IP save uses.
        config = build_current_service_config(request.app.state) or {}
        if "cameras" not in config:
            config["cameras"] = {}

        # Match on source so a re-add updates in-place rather than making
        # duplicate entries.
        existing_id = None
        for cam_id, cam_config in config["cameras"].items():
            if cam_config.get("source") == source:
                existing_id = cam_id
                break
        if existing_id:
            camera_id = existing_id
        else:
            existing_ids = [int(k) for k in config["cameras"].keys() if k.isdigit()]
            camera_id = str(max(existing_ids) + 1) if existing_ids else "1"

        camera_entry = {
            "name":       camera_name,
            "source":     source,
            "type":       "pro",
            "model":      model,
            "serial":     serial,
            "roi_enabled": False,
            "roi_xmin":    0,
            "roi_ymin":    0,
            "roi_xmax":    1280,
            "roi_ymax":    720,
            "exposure":    int(body.get("exposure", 5000)),   # µs (Basler unit)
            "gain":        int(body.get("gain", 0)),          # dB
            "brightness":  0,   # UVC-only knob; kept for schema parity
            "contrast":    0,
            "saturation":  0,
            "fps":         int(body.get("fps", 30)),
            "enabled":     True,
        }
        if existing_id:
            # 4.0.452 — keep the existing entry's settings (ROI, px/mm, auto …); only the
            # identity + the values this request actually sent change
            _prev = dict(config["cameras"].get(existing_id) or {})
            _upd = {k: camera_entry[k] for k in ("name", "source", "type", "model", "serial")}
            for k in ("exposure", "gain", "fps"):
                if k in body:
                    _upd[k] = camera_entry[k]
            camera_entry = {**_prev, **_upd}

        config["cameras"][camera_id] = camera_entry
        logger.info(
            f"{'Updated' if existing_id else 'Added'} pro camera "
            f"{camera_id}: {camera_name} ({source})"
        )

        if not save_service_config(config):
            return JSONResponse(content={
                "success": False,
                "error":   "save_service_config failed",
            }, status_code=500)

        # Hot-add to the running watcher so the operator sees the feed
        # immediately, no restart needed. Same pattern the IP-save uses.
        watcher = request.app.state.watcher_instance
        cam_id_int = int(camera_id)
        if watcher is not None and cam_id_int not in watcher.cameras:
            try:
                # CameraBuffer.__new__ dispatches to BaslerBuffer for
                # basler:// sources — see services/camera.py.
                cam = CameraBuffer(
                    source,
                    exposure=camera_entry["exposure"],
                    gain=camera_entry["gain"],
                    fps=camera_entry["fps"],
                )
                watcher.cameras[cam_id_int] = cam
                watcher.camera_metadata[cam_id_int] = {
                    "name":   camera_name,
                    "type":   "pro",
                    "model":  model,
                    "serial": serial,
                    "source": source,
                }
                while len(watcher.camera_paths) < cam_id_int:
                    watcher.camera_paths.append("")
                if cam_id_int <= len(watcher.camera_paths):
                    watcher.camera_paths[cam_id_int - 1] = source
                else:
                    watcher.camera_paths.append(source)
                logger.info(f"Hot-added pro camera {cam_id_int} to watcher: {source}")
            except Exception as e:
                logger.warning(
                    f"Pro camera saved but hot-add failed: {e}. Restart to apply."
                )

        return JSONResponse(content={
            "success":        True,
            "message":        f"Camera '{camera_name}' saved as Camera {camera_id}",
            "camera":         camera_entry,
            "camera_id":      camera_id,
            "total_cameras":  len(config["cameras"]),
        })
    except Exception as e:
        logger.error(f"save_pro_camera failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error":   str(e),
        }, status_code=500)


@router.post("/api/cameras/test")
def test_camera(request: Request, _mve_body=Depends(_json_body_dep)):
    """Test a camera URL with credentials and return a snapshot."""
    try:
        body = _req_body(_mve_body)
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

        # Test the camera (timeouts only take effect when passed at open)
        cap = open_ip_capture(test_url, 3000, 3000)

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
def save_camera(request: Request, _mve_body=Depends(_json_body_dep)):
    """Save a discovered camera to the configuration."""
    try:
        body = _req_body(_mve_body)
        camera_name = body.get("name", "Unnamed Camera")
        camera_url = body.get("url")
        camera_ip = body.get("ip")
        camera_path = body.get("path")
        resolution = body.get("resolution", {})

        if not camera_url:
            return JSONResponse(content={"success": False, "error": "Camera URL is required"}, status_code=400)

        logger.info(f"Saving camera: {camera_name} ({camera_ip}) - {safe_source(camera_url)}")

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
        _prev = dict(config["cameras"].get(existing_id) or {}) if existing_id else {}
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
        if _prev:
            # 4.0.452 — re-saving an existing camera (rename / re-test) keeps its ROI,
            # exposure, px/mm, … — only the identity fields change (it reset them to defaults)
            camera_entry = {**_prev, **{k: camera_entry[k] for k in ("name", "source", "type", "ip", "path")}}

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
                    cam = CameraBuffer(camera_url, exposure=0, gain=0, fps=camera_entry.get("fps", 25))
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
                    logger.info(f"Hot-added camera {cam_id_int} to running watcher: {safe_source(camera_url)}")
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
