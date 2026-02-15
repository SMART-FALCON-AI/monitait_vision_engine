from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from config import (
    load_data_file, save_data_file, load_service_config, save_service_config,
    # Read-only config globals (never modified via global keyword in these routes)
    DATA_FILE, WATCHER_USB, SERIAL_BAUDRATE, USER_COMMANDS,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER,
)
from services.camera import (
    get_camera_config_for_save,
    CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH,
)
from redis import Redis
from typing import Dict, Any
import config
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Helper functions
# =============================================================================

def _get_ai_models_from_redis():
    """Load all AI models from Redis. Returns dict: {"models": {...}, "active": "name"}."""
    try:
        r = Redis("redis", 6379, db=0)
        raw = r.get("ai_models")
        if raw:
            data = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
            if "models" in data:
                return data
        # Migration: check legacy single-model keys
        legacy_model = r.get("ai_model")
        legacy_key = r.get("ai_api_key")
        if legacy_model and legacy_key:
            provider = legacy_model.decode('utf-8') if isinstance(legacy_model, bytes) else legacy_model
            api_key = legacy_key.decode('utf-8') if isinstance(legacy_key, bytes) else legacy_key
            name = provider.capitalize()
            data = {"models": {name: {"provider": provider, "api_key": api_key}}, "active": name}
            r.set("ai_models", json.dumps(data))
            return data
    except Exception as e:
        logger.warning(f"Failed to load AI models from Redis: {e}")
    return {"models": {}, "active": None}


def build_current_service_config(app_state):
    """Build current service configuration from runtime state.

    Args:
        app_state: The FastAPI app.state object containing watcher_instance,
                   state_manager, and pipeline_manager.
    """
    watcher = app_state.watcher_instance
    sm = getattr(app_state, 'state_manager', None)
    pm = getattr(app_state, 'pipeline_manager', None)

    if watcher is None:
        return None

    # Get AI configuration from Redis (if available)
    ai_data = _get_ai_models_from_redis()
    ai_config = {}
    if ai_data.get("active") and ai_data["active"] in ai_data.get("models", {}):
        active = ai_data["models"][ai_data["active"]]
        ai_config = {
            "model": active["provider"],
            "api_key": active["api_key"],
            "active_name": ai_data["active"],
            "all_models": {n: {"provider": m["provider"]} for n, m in ai_data["models"].items()}
        }

    svc_config = {
        "cameras": {},
        "infrastructure": {
            "redis_host": config.REDIS_HOST,
            "redis_port": config.REDIS_PORT,
            "serial_port": WATCHER_USB,
            "serial_baudrate": SERIAL_BAUDRATE,
            "serial_mode": config.SERIAL_MODE,
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
            "enabled": config.EJECTOR_ENABLED,
            "offset": config.EJECTOR_OFFSET,
            "duration": config.EJECTOR_DURATION,
            "poll_interval": config.EJECTOR_POLL_INTERVAL
        },
        "capture": {
            "mode": config.CAPTURE_MODE,
            "time_between_packages": config.TIME_BETWEEN_TWO_PACKAGE
        },
        "image_processing": {
            "remove_raw_image_when_dm_decoded": config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED,
            "parent_object_list": config.PARENT_OBJECT_LIST
        },
        "histogram": {
            "enabled": config.HISTOGRAM_ENABLED,
            "save_image": config.HISTOGRAM_SAVE_IMAGE
        },
        "class_count_check": {
            "enabled": config.CHECK_CLASS_COUNTS_ENABLED,
            "classes": config.CHECK_CLASS_COUNTS_CLASSES,
            "confidence": config.CHECK_CLASS_COUNTS_CONFIDENCE
        },
        "datamatrix": {
            "chars_sizes": config.DM_CHARS_SIZES,
            "confidence_threshold": config.DM_CONFIDENCE_THRESHOLD,
            "overlap_threshold": config.DM_OVERLAP_THRESHOLD
        },
        "store_annotation": {
            "enabled": config.STORE_ANNOTATION_ENABLED
        }
    }

    # Add camera configs dynamically
    camera_metadata = getattr(watcher, 'camera_metadata', {})
    for cam_id, cam in watcher.cameras.items():
        cam_config = get_camera_config_for_save(cam, cam_id, camera_metadata)
        if cam_config:
            svc_config["cameras"][str(cam_id)] = cam_config

    # Add states configuration
    if sm:
        svc_config["states"] = {name: s.to_dict() for name, s in sm.states.items()}
        svc_config["current_state_name"] = sm.current_state.name if sm.current_state else "default"

    # Add pipeline configuration
    if pm:
        svc_config["pipeline_config"] = pm.to_config()

    return svc_config


# =============================================================================
# Configuration endpoints
# =============================================================================

@router.get("/config")
async def get_config(request: Request):
    """Get current configuration."""
    return JSONResponse(content={
        "ejector": {
            "enabled": config.EJECTOR_ENABLED,
            "offset": config.EJECTOR_OFFSET,
            "duration": config.EJECTOR_DURATION,
            "poll_interval": config.EJECTOR_POLL_INTERVAL
        },
        "capture": {
            "mode": config.CAPTURE_MODE,
            "time_between_packages": config.TIME_BETWEEN_TWO_PACKAGE
        },
        "image_processing": {
            "parent_object_list": config.PARENT_OBJECT_LIST,
            "remove_raw_image_when_dm_decoded": config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED,
            "enforce_parent_object": config.ENFORCE_PARENT_OBJECT
        },
        "datamatrix": {
            "chars_sizes": config.DM_CHARS_SIZES,
            "confidence_threshold": config.DM_CONFIDENCE_THRESHOLD,
            "overlap_threshold": config.DM_OVERLAP_THRESHOLD
        },
        "class_count_check": {
            "enabled": config.CHECK_CLASS_COUNTS_ENABLED,
            "classes": config.CHECK_CLASS_COUNTS_CLASSES,
            "confidence": config.CHECK_CLASS_COUNTS_CONFIDENCE
        },
        "light_control": {
            "status_check_enabled": config.LIGHT_STATUS_CHECK_ENABLED
        },
        "histogram": {
            "enabled": config.HISTOGRAM_ENABLED,
            "save_image": config.HISTOGRAM_SAVE_IMAGE
        },
        "store_annotation": {
            "enabled": config.STORE_ANNOTATION_ENABLED,
            "postgres_host": POSTGRES_HOST,
            "postgres_port": POSTGRES_PORT,
            "postgres_db": POSTGRES_DB
        },
        "data_file": DATA_FILE,
        "serial": {
            "port": WATCHER_USB,
            "baudrate": SERIAL_BAUDRATE,
            "mode": config.SERIAL_MODE
        },
        "cameras": {
            "cam_1": CAM_1_PATH,
            "cam_2": CAM_2_PATH,
            "cam_3": CAM_3_PATH,
            "cam_4": CAM_4_PATH
        },
        "redis": {
            "host": config.REDIS_HOST,
            "port": config.REDIS_PORT
        },
        "commands": USER_COMMANDS
    })


@router.post("/api/config")
async def update_config(request: Request, config_data: Dict[str, Any]):
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
    updated = {}

    try:
        # Ejector configuration
        if "ejector_enabled" in config_data:
            value = config_data["ejector_enabled"]
            config.EJECTOR_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["ejector_enabled"] = config.EJECTOR_ENABLED
            logger.info(f"Updated EJECTOR_ENABLED to {config.EJECTOR_ENABLED}")

        if "ejector_offset" in config_data:
            config.EJECTOR_OFFSET = int(config_data["ejector_offset"])
            updated["ejector_offset"] = config.EJECTOR_OFFSET
            logger.info(f"Updated EJECTOR_OFFSET to {config.EJECTOR_OFFSET}")

        if "ejector_duration" in config_data:
            config.EJECTOR_DURATION = float(config_data["ejector_duration"])
            updated["ejector_duration"] = config.EJECTOR_DURATION
            logger.info(f"Updated EJECTOR_DURATION to {config.EJECTOR_DURATION}")

        if "ejector_poll_interval" in config_data:
            config.EJECTOR_POLL_INTERVAL = float(config_data["ejector_poll_interval"])
            updated["ejector_poll_interval"] = config.EJECTOR_POLL_INTERVAL
            logger.info(f"Updated EJECTOR_POLL_INTERVAL to {config.EJECTOR_POLL_INTERVAL}")

        # Capture configuration
        if "time_between_packages" in config_data:
            config.TIME_BETWEEN_TWO_PACKAGE = float(config_data["time_between_packages"])
            config.time_between_two_package = config.TIME_BETWEEN_TWO_PACKAGE
            updated["time_between_packages"] = config.TIME_BETWEEN_TWO_PACKAGE
            logger.info(f"Updated TIME_BETWEEN_TWO_PACKAGE to {config.TIME_BETWEEN_TWO_PACKAGE}")

        if "capture_mode" in config_data:
            mode = config_data["capture_mode"]
            if mode in ["single", "multiple"]:
                config.CAPTURE_MODE = mode
                config.capture_mode = mode
                updated["capture_mode"] = config.CAPTURE_MODE
                logger.info(f"Updated CAPTURE_MODE to {config.CAPTURE_MODE}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid capture_mode: {mode}. Must be 'single' or 'multiple'")

        # Image processing configuration
        if "parent_object_list" in config_data:
            value = config_data["parent_object_list"]
            if isinstance(value, str):
                config.PARENT_OBJECT_LIST = [x.strip() for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                config.PARENT_OBJECT_LIST = value
            config.parent_object_list = config.PARENT_OBJECT_LIST
            updated["parent_object_list"] = config.PARENT_OBJECT_LIST
            logger.info(f"Updated PARENT_OBJECT_LIST to {config.PARENT_OBJECT_LIST}")

        if "remove_raw_image_when_dm_decoded" in config_data:
            value = config_data["remove_raw_image_when_dm_decoded"]
            config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED = str(value).lower() in ("true", "1", "yes")
            config.remove_raw_image_when_dm_decoded = config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED
            updated["remove_raw_image_when_dm_decoded"] = config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED
            logger.info(f"Updated REMOVE_RAW_IMAGE_WHEN_DM_DECODED to {config.REMOVE_RAW_IMAGE_WHEN_DM_DECODED}")

        # DataMatrix configuration
        if "dm_chars_sizes" in config_data:
            value = config_data["dm_chars_sizes"]
            if isinstance(value, str):
                config.DM_CHARS_SIZES = [int(x.strip()) for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                config.DM_CHARS_SIZES = [int(x) for x in value]
            updated["dm_chars_sizes"] = config.DM_CHARS_SIZES
            logger.info(f"Updated DM_CHARS_SIZES to {config.DM_CHARS_SIZES}")

        if "dm_confidence_threshold" in config_data:
            config.DM_CONFIDENCE_THRESHOLD = float(config_data["dm_confidence_threshold"])
            updated["dm_confidence_threshold"] = config.DM_CONFIDENCE_THRESHOLD
            logger.info(f"Updated DM_CONFIDENCE_THRESHOLD to {config.DM_CONFIDENCE_THRESHOLD}")

        if "dm_overlap_threshold" in config_data:
            config.DM_OVERLAP_THRESHOLD = float(config_data["dm_overlap_threshold"])
            updated["dm_overlap_threshold"] = config.DM_OVERLAP_THRESHOLD
            logger.info(f"Updated DM_OVERLAP_THRESHOLD to {config.DM_OVERLAP_THRESHOLD}")

        # Class count checking configuration
        if "check_class_counts_enabled" in config_data:
            value = config_data["check_class_counts_enabled"]
            config.CHECK_CLASS_COUNTS_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["check_class_counts_enabled"] = config.CHECK_CLASS_COUNTS_ENABLED
            logger.info(f"Updated CHECK_CLASS_COUNTS_ENABLED to {config.CHECK_CLASS_COUNTS_ENABLED}")

        if "check_class_counts_classes" in config_data:
            value = config_data["check_class_counts_classes"]
            if isinstance(value, str):
                config.CHECK_CLASS_COUNTS_CLASSES = [x.strip() for x in value.split(",") if x.strip()]
            elif isinstance(value, list):
                config.CHECK_CLASS_COUNTS_CLASSES = value
            updated["check_class_counts_classes"] = config.CHECK_CLASS_COUNTS_CLASSES
            logger.info(f"Updated CHECK_CLASS_COUNTS_CLASSES to {config.CHECK_CLASS_COUNTS_CLASSES}")

        if "check_class_counts_confidence" in config_data:
            value = config_data["check_class_counts_confidence"]
            config.CHECK_CLASS_COUNTS_CONFIDENCE = float(value)
            updated["check_class_counts_confidence"] = config.CHECK_CLASS_COUNTS_CONFIDENCE
            logger.info(f"Updated CHECK_CLASS_COUNTS_CONFIDENCE to {config.CHECK_CLASS_COUNTS_CONFIDENCE}")

        # Light control configuration
        if "light_status_check_enabled" in config_data:
            value = config_data["light_status_check_enabled"]
            config.LIGHT_STATUS_CHECK_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["light_status_check_enabled"] = config.LIGHT_STATUS_CHECK_ENABLED
            logger.info(f"Updated LIGHT_STATUS_CHECK_ENABLED to {config.LIGHT_STATUS_CHECK_ENABLED}")

        # Histogram configuration
        if "histogram_enabled" in config_data:
            value = config_data["histogram_enabled"]
            config.HISTOGRAM_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["histogram_enabled"] = config.HISTOGRAM_ENABLED
            logger.info(f"Updated HISTOGRAM_ENABLED to {config.HISTOGRAM_ENABLED}")

        if "histogram_save_image" in config_data:
            value = config_data["histogram_save_image"]
            config.HISTOGRAM_SAVE_IMAGE = str(value).lower() in ("true", "1", "yes")
            updated["histogram_save_image"] = config.HISTOGRAM_SAVE_IMAGE
            logger.info(f"Updated HISTOGRAM_SAVE_IMAGE to {config.HISTOGRAM_SAVE_IMAGE}")

        # Store annotation configuration
        if "store_annotation_enabled" in config_data:
            value = config_data["store_annotation_enabled"]
            config.STORE_ANNOTATION_ENABLED = str(value).lower() in ("true", "1", "yes")
            updated["store_annotation_enabled"] = config.STORE_ANNOTATION_ENABLED
            logger.info(f"Updated STORE_ANNOTATION_ENABLED to {config.STORE_ANNOTATION_ENABLED}")

        # Parent object enforcement configuration
        if "enforce_parent_object" in config_data:
            value = config_data["enforce_parent_object"]
            config.ENFORCE_PARENT_OBJECT = str(value).lower() in ("true", "1", "yes")
            updated["enforce_parent_object"] = config.ENFORCE_PARENT_OBJECT
            logger.info(f"Updated ENFORCE_PARENT_OBJECT to {config.ENFORCE_PARENT_OBJECT}")

        # Infrastructure configuration (Gradio/YOLO API)
        if "yolo_url" in config_data:
            config.YOLO_INFERENCE_URL = str(config_data["yolo_url"])
            updated["yolo_url"] = config.YOLO_INFERENCE_URL
            logger.info(f"Updated YOLO_INFERENCE_URL to {config.YOLO_INFERENCE_URL}")

        if "gradio_model" in config_data:
            config.GRADIO_MODEL = str(config_data["gradio_model"])
            updated["gradio_model"] = config.GRADIO_MODEL
            logger.info(f"Updated GRADIO_MODEL to {config.GRADIO_MODEL}")

        if "gradio_confidence" in config_data:
            config.GRADIO_CONFIDENCE_THRESHOLD = float(config_data["gradio_confidence"])
            updated["gradio_confidence"] = config.GRADIO_CONFIDENCE_THRESHOLD
            logger.info(f"Updated GRADIO_CONFIDENCE_THRESHOLD to {config.GRADIO_CONFIDENCE_THRESHOLD}")

        if "redis_host" in config_data:
            config.REDIS_HOST = str(config_data["redis_host"])
            updated["redis_host"] = config.REDIS_HOST
            logger.info(f"Updated REDIS_HOST to {config.REDIS_HOST}")

        if "redis_port" in config_data:
            config.REDIS_PORT = int(config_data["redis_port"])
            updated["redis_port"] = config.REDIS_PORT
            logger.info(f"Updated REDIS_PORT to {config.REDIS_PORT}")

        # Serial mode configuration
        if "serial_mode" in config_data:
            mode = str(config_data["serial_mode"])
            if mode in ["new", "legacy"]:
                config.SERIAL_MODE = mode
                # Update watcher if available
                watcher = request.app.state.watcher_instance
                if watcher:
                    watcher.serial_mode = config.SERIAL_MODE
                updated["serial_mode"] = config.SERIAL_MODE
                logger.info(f"Updated SERIAL_MODE to {config.SERIAL_MODE}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid serial_mode: {mode}. Must be 'new' or 'legacy'")

        # Shipment ID
        if "shipment" in config_data:
            shipment_id = str(config_data["shipment"]).strip() or "no_shipment"
            # Store in Redis db=3 (must match RedisConnection used by watcher)
            try:
                r = Redis("redis", 6379, db=3)
                r.set("shipment", shipment_id)
            except Exception as e:
                logger.warning(f"Failed to set shipment in Redis: {e}")
            watcher = request.app.state.watcher_instance
            if watcher:
                watcher.shipment = shipment_id
            # Create shipment directory
            try:
                import os
                os.makedirs(f"raw_images/{shipment_id}", exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create shipment directory: {e}")
            updated["shipment"] = shipment_id
            logger.info(f"Updated shipment ID to {shipment_id}")

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


# =============================================================================
# Data file endpoints
# =============================================================================

@router.get("/api/data-file")
async def get_data_file(request: Request):
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


@router.post("/api/data-file")
async def update_data_file(request: Request, data: Dict[str, Any]):
    """Update the DATA_FILE content and reload prepared_query_data.

    Expected body:
        - content: str - The new JSON content for the data file
    """
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
            request.app.state.prepared_query_data = new_data
        elif isinstance(new_data, dict) and "prepared_query_data" in new_data:
            request.app.state.prepared_query_data = new_data["prepared_query_data"]
        else:
            request.app.state.prepared_query_data = new_data
        logger.info(f"Updated DATA_FILE ({DATA_FILE}) with {len(request.app.state.prepared_query_data)} entries")

        return JSONResponse(content={
            "status": "ok",
            "file_path": DATA_FILE,
            "entries_count": len(request.app.state.prepared_query_data),
            "backup_path": backup_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating data file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update data file: {str(e)}")


# =============================================================================
# Service config persistence endpoints
# =============================================================================

@router.post("/api/save_service_config")
async def api_save_service_config(request: Request):
    """Save current service configuration to file."""
    try:
        # Build current config from runtime state (preserves unsaved changes)
        svc_config = build_current_service_config(request.app.state)
        if not svc_config:
            svc_config = load_service_config() or {}
        if save_service_config(svc_config):
            return JSONResponse(content={"success": True, "message": "Service configuration saved"})
        else:
            return JSONResponse(content={"error": "Failed to save configuration"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving service config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/save_data_file")
async def api_save_data_file(request: Request):
    """Save current data file (same as service config)."""
    try:
        # Build current config from runtime state (preserves unsaved changes)
        svc_config = build_current_service_config(request.app.state)
        if not svc_config:
            svc_config = load_service_config() or {}
        if save_service_config(svc_config):
            return JSONResponse(content={"success": True, "message": "Data file saved"})
        else:
            return JSONResponse(content={"error": "Failed to save data file"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving data file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/load_service_config")
async def api_load_service_config(request: Request):
    """Load service configuration from file (triggers page reload on client)."""
    try:
        svc_config = load_service_config()
        if svc_config:
            return JSONResponse(content={"success": True, "message": "Service configuration loaded"})
        else:
            return JSONResponse(content={"error": "No saved configuration found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error loading service config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/export_service_config")
async def api_export_service_config(request: Request):
    """Export current service configuration as downloadable JSON file."""
    try:
        svc_config = load_service_config() or {}

        # Create JSON response with proper headers for download
        json_str = json.dumps(svc_config, indent=2)
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
