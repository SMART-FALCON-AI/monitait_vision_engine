from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
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
        r = Redis("redis", 6379, db=config.REDIS_DB)
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
            "delay": config.EJECTOR_DELAY,
            "duration": config.EJECTOR_DURATION,
            "poll_interval": config.EJECTOR_POLL_INTERVAL
        },
        "capture": {
            "mode": config.CAPTURE_MODE,
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

    # --- Preserve persisted-only / user-selected fields (3.19.1) ---
    # This function rebuilds service_config from runtime state, but several
    # fields live ONLY in the saved file and have no runtime source here:
    #   - inference.current_module + module URLs (the literal above hardcodes
    #     gradio_hf, which would reset the user's active model on every save)
    #   - store_objects (per-class DB Store flags)
    #   - audio_settings (per-class Show/Narrate/Beep/min_confidence)
    # Without this, "Save All Configuration" (and camera saves) silently wiped
    # the active inference module and every per-class Store/audio flag. Carry
    # them over from the persisted config instead of clobbering them.
    existing = load_service_config() or {}
    if existing.get("inference"):
        svc_config["inference"] = existing["inference"]
    for _preserved in ("store_objects", "audio_settings"):
        if _preserved in existing:
            svc_config[_preserved] = existing[_preserved]

    # 4.0.17 — current_shipment ALSO has no source in the literal above.
    # Without this, any auto-save (camera edit, AI model change, state
    # change) silently wipes the active shipment from DB. Next restart then
    # sees current_shipment=None and skips the restore-on-boot path, so the
    # dashboard reverts to "no_shipment" even though Redis still had the
    # live value before the restart.
    #
    # Order of preference:
    #   1. watcher.shipment if set to a real (non-sentinel) value — captures
    #      a freshly-set shipment that hasn't been persisted yet.
    #   2. existing saved value if it's a real shipment — protects against
    #      the boot-race window where watcher.shipment is still None because
    #      the restore-on-boot hook hasn't fired yet but auto-save did.
    #   3. otherwise omit, so an explicit "no_shipment" propagates.
    try:
        watcher = getattr(app_state, 'watcher_instance', None)
        live_ship = (getattr(watcher, 'shipment', None) or "").strip() if watcher else ""
        exist_ship = (existing.get("current_shipment") or "").strip()
        if live_ship and live_ship != "no_shipment":
            svc_config["current_shipment"] = live_ship
        elif exist_ship and exist_ship != "no_shipment":
            svc_config["current_shipment"] = exist_ship
    except Exception:
        # Never block service-config save on this best-effort preservation.
        pass

    # 4.0.21 — same-shape regression for `ai_trainer` and `notifications` (and
    # anything else routers save under their own top-level key). The literal
    # above doesn't generate these, so every auto-save (camera edit, AI model
    # change, etc.) was silently wiping them. After restart the operator saw
    # the AI Trainer task_id back to empty and got "no task_id configured"
    # when clicking a chart dot, and Telegram/notification schedules vanished.
    # Generic preservation: ANY top-level key that exists in the saved file
    # but is NOT rebuilt by this function survives. The runtime-rebuilt keys
    # are intentional overwrites (see literal at top); everything else
    # belongs to a router that owns its own save path.
    _runtime_rebuilt = {
        "cameras", "infrastructure", "inference", "ai", "ejector", "capture",
        "image_processing", "histogram", "class_count_check", "datamatrix",
        "store_annotation", "states", "current_state_name", "pipeline_config",
        # Explicitly-preserved above:
        "store_objects", "audio_settings", "current_shipment",
    }
    for k, v in existing.items():
        if k not in svc_config and k not in _runtime_rebuilt:
            svc_config[k] = v

    return svc_config


# =============================================================================
# Configuration endpoints
# =============================================================================

@router.get("/config")
def get_config(request: Request):
    """Get current configuration."""
    return JSONResponse(content={
        "ejector": {
            "enabled": config.EJECTOR_ENABLED,
            "offset": config.EJECTOR_OFFSET,
            "delay": config.EJECTOR_DELAY,
            "duration": config.EJECTOR_DURATION,
            "poll_interval": config.EJECTOR_POLL_INTERVAL
        },
        "capture": {
            "mode": config.CAPTURE_MODE,
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
def update_config(request: Request, config_data: Dict[str, Any], background_tasks: BackgroundTasks = None):
    """Update counter service configuration at runtime.

    Supported keys:
        - ejector_offset: int - Encoder counts from camera to ejector
        - ejector_delay: float - Seconds to wait before firing ejector after encoder target reached
        - ejector_duration: float - Seconds to run ejector motor
        - ejector_poll_interval: float - Seconds between ejector checks

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

        if "ejector_delay" in config_data:
            config.EJECTOR_DELAY = float(config_data["ejector_delay"])
            updated["ejector_delay"] = config.EJECTOR_DELAY
            logger.info(f"Updated EJECTOR_DELAY to {config.EJECTOR_DELAY}")

        if "ejector_duration" in config_data:
            config.EJECTOR_DURATION = float(config_data["ejector_duration"])
            updated["ejector_duration"] = config.EJECTOR_DURATION
            logger.info(f"Updated EJECTOR_DURATION to {config.EJECTOR_DURATION}")

        if "ejector_poll_interval" in config_data:
            config.EJECTOR_POLL_INTERVAL = float(config_data["ejector_poll_interval"])
            updated["ejector_poll_interval"] = config.EJECTOR_POLL_INTERVAL
            logger.info(f"Updated EJECTOR_POLL_INTERVAL to {config.EJECTOR_POLL_INTERVAL}")

        # Capture configuration
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
            # 4.0.60 — REORDERED. Previously this handler did the persistence
            # (save_service_config → JSON read + DB round-trip + file copy +
            # fsync + os.replace) BEFORE flipping `watcher.shipment`, all on
            # the FastAPI event loop. That produced two bad symptoms:
            #   (a) Every other endpoint (SSE, dashboard polls, CSV export)
            #       stalled while the disk write ran — the operator saw the
            #       whole UI freeze after scanning.
            #   (b) During that window Redis said NEW while `watcher.shipment`
            #       said OLD, so detections landed with the new tag but the
            #       status endpoint reported the old shipment — a torn state
            #       that made CSV rows disagree with what the UI just showed.
            # New order: (1) mutate in-memory watcher, (2) set Redis (single
            # fast RTT), (3) offload persistence to BackgroundTasks so the
            # response returns immediately and other requests don't block.
            watcher = request.app.state.watcher_instance
            # Capture the new baseline encoder ONLY on actual change so
            # re-POSTing the same shipment (dashboard drift refresh) doesn't
            # zero the Length tile mid-shipment.
            new_start_encoder = None
            if watcher and shipment_id != getattr(watcher, "shipment", None):
                new_start_encoder = int(getattr(watcher, "encoder_value", 0) or 0)
                watcher.shipment_start_encoder = new_start_encoder
                logger.info(
                    f"Shipment start encoder: {new_start_encoder} "
                    f"(API set shipment → {shipment_id})"
                )
            # (1) live in-memory FIRST — the detection loop and /api/status
            # read this, so setting it here makes the change visible to the
            # very next frame before we touch anything slower.
            if watcher:
                watcher.shipment = shipment_id
            # (2) Redis — one round-trip, keeps cross-process consumers in
            # sync. Failure here is logged but non-fatal: the in-memory flip
            # above already made the change visible to the running pipeline.
            try:
                r = Redis("redis", 6379, db=config.REDIS_DB)
                r.set("shipment", shipment_id)
            except Exception as e:
                logger.warning(f"Failed to set shipment in Redis: {e}")
            # (3) persistence — offloaded so the operator's POST returns
            # immediately. On boot main.py reads service_config to prime
            # watcher + Redis, so the persistence still restores across
            # `docker compose down` / restart — it just doesn't block the
            # request thread anymore.
            def _persist_shipment(sid: str, sse: "int | None"):
                try:
                    from config import load_service_config, save_service_config
                    _svc = load_service_config() or {}
                    _svc["current_shipment"] = sid
                    if sse is not None:
                        _svc["shipment_start_encoder"] = sse
                    save_service_config(_svc)
                except Exception as _pe:
                    logger.warning(f"Failed to persist current_shipment to DB: {_pe}")
                try:
                    import os as _os
                    _os.makedirs(f"raw_images/{sid}", exist_ok=True)
                except Exception as _me:
                    logger.warning(f"Failed to create shipment directory: {_me}")
            if background_tasks is not None:
                background_tasks.add_task(_persist_shipment, shipment_id, new_start_encoder)
            else:
                # BackgroundTasks unavailable (should not happen in normal
                # FastAPI request context; kept as defensive fallback).
                _persist_shipment(shipment_id, new_start_encoder)
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
# Encoder calibration (3.21.14) — drives Shipment Quality Score normalization
# =============================================================================

@router.get("/api/encoder_calibration")
def get_encoder_calibration():
    """Return the operator-set encoder unit label and encoder-pulses-per-unit conversion.

    v4.0.140 — decoupled from the "meter" hardcode. `encoder_unit` is now a
    free-form display label (Meter, Batch, Cubic Meter, mm, tile, …) and
    `encoder_units_per_unit` is how many raw encoder pulses accumulate for
    ONE of those units. The legacy field `encoder_units_per_meter` is still
    read as a fallback when `encoder_units_per_unit` isn't present, so
    upgrade requires zero operator action. Both new and legacy fields are
    returned so older frontends keep working during the rollout window.
    """
    try:
        svc = load_service_config() or {}
        # v4.0.140 — new field wins; fall back to legacy so sites that haven't
        # touched Advanced tab post-upgrade still display correctly.
        upu_raw = svc.get("encoder_units_per_unit")
        if upu_raw is None:
            upu_raw = svc.get("encoder_units_per_meter")
        try:
            upu_val = float(upu_raw or 0) or None
        except (TypeError, ValueError):
            upu_val = None
        return JSONResponse(content={
            "encoder_unit": str(svc.get("encoder_unit") or "encoder_unit"),
            "encoder_units_per_unit":  upu_val,
            "encoder_units_per_meter": upu_val,   # legacy alias — one-release compat
        })
    except Exception as e:
        logger.error(f"get_encoder_calibration error: {e}")
        return JSONResponse(content={
            "encoder_unit": "encoder_unit",
            "encoder_units_per_unit":  None,
            "encoder_units_per_meter": None,
        })


# 4.0.32 — operator-set per-camera L*a*b* "target" baseline for the heatmap's
# 🎯 Target mode. Each camera gets a {L, a, b} tuple (E is derived). When
# set, the heatmap's 🎯 Target button enables and clicking it makes every
# cell's ΔE relative to its camera's target colour instead of the window
# median. Manual gold-standard for cross-shipment comparison.
@router.get("/api/color_target")
def get_color_target():
    """Return the per-camera colour target. Empty dict when nothing set."""
    try:
        svc = load_service_config() or {}
        return JSONResponse(content={"color_target": svc.get("color_target") or {}})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/color_target")
def set_color_target(payload: Dict[str, Any]):
    """Set per-camera L*a*b* target.
    Body: { "color_target": { "<cam_id>": {"L": <0-100>, "a": <-128..127>, "b": <-128..127>} } }
       or to clear: { "color_target": {} }"""
    try:
        raw = payload.get("color_target")
        if not isinstance(raw, dict):
            return JSONResponse(content={"error": "color_target must be an object"}, status_code=400)
        clean = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            try:
                cam_id = str(int(k))
                L = max(0.0, min(100.0, float(v.get("L") or 0)))
                a = max(-128.0, min(127.0, float(v.get("a") or 0)))
                b = max(-128.0, min(127.0, float(v.get("b") or 0)))
                E = (L*L + a*a + b*b) ** 0.5
                clean[cam_id] = {
                    "L": round(L, 2), "a": round(a, 2), "b": round(b, 2),
                    "E": round(E, 2),
                }
            except (TypeError, ValueError):
                continue
        svc = load_service_config() or {}
        if clean:
            svc["color_target"] = clean
        else:
            svc.pop("color_target", None)
        save_service_config(svc)
        return JSONResponse(content={"status": "ok", "color_target": clean})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 4.0.31 — operator-picked baseline reference point for the heatmap's
# "🖼️ Reference" mode. The operator clicks somewhere on the chart and the
# colour at that position becomes the baseline. We store the axis (encoder
# vs time) + the picked value + an optional window so the baseline computation
# can grab a small range around the click instead of a single frame.
@router.get("/api/color_reference_position")
def get_color_reference_position():
    """Return the stored reference position (or null if unset)."""
    try:
        svc = load_service_config() or {}
        pos = svc.get("color_reference_position")
        return JSONResponse(content={"color_reference_position": pos})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/color_reference_position")
def set_color_reference_position(payload: Dict[str, Any]):
    """Set the operator-picked baseline reference point.
    Body: {axis: "encoder"|"time", value: <float>, window: <float, optional>}
       - encoder: value is the encoder count clicked, window = +/-encoder counts
       - time:    value is epoch-ms,                 window = +/-milliseconds
    Setting `value=null` clears the reference (back to "unconfigured")."""
    try:
        if payload.get("value") in (None, ""):
            svc = load_service_config() or {}
            if "color_reference_position" in svc:
                svc.pop("color_reference_position", None)
                save_service_config(svc)
            return JSONResponse(content={"status": "cleared"})

        axis = str(payload.get("axis") or "").lower().strip()
        if axis not in ("encoder", "time"):
            return JSONResponse(content={"error": "axis must be 'encoder' or 'time'"}, status_code=400)
        try:
            value = float(payload["value"])
        except (TypeError, ValueError):
            return JSONResponse(content={"error": "value must be a number"}, status_code=400)
        try:
            window = float(payload.get("window") or 0)
        except (TypeError, ValueError):
            window = 0.0
        import time as _t
        svc = load_service_config() or {}
        svc["color_reference_position"] = {
            "axis": axis,
            "value": value,
            "window": max(0.0, window),
            "set_at": int(_t.time()),
        }
        save_service_config(svc)
        return JSONResponse(content={
            "status": "ok",
            "color_reference_position": svc["color_reference_position"],
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 4.0.28 — manual override for `score_scale_factor`. The 🎯 Calibrate score
# button on the Charts tab auto-tunes this against recent shipments; this
# pair lets the operator read + write it explicitly without going through the
# auto-tuner (useful when they want a specific value, or when calibration's
# 7-day window doesn't reflect what they want to score against).
@router.get("/api/score_scale_factor")
def get_score_scale_factor():
    """Return the current `score_scale_factor` (default 1.0)."""
    try:
        svc = load_service_config() or {}
        return JSONResponse(content={
            "score_scale_factor": float(svc.get("score_scale_factor") or 1.0),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/score_scale_factor")
def set_score_scale_factor(payload: Dict[str, Any]):
    """Manually set `score_scale_factor`. Same bounds as the auto-calibrator
    (0.001 .. 1e9) so a typo can't poison the score for every shipment."""
    try:
        raw = payload.get("score_scale_factor", payload.get("value"))
        val = float(raw)
        if val <= 0:
            return JSONResponse(content={"error": "score_scale_factor must be > 0"}, status_code=400)
        val = max(0.001, min(1e9, val))
        svc = load_service_config() or {}
        svc["score_scale_factor"] = round(val, 4)
        save_service_config(svc)
        return JSONResponse(content={
            "status": "ok",
            "score_scale_factor": svc["score_scale_factor"],
        })
    except (TypeError, ValueError):
        return JSONResponse(content={"error": "score_scale_factor must be a number"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/encoder_calibration")
def set_encoder_calibration(payload: Dict[str, Any]):
    """Set the encoder calibration (unit label + encoder-pulses-per-unit).

    v4.0.140 — accepts new `encoder_units_per_unit`, still accepts legacy
    `encoder_units_per_meter` alias for one release so an older frontend
    talking to a newer backend keeps working. Writes BOTH fields on save
    (identical value) so any downstream integration reading either name
    keeps seeing the current calibration.

    Label sanitization was widened: letters, digits, spaces, and a few
    punctuation marks are now allowed so operators can write "Cubic Meter"
    or "Roll #" instead of being reduced to alphanumerics.
    """
    try:
        svc = load_service_config() or {}
        if "encoder_unit" in payload:
            unit = str(payload["encoder_unit"] or "encoder_unit").strip() or "encoder_unit"
            # v4.0.140 — allow spaces + a small punctuation whitelist so
            # multi-word labels ("Cubic Meter", "Batch #A") are preserved.
            unit = "".join(c for c in unit if c.isalnum() or c.isspace() or c in "_-/.#+()").strip()[:32] or "encoder_unit"
            svc["encoder_unit"] = unit
        # v4.0.140 — accept either name; new field wins when both present.
        raw = None
        if "encoder_units_per_unit" in payload:
            raw = payload["encoder_units_per_unit"]
        elif "encoder_units_per_meter" in payload:
            raw = payload["encoder_units_per_meter"]
        if raw is not None:
            try:
                v = float(raw or 0)
                v = max(0.0, v) if v > 0 else 0
                svc["encoder_units_per_unit"]  = v
                svc["encoder_units_per_meter"] = v   # keep legacy in sync one release
            except (TypeError, ValueError):
                pass
        save_service_config(svc)
        upu_saved = svc.get("encoder_units_per_unit") or None
        return JSONResponse(content={
            "status": "ok",
            "encoder_unit": svc.get("encoder_unit", "encoder_unit"),
            "encoder_units_per_unit":  upu_saved,
            "encoder_units_per_meter": upu_saved,   # legacy alias
        })
    except Exception as e:
        logger.error(f"set_encoder_calibration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# v4.0.153 — Max shipment length auto-cutoff
# =============================================================================

_VALID_CUTOFF_METRICS = ("encoder", "time_sec", "length")


@router.get("/api/auto_cutoff_config")
def get_auto_cutoff_config():
    """v4.0.154 — return operator-configured auto-cutoff fields.

    Fields:
      - `auto_cutoff_metric`   : "encoder" | "time_sec" | "length" (default "length")
      - `auto_cutoff_threshold`: numeric threshold in the chosen metric's unit
                                 (encoder pulses, seconds, or display unit)
      - `auto_shipment_id_prefix`: prefix for the auto-generated shipment id
      - `auto_cutoff_cooldown_sec`: minimum seconds between two auto-cutoffs

    Legacy: v4.0.153's `max_shipment_length` field is still recognised and
    returned in the payload for one release so an older frontend keeps
    working. New callers should use `auto_cutoff_threshold`.
    """
    try:
        svc = load_service_config() or {}
        # v4.0.155 — new threshold field wins; fall back to the v4.0.153 alias.
        thr_raw = svc.get("auto_cutoff_threshold")
        if thr_raw is None:
            thr_raw = svc.get("max_shipment_length")
        try:
            threshold = float(thr_raw) if thr_raw is not None else 0.0
        except (TypeError, ValueError):
            threshold = 0.0
        metric = str(svc.get("auto_cutoff_metric") or "length").strip().lower()
        if metric not in _VALID_CUTOFF_METRICS:
            metric = "length"
        # v4.0.155 — explicit enabled bool. Legacy migration: when unset,
        # infer from the pre-v4.0.155 sentinel (threshold > 0 meant enabled).
        enabled_raw = svc.get("auto_cutoff_enabled")
        if enabled_raw is None:
            enabled = threshold > 0
        else:
            enabled = bool(enabled_raw)
        try:
            cooldown = float(svc.get("auto_cutoff_cooldown_sec") or 5)
        except (TypeError, ValueError):
            cooldown = 5.0
        return JSONResponse(content={
            "auto_cutoff_enabled": enabled,
            "auto_cutoff_metric": metric,
            "auto_cutoff_threshold": threshold,
            "auto_shipment_id_prefix": str(svc.get("auto_shipment_id_prefix") or "MR"),
            "auto_cutoff_cooldown_sec": cooldown,
            "encoder_unit": str(svc.get("encoder_unit") or "units"),
            "max_shipment_length": threshold if metric == "length" else 0,
        })
    except Exception as e:
        logger.error(f"get_auto_cutoff_config error: {e}")
        return JSONResponse(content={
            "auto_cutoff_enabled": False,
            "auto_cutoff_metric": "length",
            "auto_cutoff_threshold": 0,
            "auto_shipment_id_prefix": "MR",
            "auto_cutoff_cooldown_sec": 5.0,
            "encoder_unit": "units",
            "max_shipment_length": 0,
        })


@router.post("/api/auto_cutoff_config")
def set_auto_cutoff_config(payload: Dict[str, Any]):
    """v4.0.154 — persist auto-cutoff fields. Dual-writes DB + file.

    All fields optional; unspecified fields keep their current value. Invalid
    values are silently ignored so a malformed request can never poison the
    config file.

    Accepts `auto_cutoff_threshold` (new) and `max_shipment_length` (legacy
    alias). If both are given, new wins.
    """
    try:
        svc = load_service_config() or {}
        # v4.0.155 — explicit boolean enable toggle.
        if "auto_cutoff_enabled" in payload:
            svc["auto_cutoff_enabled"] = bool(payload["auto_cutoff_enabled"])
        # Metric
        if "auto_cutoff_metric" in payload:
            m = str(payload["auto_cutoff_metric"] or "").strip().lower()
            if m in _VALID_CUTOFF_METRICS:
                svc["auto_cutoff_metric"] = m
        # Threshold (new field wins over legacy alias in the same payload).
        # v4.0.155 — threshold is no longer the on/off signal, so it accepts
        # any float. Negative values are allowed (though unusual) so a real
        # negative encoder-wrap situation can be handled by the operator.
        thr_key = None
        if "auto_cutoff_threshold" in payload:
            thr_key = "auto_cutoff_threshold"
        elif "max_shipment_length" in payload:
            thr_key = "max_shipment_length"
        if thr_key is not None:
            try:
                v = float(payload[thr_key])
                svc["auto_cutoff_threshold"] = v
                svc["max_shipment_length"] = v   # keep legacy alias in sync
            except (TypeError, ValueError):
                pass
        if "auto_shipment_id_prefix" in payload:
            p = str(payload["auto_shipment_id_prefix"] or "").strip()
            p = "".join(c for c in p if c.isalnum() or c in "-_.")[:16] or "MR"
            svc["auto_shipment_id_prefix"] = p
        if "auto_cutoff_cooldown_sec" in payload:
            try:
                c = float(payload["auto_cutoff_cooldown_sec"])
                svc["auto_cutoff_cooldown_sec"] = max(0.5, c)
            except (TypeError, ValueError):
                pass
        save_service_config(svc)
        return JSONResponse(content={
            "status": "ok",
            "auto_cutoff_enabled": bool(svc.get("auto_cutoff_enabled", False)),
            "auto_cutoff_metric": svc.get("auto_cutoff_metric", "length"),
            "auto_cutoff_threshold": svc.get("auto_cutoff_threshold", 0),
            "auto_shipment_id_prefix": svc.get("auto_shipment_id_prefix", "MR"),
            "auto_cutoff_cooldown_sec": svc.get("auto_cutoff_cooldown_sec", 5.0),
            "max_shipment_length": svc.get("max_shipment_length", 0),
        })
    except Exception as e:
        logger.error(f"set_auto_cutoff_config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Storage path (DATA_ROOT) endpoints (3.21.11)
# =============================================================================
# The raw_images bind-mount is `${DATA_ROOT:-.}/raw_images:/code/raw_images`.
# This pair of endpoints lets the UI read/write the DATA_ROOT value in the
# compose .env file. The change only takes effect after the user restarts the
# MVE container (since bind-mounts are baked in at container creation).

import os as _os
from pathlib import Path as _Path

# Compose .env lives one level up from /code (vision_engine/) — at the project
# root. In a container that's /code/../.env. On host it's the bind-mount source.
def _compose_env_path():
    # Try common locations
    candidates = [
        "/host_compose/.env",                                  # if user adds this bind-mount
        "/code/../.env",                                       # relative from /code
        _os.environ.get("COMPOSE_ENV_FILE", "").strip(),       # explicit override
    ]
    for c in candidates:
        if c and _os.path.isfile(c):
            return c
    return None


def _read_env_var(path, key):
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _write_env_var(path, key, value):
    """Update or append KEY=value in .env. Preserves all other lines."""
    lines = []
    found = False
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
        if not found:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key}={value}\n")
        with open(path, "w") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        logger.error(f"Failed to write {key} to {path}: {e}")
        return False


@router.get("/api/data_root")
def get_data_root():
    """Read DATA_ROOT from the compose .env file. Default = /mnt/SSD-RESERVE."""
    env_path = _compose_env_path()
    current = None
    if env_path:
        current = _read_env_var(env_path, "DATA_ROOT")
    return JSONResponse(content={
        "data_root":   current or "/mnt/SSD-RESERVE",
        "env_file":    env_path,
        "default":     "/mnt/SSD-RESERVE",
        "note":        "Change requires MVE container restart to take effect.",
    })


@router.post("/api/data_root")
def set_data_root(request: Request, body: Dict[str, Any]):
    """Set DATA_ROOT in the compose .env file. Validates path + writability."""
    new_path = (body.get("path") or "").strip()
    if not new_path:
        raise HTTPException(status_code=400, detail="path is required")
    if not new_path.startswith("/"):
        raise HTTPException(status_code=400, detail="path must be absolute (start with /)")
    if any(c in new_path for c in (";", "&", "|", "$", "`", "\n", "\r")):
        raise HTTPException(status_code=400, detail="path contains invalid characters")

    env_path = _compose_env_path()
    if not env_path:
        raise HTTPException(
            status_code=500,
            detail="compose .env file not found. Mount host's compose dir to /host_compose or set COMPOSE_ENV_FILE env var.",
        )

    # Best-effort path-exists check (the container can't verify host paths
    # accurately, so don't fail on missing — just warn).
    warning = None
    target_marker = _Path("/host_data_root") / new_path.lstrip("/")
    if not target_marker.exists() and not new_path.startswith("/mnt/"):
        warning = f"Path {new_path} may not exist on host. Verify before restart."

    if not _write_env_var(env_path, "DATA_ROOT", new_path):
        raise HTTPException(status_code=500, detail="Failed to write to .env file")

    return JSONResponse(content={
        "status":   "ok",
        "data_root": new_path,
        "env_file": env_path,
        "warning":  warning,
        "next":     "Run `docker compose -p monitaqc up -d --force-recreate monitait_vision_engine` to apply.",
    })


# =============================================================================
# Data file endpoints
# =============================================================================

@router.get("/api/data-file")
def get_data_file(request: Request):
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
def update_data_file(request: Request, data: Dict[str, Any]):
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
def api_save_service_config(request: Request):
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
def api_save_data_file(request: Request):
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
def api_load_service_config(request: Request):
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
def api_export_service_config(request: Request):
    """Export the COMPLETE configuration as a downloadable JSON file.

    Previously this dumped only `service_config` (via load_service_config), which
    silently dropped root-level keys — most importantly `timeline_config` (which
    holds the **ejection procedures** + timeline object filters). A user who
    exported, wiped, then re-imported would lose all procedures. We now export
    the ENTIRE data file (`load_data_file()`), so every top-level section round
    -trips: service_config (cameras, states, pipeline_config, store_objects,
    audio_settings, …) AND timeline_config (procedures) AND anything else.

    The import handler (`POST /api/cameras/config/upload`) detects this full
    -bundle shape (presence of a `service_config` key) and restores all of it.
    """
    try:
        from config import load_data_file
        full = load_data_file() or {}
        # Defensive: if the file somehow has no service_config wrapper (very old
        # flat format), fall back to wrapping the service config so the export is
        # still a valid full-bundle the importer understands.
        if "service_config" not in full:
            full = {"service_config": load_service_config() or {}}

        json_str = json.dumps(full, indent=2)
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


# =============================================================================
# Per-object DB-persistence opt-in (`Store` checkbox in Process tab UI)
# =============================================================================
# Each class name maps to bool. Detections of classes with True are written to
# `inference_results`; everything else is dropped at the gate in
# services/detection.py. Default is False (no storage) — explicit opt-in only.
# Persisted under service_config["store_objects"] so it survives restarts.

@router.get("/api/store_objects")
def api_get_store_objects():
    """Return the per-class store flags (server-side source of truth)."""
    try:
        svc = load_service_config() or {}
        return JSONResponse(content={"store_objects": svc.get("store_objects", {})})
    except Exception as e:
        logger.error(f"Error reading store_objects: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# Per-object audio/display settings (`Show` / `Narrate` / `Beep` / `Min conf`)
# =============================================================================
# Mirror of the Store endpoint above. Previously these toggles lived only in
# browser localStorage (vision_engine/static/js/audio.js → audioSettings),
# which meant an AI agent or another browser couldn't read/change them. Now
# they're persisted server-side under service_config["audio_settings"] so the
# /api/ai_query agentic loop can adjust per-class alarm behavior via REST.
#
# Schema: { className: { show: bool, narrate: bool, beep: bool, min_confidence: float } }
# All four fields optional; missing fields use safe defaults (show=true,
# narrate=false, beep=false, min_confidence=0.01).

@router.get("/api/audio_settings")
def api_get_audio_settings():
    """Return per-class audio/display settings (server-side source of truth)."""
    try:
        svc = load_service_config() or {}
        return JSONResponse(content={"audio_settings": svc.get("audio_settings", {})})
    except Exception as e:
        logger.error(f"Error reading audio_settings: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/audio_settings")
def api_set_audio_settings(payload: Dict[str, Any]):
    """Update per-class audio settings. Accepts either:
       - {"class_name": "DIE_LINE", "show": true, "narrate": false, "beep": true, "min_confidence": 0.5}
         (single-class partial update — only provided fields are changed)
       - {"audio_settings": {"DIE_LINE": {...}, "HOLE": {...}, ...}}  (bulk replace)
    """
    try:
        svc = load_service_config() or {}
        settings = svc.get("audio_settings", {})

        # 4.0.26 — semantic ROLE per class. Replaces the global parent_object_list
        # text-box. Valid values:
        #   "context" (default) — informational, no special wiring
        #   "defect"  — counted into the quality score / NG counter
        #   "parent"  — defines inspection area (`parent_object_list` is derived
        #               from any class with role=parent; falls back to ['_root']
        #               when no class is flagged parent)
        #   "marker"  — encoder-axis landmark, not scored
        _VALID_ROLES = {"context", "defect", "parent", "marker"}
        def _clean_role(v):
            s = str(v or "").strip().lower()
            return s if s in _VALID_ROLES else "context"

        if "audio_settings" in payload and isinstance(payload["audio_settings"], dict):
            # Bulk replace — caller owns the full mapping
            new_settings = {}
            for cls, cfg in payload["audio_settings"].items():
                if not isinstance(cfg, dict):
                    continue
                role = _clean_role(cfg.get("role", "context"))
                # 4.0.26 — role=parent forces severity to 0 so the impact-score
                # numbers stay clean even if the caller sent a non-zero value.
                sev = 0 if role == "parent" else max(0, min(100, int(cfg.get("severity", 0) or 0)))
                new_settings[str(cls)] = {
                    "show": bool(cfg.get("show", True)),
                    "narrate": bool(cfg.get("narrate", False)),
                    "beep": bool(cfg.get("beep", False)),
                    "min_confidence": float(cfg.get("min_confidence", 0.01)),
                    # 3.21.12 — per-class severity weight (0-100). Default 0 = no impact.
                    "severity": sev,
                    # 4.0.26 — context | defect | parent | marker
                    "role": role,
                    # 3.22.2 — ColorE: track CIELAB ΔE drift over time for this class.
                    "color_e": bool(cfg.get("color_e", False)),
                    # 3.22.3 — Area: show bbox-area percentiles for this class on the card.
                    "area": bool(cfg.get("area", False)),
                }
            settings = new_settings
        elif "class_name" in payload:
            cls = str(payload["class_name"])
            entry = settings.get(cls, {"show": True, "narrate": False, "beep": False, "min_confidence": 0.01, "severity": 0, "role": "context", "color_e": False, "area": False})
            for k in ("show", "narrate", "beep", "color_e", "area"):
                if k in payload:
                    entry[k] = bool(payload[k])
            if "min_confidence" in payload:
                try:
                    entry["min_confidence"] = float(payload["min_confidence"])
                except (TypeError, ValueError):
                    pass
            if "role" in payload:
                entry["role"] = _clean_role(payload["role"])
            if "severity" in payload:
                try:
                    entry["severity"] = max(0, min(100, int(payload["severity"] or 0)))
                except (TypeError, ValueError):
                    pass
            # 4.0.26 — enforce the parent-zeroes-severity invariant even when
            # the partial update only changes role (not severity).
            if entry.get("role") == "parent":
                entry["severity"] = 0
            settings[cls] = entry
        else:
            return JSONResponse(
                content={"error": "expected {class_name, show?, narrate?, beep?, min_confidence?} or {audio_settings: {...}}"},
                status_code=400,
            )

        svc["audio_settings"] = settings
        save_service_config(svc)
        # 3.21.26 — kick the draw_filters cache so the new show/min_conf takes
        # effect on the very next captured frame instead of waiting for TTL.
        try:
            from services.draw_filters import invalidate_cache as _df_inv
            _df_inv()
        except Exception:
            pass
        return JSONResponse(content={"success": True, "audio_settings": settings})
    except Exception as e:
        logger.error(f"Error writing audio_settings: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/store_objects")
def api_set_store_objects(payload: Dict[str, Any]):
    """Toggle/set per-class store flag. Accepts either:
       - {"class_name": "DIE_LINE", "store": true}  (single-class update)
       - {"store_objects": {"DIE_LINE": true, "HOLE": false, ...}}  (bulk set)
    """
    try:
        svc = load_service_config() or {}
        store_map = svc.get("store_objects", {})

        if "store_objects" in payload and isinstance(payload["store_objects"], dict):
            # Bulk replace
            store_map = {k: bool(v) for k, v in payload["store_objects"].items()}
        elif "class_name" in payload:
            cls = str(payload["class_name"])
            store_map[cls] = bool(payload.get("store", False))
        else:
            return JSONResponse(
                content={"error": "expected {class_name, store} or {store_objects: {...}}"},
                status_code=400,
            )

        svc["store_objects"] = store_map
        save_service_config(svc)
        return JSONResponse(content={"success": True, "store_objects": store_map})
    except Exception as e:
        logger.error(f"Error writing store_objects: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
