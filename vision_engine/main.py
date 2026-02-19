import time
import json
import threading
import os
import logging
import queue
import psutil
from fastapi import FastAPI
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all configuration (temporary bridge for refactoring)
from config import *
import config as cfg_module  # Direct module ref for syncing config changes

# Import extracted services
from services.state_machine import CaptureState, CapturePhase, State, StateManager
from services.pipeline import InferenceModel, PipelinePhase, Pipeline, PipelineManager
from services.redis_service import RedisConnection
from services.db import (
    db_connection_pool, get_db_connection, release_db_connection,
    write_inference_to_db, write_production_metrics_to_db
)
from services.camera import (
    CameraBuffer, detect_video_devices, scan_network_for_cameras,
    scan_network_for_camera_devices, test_camera_stream, get_all_cameras,
    get_camera_config_for_save, apply_camera_config_from_saved,
    format_relative_time, DETECTED_CAMERAS,
    CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH,
    IP_CAMERA_USER, IP_CAMERA_PASS, IP_CAMERA_SUBNET,
    IP_CAMERA_BRIGHTNESS, IP_CAMERA_CONTRAST, IP_CAMERA_SATURATION
)
from services.watcher import ArduinoSocket, set_state_manager, set_app, _inference_queue
from services import detection
from services.detection import (
    add_detection_event, add_frame_to_timeline, evaluate_eject_from_detections,
    process_frame_helper, most_frequent_string, create_dynamic_images,
)

logger.info(f"Auto-detected cameras: {DETECTED_CAMERAS}")

# Global state manager instance (initialized after watcher)
state_manager: Optional[StateManager] = None

# Global pipeline manager instance (initialized at startup)
pipeline_manager: Optional[PipelineManager] = None

# =============================================================================

# Read version from VERSION file (single source of truth)
# Docker: mounted at /code/VERSION; local dev: ../VERSION relative to main.py
_code_dir = os.path.dirname(os.path.abspath(__file__))
for _vpath in [os.path.join(_code_dir, 'VERSION'), os.path.join(_code_dir, '..', 'VERSION')]:
    if os.path.exists(_vpath):
        _version_file = _vpath
        break
else:
    _version_file = None
try:
    with open(_version_file) as _vf:
        APP_VERSION = _vf.read().strip()
except Exception:
    APP_VERSION = "0.0.0"

# Initialize FastAPI app
app = FastAPI(
    title="MonitaQC Vision Engine",
    description="Industrial quality control and object detection platform",
    version=APP_VERSION
)

# Mount static files
from fastapi.staticfiles import StaticFiles
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static directory: {e}")

# Global reference to watcher instance (set after ArduinoSocket initialization)
watcher_instance = None

def apply_config_settings(config, watcher_inst=None, full_data=None):
    """Apply configuration settings from a config dict.

    This is a helper function used by startup loading, API endpoints, and config upload.

    Args:
        config: Service configuration dict
        watcher_inst: Watcher instance
        full_data: Full data file (including root-level keys like timeline_config)
    Returns tuple of (settings_applied: dict, cameras_loaded: int).
    """
    global EJECTOR_ENABLED, EJECTOR_OFFSET, EJECTOR_DURATION, EJECTOR_POLL_INTERVAL
    global CAPTURE_MODE, REMOVE_RAW_IMAGE_WHEN_DM_DECODED
    global PARENT_OBJECT_LIST, HISTOGRAM_ENABLED, HISTOGRAM_SAVE_IMAGE
    global CHECK_CLASS_COUNTS_ENABLED, CHECK_CLASS_COUNTS_CLASSES, CHECK_CLASS_COUNTS_CONFIDENCE
    global DM_CHARS_SIZES, DM_CONFIDENCE_THRESHOLD, DM_OVERLAP_THRESHOLD
    global STORE_ANNOTATION_ENABLED, ENFORCE_PARENT_OBJECT
    global YOLO_INFERENCE_URL, GRADIO_MODEL, GRADIO_CONFIDENCE_THRESHOLD
    global capture_mode, remove_raw_image_when_dm_decoded, parent_object_list
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
        capture_mode = CAPTURE_MODE
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

    # Apply timeline configuration (stored at root level, not in service_config)
    if full_data and "timeline_config" in full_data:
        app.state.timeline_config = full_data["timeline_config"]
        settings_applied["timeline_config"] = True
        logger.info(f"Timeline config loaded: {full_data['timeline_config']}")

        # Migration: convert legacy per-object eject to procedures
        if 'procedures' not in app.state.timeline_config:
            obj_filters = app.state.timeline_config.get('object_filters', {})
            legacy_rules = []
            for obj_name, obj_cfg in obj_filters.items():
                if obj_cfg.get('eject', False):
                    legacy_rules.append({
                        'object': obj_name,
                        'condition': obj_cfg.get('eject_condition', 'present'),
                        'min_confidence': int(obj_cfg.get('min_confidence', 0.01) * 100)
                    })
                obj_cfg.pop('eject', None)
                obj_cfg.pop('eject_condition', None)
            if legacy_rules:
                app.state.timeline_config['procedures'] = [{
                    'id': 'proc_legacy',
                    'name': 'Migrated Eject Rules',
                    'enabled': True,
                    'logic': 'any',
                    'rules': legacy_rules
                }]
                logger.info(f"Migrated {len(legacy_rules)} legacy eject rules to procedure")
            else:
                app.state.timeline_config['procedures'] = []
            full_data['timeline_config'] = app.state.timeline_config
            save_data_file(full_data)
            logger.info("Legacy eject config migrated to procedures format")
    else:
        # Initialize with defaults if not in config
        app.state.timeline_config = {
            'show_bounding_boxes': True,
            'camera_order': 'normal',
            'image_quality': 85,
            'num_rows': 20,
            'buffer_size': 100,
            'object_filters': {},
            'procedures': []
        }
        if full_data is None:
            logger.debug("Timeline config: full_data not provided, using defaults")

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

    # Sync all config changes back to the config module so other modules see them
    for _name in [
        'EJECTOR_ENABLED', 'EJECTOR_OFFSET', 'EJECTOR_DURATION', 'EJECTOR_POLL_INTERVAL',
        'CAPTURE_MODE', 'REMOVE_RAW_IMAGE_WHEN_DM_DECODED',
        'PARENT_OBJECT_LIST', 'HISTOGRAM_ENABLED', 'HISTOGRAM_SAVE_IMAGE',
        'CHECK_CLASS_COUNTS_ENABLED', 'CHECK_CLASS_COUNTS_CLASSES', 'CHECK_CLASS_COUNTS_CONFIDENCE',
        'DM_CHARS_SIZES', 'DM_CONFIDENCE_THRESHOLD', 'DM_OVERLAP_THRESHOLD',
        'STORE_ANNOTATION_ENABLED', 'ENFORCE_PARENT_OBJECT',
        'YOLO_INFERENCE_URL', 'GRADIO_MODEL', 'GRADIO_CONFIDENCE_THRESHOLD',
        'capture_mode', 'remove_raw_image_when_dm_decoded', 'parent_object_list',
        'SERIAL_MODE',
    ]:
        setattr(cfg_module, _name, globals()[_name])

    return settings_applied, cameras_loaded

def apply_saved_config_at_startup(watcher_inst):
    """Load and apply saved service configuration at startup."""
    try:
        # Load both service_config and root-level configs (like timeline_config)
        full_data = load_data_file()
        config = full_data.get("service_config", {})

        if not config:
            logger.info("No saved configuration found - using defaults")
            return False

        logger.info(f"Loading saved configuration from {DATA_FILE} (saved at: {config.get('saved_at', 'unknown')})")

        # Use the helper function to apply all settings (pass full_data for root-level configs)
        settings_applied, cameras_loaded = apply_config_settings(config, watcher_inst, full_data)

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


# =============================================================================
# ROUTERS
# =============================================================================
from routers.health import router as health_router
from routers.cameras import router as cameras_router
from routers.timeline import router as timeline_router
from routers.procedures import router as procedures_router
from routers.inference import router as inference_router
from routers.states import router as states_router
from routers.config_routes import router as config_router
from routers.ai import router as ai_router
from routers.commands import router as commands_router
from routers.websocket import router as ws_router

app.include_router(health_router)
app.include_router(cameras_router)
app.include_router(timeline_router)
app.include_router(procedures_router)
app.include_router(inference_router)
app.include_router(states_router)
app.include_router(config_router)
app.include_router(ai_router)
app.include_router(ws_router)
app.include_router(commands_router)  # MUST be last (catch-all /{command})


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
remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
parent_object_list = PARENT_OBJECT_LIST

try:
    os.mkdir("raw_images")
except:
    pass

# prepared data based on: https://docs.google.com/spreadsheets/d/1G1StYfEsSuQq9S6EWPO7WDpGGeqtRcZu6vN2-ixSqV8/edit?gid=1480612965#gid=1480612965&range=46:46
try:
    with open(DATA_FILE, "r") as file:
        data = json.load(file)
        if isinstance(data, list):
            prepared_query_data = data
        elif isinstance(data, dict) and "prepared_query_data" in data:
            prepared_query_data = data["prepared_query_data"]
        else:
            logger.info(f"No prepared_query_data found in {DATA_FILE}, using empty list (detection only mode)")
            prepared_query_data = []
except Exception as e:
    logger.error(f"Error loading data from {DATA_FILE}: {e}")
    prepared_query_data = []

# =============================================================================
# Startup
# =============================================================================

watcher = ArduinoSocket(
    camera_paths=DETECTED_CAMERAS if DETECTED_CAMERAS else None,
    serial_port=WATCHER_USB,
    serial_baudrate=SERIAL_BAUDRATE
)
# Set global reference for FastAPI endpoints
watcher_instance = watcher

# Initialize state manager with watcher
state_manager = StateManager(watcher)
set_state_manager(state_manager)

# Initialize pipeline manager for inference
pipeline_manager = PipelineManager()
logger.info(f"Pipeline manager initialized: current pipeline={pipeline_manager.current_pipeline.name if pipeline_manager.current_pipeline else 'none'}")

# Store in app.state for FastAPI endpoint access
app.state.watcher = watcher
app.state.watcher_instance = watcher  # Alias used by routers
app.state.cameras = watcher.cameras
app.state.redis_conn = watcher.redis_connection
app.state.state_manager = state_manager
app.state.pipeline_manager = pipeline_manager
app.state.apply_config_settings = apply_config_settings
app.state.apply_saved_config_at_startup = apply_saved_config_at_startup
# System capacity — detected once at startup
_cpu_logical = psutil.cpu_count(logical=True) or 1
_cpu_physical = psutil.cpu_count(logical=False) or 1
_mem = psutil.virtual_memory()
_mem_total_gb = round(_mem.total / (1024**3), 1)
_max_disk_writers = max(4, _cpu_logical)
_max_inference_workers = max(8, min(_cpu_logical * 2, 32))
app.state.system_capacity = {
    "cpu_logical": _cpu_logical,
    "cpu_physical": _cpu_physical,
    "ram_total_gb": _mem_total_gb,
    "max_disk_writers": _max_disk_writers,
    "max_inference_workers": _max_inference_workers,
}
logger.info(
    f"System capacity: {_cpu_physical} physical cores ({_cpu_logical} logical), "
    f"{_mem_total_gb}GB RAM | Max workers: {_max_disk_writers} disk, {_max_inference_workers} inference"
)

# Autoscaler status — published here, read by /api/inference/stats
app.state.autoscaler = {
    "disk_level": "OK",
    "inf_level": "OK",
    "disk_writers": 0,
    "inference_workers": 0,
    "disk_queue_pct": 0,
    "inf_queue_len": 0,
}
set_app(app)

# Initialize detection module with runtime references
detection.init(watcher, pipeline_manager, app, prepared_query_data)

# Load and apply saved configuration at startup
apply_saved_config_at_startup(watcher)

# Load saved states configuration
apply_states_from_config(state_manager)

# Load saved pipeline configuration
apply_pipeline_config_at_startup(pipeline_manager)

# Concurrent inference threads — each thread independently pops a frame from
# the hot queue (RAM) or cold queue (disk), calls YOLO, and handles results.
# Docker DNS round-robin distributes requests across YOLO replicas.
_num_cameras = len(watcher.cameras) if watcher.cameras else 1
INFERENCE_WORKERS = max(4, _num_cameras * 4)
# No semaphore — worker count IS the concurrency limit.
# Autoscaler adds workers when queue backs up, which naturally increases
# concurrency to match the API's capacity (local or remote).

# Initialize hot (RAM) + cold (disk) inference queues
from services.watcher import init_queues
init_queues(_num_cameras, cfg_module.EJECTOR_OFFSET, cfg_module.EJECTOR_ENABLED)

logger.info(f"Inference workers: {INFERENCE_WORKERS} threads for {_num_cameras} camera(s)")

# Update autoscaler initial state with actual values
from services.watcher import _disk_writers_count as _initial_disk
app.state.autoscaler["disk_writers"] = _initial_disk
app.state.autoscaler["inference_workers"] = INFERENCE_WORKERS

# Start web server in a separate thread
web_server_thread = threading.Thread(target=start_web_server, daemon=True)
web_server_thread.start()
logger.info(f"Web server started on http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")


# ── Shared frame processing logic ──
def _process_frame_batch(frames_data, allow_eject=True):
    """Process a frame batch through YOLO and handle results.

    Args:
        frames_data: dict with 'frames', 'encoder', 'shipment', 'capture_t'
        allow_eject: if True, evaluate ejector rules (hot path). If False, skip (cold path).

    Returns:
        True if YOLO produced results, False if inference failed.
    """
    st_ts = time.time()
    frames = frames_data.get("frames", [])
    capture_encoder = frames_data.get("encoder", 0)
    capture_shipment = frames_data.get("shipment", "no_shipment")
    capture_t = frames_data.get("capture_t", None)

    # Validate frame structure before processing
    valid_frames = []
    for frame in frames:
        if isinstance(frame, (list, tuple)) and len(frame) > 0:
            valid_frames.append(frame)

    if not valid_frames:
        return True  # Nothing to process, not a failure

    results = [process_frame_helper(f, capture_t=capture_t, encoder=capture_encoder) for f in valid_frames]

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        # Retry once
        results = [process_frame_helper(f, capture_t=capture_t, encoder=capture_encoder) for f in valid_frames]
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return False  # Inference failed

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

    # Log results
    for msg in queue_messages:
        ts = msg.get('ts', 'N/A')
        dms_list = msg.get('dms', [])
        first_dms = dms_list[0] if isinstance(dms_list, list) and len(dms_list) > 0 else None
        timestamp = time.time() - st_ts
        detection_info = msg.get('detection', [])
        if not isinstance(detection_info, list):
            detection_info = []
        classes = [d.get('name', '?') for d in detection_info if isinstance(d, dict)]
        logger.info(f"TS:{ts} | DM:{first_dms} | {','.join(classes)} | {timestamp:.2f}s")

    # Evaluate procedure-based eject rules (only for hot path)
    if allow_eject:
        try:
            _tl_cfg = getattr(app.state, 'timeline_config', {})
            _procedures = _tl_cfg.get('procedures', [])
            if _procedures and cfg_module.EJECTOR_ENABLED:
                # Collect all detections across all messages
                all_detections = []
                for msg in queue_messages:
                    det_list = msg.get('detection', [])
                    if isinstance(det_list, list):
                        all_detections.extend(d for d in det_list if isinstance(d, dict))

                should_eject, eject_reasons = evaluate_eject_from_detections(all_detections, _procedures)

                if should_eject:
                    eject_data = json.dumps({"encoder": capture_encoder, "dm": first_dms})
                    watcher.redis_connection.update_queue_messages_redis(eject_data, stream_name="ejector_queue")
                    watcher.eject_ng_counter += 1
                    logger.info(f"EJECT triggered: {' '.join(eject_reasons)} | encoder={capture_encoder}")
                else:
                    watcher.eject_ok_counter += 1
        except Exception as e:
            logger.warning(f"Procedure eject evaluation error: {e}")

    return True


# ── Inference worker thread: hot (RAM, LIFO) first, then cold (disk, FIFO) ──
def inference_worker_thread():
    """
    Two-phase inference worker:
    1. Try hot queue (RAM, LIFO) — for ejector decisions. On YOLO failure → fail-safe eject + spill to cold.
    2. If hot empty → try cold queue (disk, FIFO) — for historical processing, no ejector.
    No frame is ever dropped.
    """
    # Re-read module globals (set by init_queues before threads start)
    import services.watcher as _watcher_mod
    hot_q = _watcher_mod._hot_queue
    cold_q = _watcher_mod._cold_queue_disk

    logger.info("Inference worker thread started")

    while True:
        try:
            # Phase 1: Try hot queue (LIFO — newest first for ejector)
            try:
                frames_data = hot_q.get(timeout=0.05)
            except queue.Empty:
                frames_data = None

            if frames_data:
                try:
                    success = _process_frame_batch(frames_data, allow_eject=True)
                    if not success:
                        # YOLO failed after retry — fail-safe eject + spill to cold
                        if cfg_module.EJECTOR_ENABLED:
                            capture_encoder = frames_data.get("encoder", 0)
                            dms_list = frames_data.get("frames", [])
                            eject_data = json.dumps({"encoder": capture_encoder, "dm": None})
                            watcher.redis_connection.update_queue_messages_redis(
                                eject_data, stream_name="ejector_queue"
                            )
                            logger.warning(f"FAIL-SAFE EJECT: YOLO failed, ejecting encoder={capture_encoder}")
                        cold_q.put(frames_data)
                        logger.debug("Hot frame spilled to cold queue after YOLO failure")
                except Exception as e:
                    logger.error(f"Hot processing error: {e}")
                    # Don't lose the frame — spill to cold
                    try:
                        cold_q.put(frames_data)
                    except Exception:
                        pass
                continue

            # Phase 2: Hot queue empty — try cold queue (FIFO — oldest first)
            frames_data = cold_q.get()
            if frames_data:
                try:
                    _process_frame_batch(frames_data, allow_eject=False)
                except Exception as e:
                    logger.error(f"Cold processing error: {e}")
                continue

            # Both queues empty — brief sleep to avoid busy-wait
            time.sleep(0.05)

        except Exception as e:
            logger.error(f"Inference worker error: {e}")
            time.sleep(0.1)

# Start inference worker threads
for i in range(INFERENCE_WORKERS):
    t = threading.Thread(target=inference_worker_thread, daemon=True, name=f"inference-worker-{i+1}")
    t.start()
logger.info(f"{INFERENCE_WORKERS} inference worker thread(s) started")


# =============================================================================
# Autoscaler — monitors queue pressure every 5 min, scales workers dynamically
# =============================================================================
AUTOSCALE_INTERVAL = 30    # seconds between checks
MAX_DISK_WRITERS = _max_disk_writers
MAX_INFERENCE_WORKERS = _max_inference_workers

def _autoscaler():
    """Background thread: checks queue depth every 5 min and scales resources.

    Levels based on actual queue pressure:
        OK       — queues near-empty          → no change
        WARNING  — queues building up         → double current resources
        CRITICAL — queues overflowing         → quadruple current resources
    """
    global INFERENCE_WORKERS
    import services.watcher as _watcher_mod
    from services.watcher import _disk_queue, add_disk_writers

    # First check after 30s (let system warm up), then every AUTOSCALE_INTERVAL
    time.sleep(30)
    while True:
        try:
            # ── Measure disk queue pressure ──
            disk_qsize = _disk_queue.qsize()
            disk_pct = (disk_qsize / _disk_queue.maxsize * 100) if _disk_queue.maxsize else 0
            disk_count = _watcher_mod._disk_writers_count  # read live value from module

            # ── Measure inference queue pressure (hot=RAM, cold=disk) ──
            inf_hot = _watcher_mod._hot_queue.qsize() if _watcher_mod._hot_queue else 0
            inf_cold = _watcher_mod._cold_queue_disk.qsize() if _watcher_mod._cold_queue_disk else 0
            inf_qlen = inf_hot + inf_cold
            inf_qmax = _watcher_mod._hot_queue.maxsize if _watcher_mod._hot_queue else 0

            # ── Classify ──
            #   Disk:      >25% full → CRITICAL,  >5% → WARNING
            #   Inference: hot frames queuing means ejector can't keep up
            if disk_pct > 25:
                disk_level = "CRITICAL"
            elif disk_pct > 5:
                disk_level = "WARNING"
            else:
                disk_level = "OK"

            if inf_hot > 10 or inf_cold > 1000:
                inf_level = "CRITICAL"    # Hot backing up OR cold growing large
            elif inf_hot > 3 or inf_cold > 100:
                inf_level = "WARNING"
            else:
                inf_level = "OK"

            # ── Publish state so API/dashboard can read it ──
            app.state.autoscaler = {
                "disk_level": disk_level,
                "inf_level": inf_level,
                "disk_writers": disk_count,
                "inference_workers": INFERENCE_WORKERS,
                "disk_queue_pct": round(disk_pct, 1),
                "disk_queue_len": disk_qsize,
                "disk_queue_max": _disk_queue.maxsize,
                "inf_queue_len": inf_qlen,
                "inf_queue_max": inf_qmax,
                "inf_hot": inf_hot,
                "inf_cold": inf_cold,
            }

            logger.info(
                f"[Autoscaler] Disk: {disk_qsize}/{_disk_queue.maxsize} ({disk_pct:.0f}%) [{disk_level}] | "
                f"Inf: hot={inf_hot} cold={inf_cold} [{inf_level}] | "
                f"Workers: {disk_count} disk, {INFERENCE_WORKERS} inference"
            )

            # ── Scale disk writers ──
            if disk_level == "CRITICAL" and disk_count < MAX_DISK_WRITERS:
                # Quadruple: add 3× current (so total = 4× current)
                add = min(disk_count * 3, MAX_DISK_WRITERS - disk_count)
                if add > 0:
                    add_disk_writers(add)
                    logger.warning(f"[Autoscaler] CRITICAL disk — scaled {disk_count} -> {_watcher_mod._disk_writers_count}")
            elif disk_level == "WARNING" and disk_count < MAX_DISK_WRITERS:
                # Double: add current count (so total = 2× current)
                add = min(disk_count, MAX_DISK_WRITERS - disk_count)
                if add > 0:
                    add_disk_writers(add)
                    logger.warning(f"[Autoscaler] WARNING disk — scaled {disk_count} -> {_watcher_mod._disk_writers_count}")

            # ── Scale inference workers ──
            if inf_level == "CRITICAL":
                new_workers = min(INFERENCE_WORKERS * 4, MAX_INFERENCE_WORKERS)
            elif inf_level == "WARNING":
                new_workers = min(INFERENCE_WORKERS * 2, MAX_INFERENCE_WORKERS)
            else:
                new_workers = INFERENCE_WORKERS

            if new_workers > INFERENCE_WORKERS:
                old_count = INFERENCE_WORKERS

                # Spawn additional inference worker threads
                for i in range(old_count, new_workers):
                    t = threading.Thread(
                        target=inference_worker_thread, daemon=True,
                        name=f"inference-worker-{i+1}"
                    )
                    t.start()

                INFERENCE_WORKERS = new_workers
                logger.warning(
                    f"[Autoscaler] {'CRITICAL' if inf_level == 'CRITICAL' else 'WARNING'} inference — "
                    f"scaled {old_count} → {new_workers} workers"
                )

        except Exception as e:
            logger.error(f"[Autoscaler] Error: {e}")

        time.sleep(AUTOSCALE_INTERVAL)

threading.Thread(target=_autoscaler, daemon=True, name="autoscaler").start()
logger.info(f"Autoscaler started — first check in 30s, then every {AUTOSCALE_INTERVAL}s")


# Main thread - keep application alive
# All work is done in separate threads (uvicorn, inference, capture, etc.)
logger.info('All worker threads started - main thread entering keep-alive loop')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info('Shutting down...')
