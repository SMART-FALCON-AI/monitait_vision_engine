"""Detection, inference, and frame processing for MonitaQC Vision Engine.

Contains all detection functions extracted from main.py:
- req_predict: Unified YOLO/Gradio inference
- decode_dm: DataMatrix decoding with multiple preprocessing stages
- nest_objects: Object nesting based on spatial hierarchy
- decode_objects: Full object decoding pipeline
- process_frame: Complete frame processing pipeline
- check_class_counts: Object class count validation
- add_detection_event: Audio notification events via Redis
- add_frame_to_timeline: Timeline frame storage via Redis
"""

import cv2
import time
import json
import os
import numpy as np
import math
import requests
import logging
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pylibdmtx import pylibdmtx
from redis import Redis

import config as cfg
from services.db import write_inference_to_db

logger = logging.getLogger(__name__)

# Module-level runtime references (set by init() from main.py at startup)
_watcher = None           # ArduinoSocket instance
_pipeline_manager = None  # PipelineManager instance
_app = None               # FastAPI app instance (for app.state access)

# Prepared query data for DataMatrix validation
prepared_query_data = []

# HTTP session for inference requests
session = requests.Session()

# Reusable process pool for DataMatrix decoding (avoids fork overhead per call)
_dm_process_pool = None

def _get_dm_pool():
    """Lazy-init a reusable ProcessPoolExecutor for DataMatrix decoding."""
    global _dm_process_pool
    if _dm_process_pool is None:
        _dm_process_pool = ProcessPoolExecutor(max_workers=4)
    return _dm_process_pool

# Reusable Redis connection for timeline + timing (avoids new connection per call)
_timeline_redis = None

def _get_timeline_redis():
    """Lazy-init a reusable Redis connection for timeline and timing writes."""
    global _timeline_redis
    if _timeline_redis is None:
        _timeline_redis = Redis("redis", 6379, db=0)
    return _timeline_redis


def init(watcher, pipeline_manager, app, query_data):
    """Initialize detection module with runtime dependencies.

    Must be called from main.py after watcher/pipeline_manager/app are created.
    Sets module-level references that detection functions need at runtime.
    ProcessPoolExecutor forks after this, so child processes get snapshots of these values.
    """
    global _watcher, _pipeline_manager, _app, prepared_query_data
    _watcher = watcher
    _pipeline_manager = pipeline_manager
    _app = app
    prepared_query_data = query_data


def add_detection_event(event_type: str, details: dict = None):
    """Add a detection event for audio notification (cross-process via Redis)."""
    try:
        # Use global _watcher's redis connection
        if _watcher and _watcher.redis_connection and _watcher.redis_connection.redis_connection:
            event = {
                "type": event_type,  # "ok", "ng", "datamatrix", "object"
                "details": details or {},
                "timestamp": time.time()
            }
            # Add to Redis list (LPUSH adds to front)
            _watcher.redis_connection.redis_connection.lpush(cfg.DETECTION_EVENTS_REDIS_KEY, json.dumps(event))
            # Trim to max size
            _watcher.redis_connection.redis_connection.ltrim(cfg.DETECTION_EVENTS_REDIS_KEY, 0, cfg.DETECTION_EVENTS_MAX_SIZE - 1)
    except Exception as e:
        logger.error(f"Error adding detection event to Redis: {e}")


# ---------------------------------------------------------------------------
# Color Delta E helpers
# ---------------------------------------------------------------------------
_COLOR_REF_PREFIX = "color_ref:"
_COLOR_RUNNING_AVG_SIZE = 20  # Window size for running average


def extract_lab_color(image, detection):
    """Extract mean CIE L*a*b* color from a detection's bounding box.

    Uses center 50% of the crop to avoid edge artifacts.
    Normalizes OpenCV LAB (L 0-255, a/b 0-255) to standard CIE scale
    (L* 0-100, a*/b* centered at 0).

    Returns [L*, a*, b*] or None on failure.
    """
    try:
        h, w = image.shape[:2]
        x1 = max(0, int(detection.get('xmin', 0)))
        y1 = max(0, int(detection.get('ymin', 0)))
        x2 = min(w, int(detection.get('xmax', 0)))
        y2 = min(h, int(detection.get('ymax', 0)))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None

        crop = image[y1:y2, x1:x2]

        # Use center 50% to avoid edge artifacts
        ch, cw = crop.shape[:2]
        my, mx = ch // 4, cw // 4
        if my > 0 and mx > 0:
            crop = crop[my:ch - my, mx:cw - mx]

        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        mean_lab = lab_crop.mean(axis=(0, 1))

        L_star = round(float(mean_lab[0]) * 100.0 / 255.0, 2)
        a_star = round(float(mean_lab[1]) - 128.0, 2)
        b_star = round(float(mean_lab[2]) - 128.0, 2)
        return [L_star, a_star, b_star]
    except Exception as e:
        logger.debug(f"LAB extraction failed: {e}")
        return None


def update_color_references(detections):
    """Update Redis color references (previous + running_avg) after LAB extraction."""
    try:
        redis_client = _get_timeline_redis()
        best_per_class = {}
        for det in detections:
            if not isinstance(det, dict) or 'lab_color' not in det:
                continue
            name = det.get('name', '')
            if not name:
                continue
            conf = det.get('confidence', 0)
            if name not in best_per_class or conf > best_per_class[name]['confidence']:
                best_per_class[name] = det

        if not best_per_class:
            return

        pipe = redis_client.pipeline()
        for class_name, det in best_per_class.items():
            lab_json = json.dumps(det['lab_color'])
            pipe.set(f"{_COLOR_REF_PREFIX}{class_name}:previous", lab_json)
            running_key = f"{_COLOR_REF_PREFIX}{class_name}:running_avg_list"
            pipe.lpush(running_key, lab_json)
            pipe.ltrim(running_key, 0, _COLOR_RUNNING_AVG_SIZE - 1)
        pipe.execute()
    except Exception as e:
        logger.debug(f"Color reference update failed: {e}")


def get_color_reference(class_name, mode):
    """Retrieve a color reference [L*, a*, b*] from Redis, or None."""
    try:
        redis_client = _get_timeline_redis()
        if mode in ("fixed", "previous"):
            raw = redis_client.get(f"{_COLOR_REF_PREFIX}{class_name}:{mode}")
            return json.loads(raw) if raw else None
        elif mode == "running_avg":
            raw_list = redis_client.lrange(
                f"{_COLOR_REF_PREFIX}{class_name}:running_avg_list", 0, -1
            )
            if not raw_list:
                return None
            total = [0.0, 0.0, 0.0]
            count = 0
            for raw in raw_list:
                try:
                    lab = json.loads(raw)
                    total[0] += lab[0]; total[1] += lab[1]; total[2] += lab[2]
                    count += 1
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
            if count == 0:
                return None
            return [round(total[i] / count, 2) for i in range(3)]
        return None
    except Exception as e:
        logger.debug(f"Color reference read failed: {e}")
        return None


def set_fixed_color_reference(class_name, lab_color):
    """Set a fixed color reference (called from API)."""
    try:
        redis_client = _get_timeline_redis()
        redis_client.set(f"{_COLOR_REF_PREFIX}{class_name}:fixed", json.dumps(lab_color))
        return True
    except Exception as e:
        logger.error(f"Failed to set fixed color reference: {e}")
        return False


def _has_color_delta_procedure():
    """Check if any enabled procedure has a color_delta condition."""
    try:
        if _app is None:
            return False
        tl_cfg = getattr(_app.state, 'timeline_config', {})
        for proc in tl_cfg.get('procedures', []):
            if not proc.get('enabled', False):
                continue
            for rule in proc.get('rules', []):
                if rule.get('condition') == 'color_delta':
                    return True
        return False
    except Exception:
        return False


def evaluate_eject_from_detections(detections_list, procedures):
    """Evaluate eject rules from stored detections and procedures config.

    Supports 9 condition types:
        count_equals  — exactly N instances detected
        count_greater — more than N instances detected
        count_less    — fewer than N instances detected
        area_greater  — object bbox area > N pixels
        area_less     — object bbox area < N pixels
        area_equals   — object bbox area = N pixels
        color_delta   — color ΔE exceeds threshold vs reference
        present       — (legacy) count > 0
        not_present   — (legacy) count = 0

    Each procedure can optionally specify ``cameras`` (list of int camera IDs).
    When set, only detections from those cameras are evaluated.

    Args:
        detections_list: list of detection dicts (from one or more cameras)
        procedures: list of procedure configs from timeline_config

    Returns:
        (should_eject: bool, reasons: list[str])  — reasons are human-readable
    """
    should_eject = False
    eject_reasons = []

    if not detections_list:
        return False, []

    # --- Procedure rules ---
    for proc in (procedures or []):
        if not proc.get('enabled', False):
            continue
        rules = proc.get('rules', [])
        if not rules:
            continue
        logic = proc.get('logic', 'any')

        # Per-procedure camera filter
        proc_cameras = proc.get('cameras', [])
        if proc_cameras:
            proc_dets = [d for d in detections_list
                         if isinstance(d, dict) and d.get('_cam') in proc_cameras]
        else:
            proc_dets = detections_list  # all cameras

        def _eval_rule(rule, dets=proc_dets):
            obj = rule.get('object', '')
            cond = rule.get('condition', 'count_equals')
            min_conf = rule.get('min_confidence', 0) / 100.0
            expected_count = rule.get('count', 1)

            # Filtered detections for this object above confidence
            obj_dets = [d for d in dets
                        if isinstance(d, dict) and d.get('name') == obj
                        and d.get('confidence', 0) >= min_conf]
            actual = len(obj_dets)

            # Legacy conditions
            if cond == 'present':
                return actual > 0
            elif cond == 'not_present':
                return actual == 0
            # Count conditions
            elif cond == 'count_equals':
                return actual == expected_count
            elif cond == 'count_greater':
                return actual > expected_count
            elif cond == 'count_less':
                return actual < expected_count
            # Area conditions (compare bbox area in pixels)
            elif cond in ('area_greater', 'area_less', 'area_equals'):
                threshold = rule.get('area', 10000)
                if not obj_dets:
                    return False
                # Use highest-confidence detection's area
                best = max(obj_dets, key=lambda d: d.get('confidence', 0))
                w = int(best.get('xmax', 0)) - int(best.get('xmin', 0))
                h = int(best.get('ymax', 0)) - int(best.get('ymin', 0))
                if w <= 0 or h <= 0:
                    return False
                area = w * h
                best['_area'] = area
                if cond == 'area_greater':
                    return area > threshold
                elif cond == 'area_less':
                    return area < threshold
                elif cond == 'area_equals':
                    return area == threshold
            # Color Delta E
            elif cond == 'color_delta':
                max_de = rule.get('max_delta_e', 5.0)
                ref_mode = rule.get('reference_mode', 'previous')
                best_det = None
                best_conf = -1
                for det in obj_dets:
                    if 'lab_color' in det and det['confidence'] > best_conf:
                        best_det = det
                        best_conf = det['confidence']
                if best_det is None:
                    return False
                cur = best_det.get('lab_color')
                ref = get_color_reference(obj, ref_mode)
                if not cur or not ref or len(cur) != 3 or len(ref) != 3:
                    return False
                de = math.sqrt((cur[0]-ref[0])**2 + (cur[1]-ref[1])**2 + (cur[2]-ref[2])**2)
                best_det['_delta_e'] = round(de, 2)
                return de > max_de
            return False

        def _rule_detail(rule, triggered):
            """Build human-readable detail string for a triggered rule."""
            if not triggered:
                return None
            obj = rule.get('object', '?')
            cond = rule.get('condition', 'count_equals')

            if cond == 'color_delta':
                for det in proc_dets:
                    if isinstance(det, dict) and det.get('name') == obj and '_delta_e' in det:
                        return f"'{obj}' ΔE {det['_delta_e']} > {rule.get('max_delta_e', 5.0)}"
                return f"'{obj}' color ΔE exceeded"

            if cond in ('area_greater', 'area_less', 'area_equals'):
                for det in proc_dets:
                    if isinstance(det, dict) and det.get('name') == obj and '_area' in det:
                        op = {'area_greater': '>', 'area_less': '<', 'area_equals': '='}[cond]
                        return f"'{obj}' area {det['_area']}px {op} {rule.get('area', 10000)}"
                return f"'{obj}' area out of range"

            expected = rule.get('count', 1)
            min_conf = rule.get('min_confidence', 0) / 100.0
            actual = len([d for d in proc_dets
                         if isinstance(d, dict) and d.get('name') == obj
                         and d.get('confidence', 0) >= min_conf])
            op = {'count_equals': '=', 'count_greater': '>', 'count_less': '<',
                  'present': '>', 'not_present': '='}
            return f"'{obj}' count {actual} {op.get(cond, '?')} {expected}"

        results = [_eval_rule(r) for r in rules]
        triggered = all(results) if logic == 'all' else any(results)
        if triggered:
            should_eject = True
            proc_name = proc.get('name', 'Unnamed')
            details = [_rule_detail(r, res) for r, res in zip(rules, results) if res]
            details = [d for d in details if d]
            reason = f"{proc_name}: {', '.join(details)}" if details else proc_name
            eject_reasons.append(reason)

    return should_eject, eject_reasons


def add_frame_to_timeline(camera_id, frame, capture_t=None, detections=None, d_path=None, encoder=None):
    """Add a frame thumbnail to the timeline buffer (stored in Redis for cross-process access).

    Args:
        capture_t: Capture timestamp. If provided, used for timeline ordering instead of current time.
        detections: List of detection dicts (optional). Stored alongside frame for bbox rendering.
        d_path: Relative path to raw image (e.g. 'shipment/hour/timestamp_camid').
        encoder: Encoder position at time of capture.
    """
    if frame is None:
        logger.warning("add_frame_to_timeline: frame is None")
        return
    try:
        # Get configuration (with defaults)
        try:
            config = _app.state.timeline_config
            base_buffer_size = config.get('buffer_size', 100)  # User-configured buffer size
            quality = config.get('image_quality', 85)
        except Exception as e:
            logger.debug(f"Could not load timeline config, using defaults: {e}")
            base_buffer_size = 100
            quality = 85

        buffer_size = base_buffer_size

        # Resize to fixed thumbnail height (preserving aspect ratio)
        # Height is chosen to be readable on a QC monitor: 240px
        h, w = frame.shape[:2]
        thumb_h = 240
        if h > thumb_h:
            scale = thumb_h / h
            image_to_encode = cv2.resize(frame, (int(w * scale), thumb_h))
        else:
            image_to_encode = frame

        # Quality slider controls JPEG compression only (lower = smaller/faster)
        jpeg_q = max(30, min(95, quality))
        _, jpeg = cv2.imencode('.jpg', image_to_encode, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

        redis_client = _get_timeline_redis()

        # Store in Redis list (FIFO with max length)
        redis_key = f"{cfg.TIMELINE_REDIS_PREFIX}{camera_id}"
        ts = capture_t if capture_t else time.time()
        meta = {"d_path": d_path, "encoder": encoder}
        frame_data = pickle.dumps((ts, jpeg.tobytes(), detections, meta))

        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        pipe.rpush(redis_key, frame_data)  # Add to right (newest)
        pipe.ltrim(redis_key, -buffer_size, -1)  # Keep only last N frames (from config)
        pipe.execute()

        cfg.timeline_frame_counter += 1
        try:
            redis_client.publish("ws:timeline_update", str(camera_id))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Timeline add frame error: {e}", exc_info=True)

def req_predict(image):
    """
    Unified prediction function supporting both YOLO and Gradio APIs.
    Returns detections in standardized format with keys: x1, y1, x2, y2, confidence, class_id, name
    """
    start_time = time.time()

    # Check if we're using Gradio API (HuggingFace Space)
    if "hf.space" in cfg.YOLO_INFERENCE_URL or "huggingface" in cfg.YOLO_INFERENCE_URL:
        try:
            from gradio_client import Client, handle_file
            import tempfile

            # Initialize Gradio client (cached globally for reuse)
            if not hasattr(req_predict, 'gradio_client'):
                logger.info(f"Initializing Gradio client for {cfg.YOLO_INFERENCE_URL}")
                req_predict.gradio_client = Client(cfg.YOLO_INFERENCE_URL)
                logger.info("Gradio client initialized successfully")

            # Decode image bytes and save to temp file
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, img)
                tmp_path = tmp.name

            # Get model name and confidence from globals (loaded from config)
            model_name = cfg.GRADIO_MODEL
            confidence = cfg.GRADIO_CONFIDENCE_THRESHOLD

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
            except Exception as e:
                logger.debug(f"Could not delete temp file {tmp_path}: {e}")

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
            cfg.latest_detections = normalized_detections
            cfg.latest_detections_timestamp = time.time()

            # Also store in Redis for cross-process access
            if _watcher and _watcher.redis_connection:
                try:
                    _watcher.redis_connection.redis_connection.setex(
                        "gradio_last_detection_timestamp",
                        60,  # Expire after 60 seconds
                        str(cfg.latest_detections_timestamp)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache detection timestamp in Redis: {e}")

            logger.info(f"Cached {len(normalized_detections)} detections at timestamp {cfg.latest_detections_timestamp}")

            # Note: detection events and Redis timing are handled by process_frame()
            # to avoid duplicate writes when called from the inference pipeline.

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
            response = requests.post(cfg.YOLO_INFERENCE_URL, files={"image": image}, headers=headers, timeout=10)
            response.raise_for_status()

            # Note: Redis timing is tracked by process_frame() to avoid duplicate writes
            return response.text  # Return JSON string, not parsed dict
        except requests.exceptions.RequestException as e:
            logger.error(f"YOLO prediction failed: {e}")
            return "[]"  # Return empty JSON array string

def create_dynamic_images(frame_height):
    h = frame_height # Extract height from the frame's image (frame[1] contains the image)
    blank_image = np.zeros((h, 1, 3), dtype=np.uint8)  # (height, width, channels)
    
    # Create green and red 7px width images with height from the frame
    green_image = np.zeros((h, 7, 3), dtype=np.uint8)  # (height, width, channels)
    green_image[:, :, 1] = 255  # Set green channel to full (R=0, G=255, B=0)

    red_image = np.zeros((h, 7, 3), dtype=np.uint8)
    red_image[:, :, 2] = 255  # Set red channel to full (R=255, G=0, B=0)

    return blank_image, green_image, red_image

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

        # Start decoding (uses shared process pool — no fork overhead per call)
        pool = _get_dm_pool()
        futures = [
            pool.submit(preprocess_and_decode, stage_name, preprocess, cropped_dm_gray, dm_chars_sizes)
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
                                   If None, uses global cfg.PARENT_OBJECT_LIST.

    Returns:
        dict: Nested structure of objects with parents ("box", "pack") and their children.
               If no parents found, all objects are nested under a single virtual "_root" parent.
    """
    try:
        # Use global cfg.PARENT_OBJECT_LIST if no custom list provided
        parent_list = custom_parent_list if custom_parent_list is not None else cfg.PARENT_OBJECT_LIST
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

def check_class_counts(yolo_results, confidence_threshold=None):
    """
    Check if the number of specific classes in YOLO results are all equal.
    Uses cfg.CHECK_CLASS_COUNTS_CLASSES from config.
    Only counts detections above the confidence threshold.
    Returns True if all specified class counts are equal, False otherwise.
    """
    target_classes = cfg.CHECK_CLASS_COUNTS_CLASSES
    conf_threshold = confidence_threshold if confidence_threshold is not None else cfg.CHECK_CLASS_COUNTS_CONFIDENCE

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

def process_frame(frame, capture_mode, capture_t=None, encoder=None):
    try:

        frame_id, jpeg_bytes = frame
        frame_path = os.path.join("raw_images", f"{frame_id}.jpg")

        if jpeg_bytes is not None:
            # In-memory path: decode JPEG bytes directly (no disk I/O)
            img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_bytes = jpeg_bytes  # Already encoded — skip cv2.imencode entirely
        else:
            # Fallback: legacy disk path (backward compatibility / retry)
            for _wait in range(100):  # Up to 1 second
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    break
                time.sleep(0.01)
            image = cv2.imread(frame_path)
            img_encoded = cv2.imencode('.jpg', image)[1]
            img_bytes = img_encoded.tobytes()

        if image is None:
            logger.error(f"Failed to decode image for {frame_id}")
            return None

        # Use pipeline manager if available, otherwise fallback to legacy req_predict
        start_time = time.time()
        model_name_used = "unknown"

        if _pipeline_manager:
            yolo_res, model_name_used = _pipeline_manager.run_inference(img_bytes)
        else:
            # Fallback to legacy inference
            yolo_res = json.loads(req_predict(img_bytes))
            model_name_used = "legacy"

        processing_time = time.time() - start_time
        elapsed_ms = processing_time * 1000

        # Track inference timing in Redis (cross-process safe)
        try:
            redis_client = _get_timeline_redis()
            redis_client.lpush("inference_times", str(elapsed_ms))
            redis_client.ltrim("inference_times", 0, 9)  # Keep last 10

            # Atomic counter: total inference frames processed (all workers)
            now = time.time()
            redis_client.incr("inf_frame_count")
            redis_client.lpush("inf_frame_timestamps", str(now))
            redis_client.ltrim("inf_frame_timestamps", 0, 1999)  # Keep last 2000

            # Legacy: single-worker frame interval (kept for backward compat)
            last_ts_raw = redis_client.get("last_inference_timestamp")
            if last_ts_raw:
                last_ts = float(last_ts_raw.decode('utf-8'))
                interval_ms = (now - last_ts) * 1000
                redis_client.lpush("frame_intervals", str(interval_ms))
                redis_client.ltrim("frame_intervals", 0, 9)
            redis_client.set("last_inference_timestamp", str(now))
        except Exception as e:
            logger.debug(f"Failed to track inference timing in Redis: {e}")

        # Cache frame_id for video feed to display processed image
        if _watcher:
            _watcher.latest_frame_id = frame_id

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

            # Attach camera ID to each detection for per-camera procedure filtering
            try:
                cam_id = int(frame_id.rsplit('_', 1)[-1])
            except (ValueError, IndexError):
                cam_id = 0
            for det in yolo_res:
                det['_cam'] = cam_id

            # Extract L*a*b* color only if a color_delta procedure exists
            if _has_color_delta_procedure():
                for det in yolo_res:
                    lab = extract_lab_color(image, det)
                    if lab is not None:
                        det['lab_color'] = lab
                update_color_references(yolo_res)

            # Add audio notification with detected object names
            try:
                if yolo_res:
                    logger.info(f"Processing {len(yolo_res)} detections for audio notification")
                    # Extract class names from detections (use 'name' if available, fallback to 'Class X')
                    detected_names = [det.get('name', f"Class {det.get('class', 'Unknown')}") for det in yolo_res]
                    logger.info(f"Detected names: {detected_names}")

                    # Count occurrences of each class name
                    from collections import Counter
                    name_counts = Counter(detected_names)

                    # Create announcement text
                    objects_text = ", ".join([f"{count} {name}" for name, count in name_counts.items()])
                    logger.info(f"Audio notification text: {objects_text}")

                    # Format detections with class_name field for frontend
                    formatted_detections = [
                        {
                            "class_name": det.get('name', f"Class {det.get('class', 'Unknown')}"),
                            "confidence": det.get('confidence', 0),
                            "bbox": det.get('box', [])
                        }
                        for det in yolo_res
                    ]

                    add_detection_event("object", {
                        "objects": objects_text,
                        "count": len(yolo_res),
                        "frame_id": frame_id,
                        "detections": formatted_detections  # Add formatted detections array
                    })
                    logger.info(f"Detection event added successfully with {len(formatted_detections)} detections")
            except Exception as e:
                logger.error(f"Error creating detection event: {e}", exc_info=True)

            # Write inference result to database with model name
            try:
                inference_time_ms = int(processing_time * 1000)
                # Get shipment from Redis (works across processes, unlike watcher object)
                try:
                    shipment_val = _get_timeline_redis().get("shipment")
                    shipment_str = shipment_val.decode('utf-8') if isinstance(shipment_val, bytes) and shipment_val else "no_shipment"
                except Exception:
                    shipment_str = "no_shipment"
                write_inference_to_db(
                    shipment=shipment_str,
                    image_path=annotated_path,
                    detections=yolo_res,
                    inference_time_ms=inference_time_ms,
                    model_used=model_name_used
                )
            except Exception as e:
                logger.error(f"Failed to write inference to DB: {e}")

        # Push inferenced frame to timeline with detection data for bbox overlay
        try:
            cam_id = int(frame_id.rsplit('_', 1)[-1])
            add_frame_to_timeline(cam_id, image, capture_t=capture_t, detections=yolo_res if yolo_res else None, d_path=frame_id, encoder=encoder)
        except Exception as e:
            logger.debug(f"Timeline push after inference failed: {e}")

        frame_data = [frame_id, image, yolo_res, 5]

        if yolo_res:
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
            "shipment": _watcher.shipment
        }

        return queue_message, frame_data

    except Exception as e:
        logger.error(f"Error processing frame {frame_id}: {e}")
        return None

def most_frequent_string(dms_list):
    """Find the most frequent string in the list of 'dms[0]'."""
    counter = Counter(dms_list)
    return counter.most_common(1)[0][0] if counter else None

def process_frame_helper(frame, capture_t=None, encoder=None):
    capture_mode = "single"  # or "multiple"
    return process_frame(frame, capture_mode, capture_t=capture_t, encoder=encoder)

