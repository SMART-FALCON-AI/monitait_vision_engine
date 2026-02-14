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

def add_frame_to_timeline(camera_id, frame):
    """Add a frame thumbnail to the timeline buffer (stored in Redis for cross-process access)."""
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

        # Dynamically adjust max buffer size based on quality to manage memory
        # Higher quality = lower max buffer, lower quality = higher max buffer
        if quality >= 95:
            # Very high quality (95-100%): cap at 100 frames to save memory
            max_buffer_size = 100
        elif quality >= 85:
            # High quality (85-94%): cap at 300 frames
            max_buffer_size = 300
        elif quality >= 70:
            # Medium quality (70-84%): cap at 500 frames (baseline)
            max_buffer_size = 500
        else:
            # Low quality (<70%): allow larger buffer (1000 frames)
            max_buffer_size = 1000

        # Use the smaller of user's configured size or quality-based limit
        buffer_size = min(base_buffer_size, max_buffer_size)

        # Scale image based on quality percentage
        # 100% = original size, lower % = smaller thumbnail
        if quality == 100:
            # Keep original full-resolution image
            image_to_encode = frame
        else:
            # Calculate scale factor based on quality percentage
            # At 85%, use standard thumbnail size (120x90)
            # Scale proportionally for other quality levels
            scale_factor = quality / 85.0  # 85% is baseline for standard thumbnail
            target_width = int(cfg.TIMELINE_THUMBNAIL_WIDTH * scale_factor)
            target_height = int(cfg.TIMELINE_THUMBNAIL_HEIGHT * scale_factor)

            # Ensure minimum size of at least the standard thumbnail
            target_width = max(target_width, cfg.TIMELINE_THUMBNAIL_WIDTH)
            target_height = max(target_height, cfg.TIMELINE_THUMBNAIL_HEIGHT)

            # Resize to calculated dimensions
            image_to_encode = cv2.resize(frame, (target_width, target_height))

        # Encode as JPEG bytes with configured quality
        _, jpeg = cv2.imencode('.jpg', image_to_encode, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Create Redis connection (each process needs its own connection)
        redis_client = Redis("redis", 6379, db=0)

        # Store in Redis list (FIFO with max length)
        redis_key = f"{cfg.TIMELINE_REDIS_PREFIX}{camera_id}"
        frame_data = pickle.dumps((time.time(), jpeg.tobytes()))

        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        pipe.rpush(redis_key, frame_data)  # Add to right (newest)
        pipe.ltrim(redis_key, -buffer_size, -1)  # Keep only last N frames (from config)
        pipe.execute()

        cfg.timeline_frame_counter += 1
        if buffer_size < base_buffer_size:
            logger.info(f"Timeline: Added frame for camera {camera_id} to Redis (keeping last {buffer_size} frames, capped from {base_buffer_size} due to quality={quality}%)")
        else:
            logger.info(f"Timeline: Added frame for camera {camera_id} to Redis (keeping last {buffer_size} frames)")
    except Exception as e:
        logger.error(f"Timeline add frame error: {e}", exc_info=True)

def req_predict(image):
    """
    Unified prediction function supporting both YOLO and Gradio APIs.
    Returns detections in standardized format with keys: x1, y1, x2, y2, confidence, class_id, name
    """
    start_time = time.time()

    # Track frame-to-frame interval
    if cfg.last_inference_timestamp > 0:
        interval = (start_time - cfg.last_inference_timestamp) * 1000  # Convert to ms
        cfg.frame_intervals.append(interval)
        if len(cfg.frame_intervals) > cfg.max_inference_samples:
            cfg.frame_intervals.pop(0)

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

            # Add detection event for audio notification
            if normalized_detections:
                # Extract detection types for audio
                detection_names = [d.get('name', 'unknown') for d in normalized_detections]
                has_datamatrix = any('matrix' in n.lower() or 'dm' in n.lower() for n in detection_names)
                add_detection_event(
                    "datamatrix" if has_datamatrix else "object",
                    {"count": len(normalized_detections), "classes": list(set(detection_names))}
                )

            # Track inference time and update timestamp
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            cfg.inference_times.append(elapsed)
            if len(cfg.inference_times) > cfg.max_inference_samples:
                cfg.inference_times.pop(0)
            cfg.last_inference_timestamp = time.time()

            # Store timing in Redis for cross-process access
            try:
                if _watcher and _watcher.redis_connection:
                    _watcher.redis_connection.redis_connection.lpush("inference_times", str(elapsed))
                    _watcher.redis_connection.redis_connection.ltrim("inference_times", 0, cfg.max_inference_samples - 1)
                    if cfg.last_inference_timestamp > 0:
                        interval_ms = (time.time() - cfg.last_inference_timestamp) * 1000
                        _watcher.redis_connection.redis_connection.lpush("frame_intervals", str(interval_ms))
                        _watcher.redis_connection.redis_connection.ltrim("frame_intervals", 0, cfg.max_inference_samples - 1)
                    _watcher.redis_connection.redis_connection.set("last_inference_timestamp", str(time.time()))
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
            response = requests.post(cfg.YOLO_INFERENCE_URL, files={"image": image}, headers=headers, timeout=10)
            response.raise_for_status()

            # Track inference time and update timestamp
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            cfg.inference_times.append(elapsed)
            if len(cfg.inference_times) > cfg.max_inference_samples:
                cfg.inference_times.pop(0)

            # Store timing in Redis for cross-process access
            try:
                if _watcher and _watcher.redis_connection:
                    _watcher.redis_connection.redis_connection.lpush("inference_times", str(elapsed))
                    _watcher.redis_connection.redis_connection.ltrim("inference_times", 0, cfg.max_inference_samples - 1)

                    # Calculate and store frame interval
                    last_ts_str = _watcher.redis_connection.redis_connection.get("last_inference_timestamp")
                    if last_ts_str:
                        last_ts = float(last_ts_str.decode('utf-8'))
                        interval_ms = (time.time() - last_ts) * 1000
                        _watcher.redis_connection.redis_connection.lpush("frame_intervals", str(interval_ms))
                        _watcher.redis_connection.redis_connection.ltrim("frame_intervals", 0, cfg.max_inference_samples - 1)

                    _watcher.redis_connection.redis_connection.set("last_inference_timestamp", str(time.time()))
            except Exception as e:
                logger.warning(f"Failed to store timing in Redis: {e}")

            cfg.last_inference_timestamp = time.time()
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

def process_frame(frame, capture_mode):
    try:

        frame_id , _ = frame
        frame_path = os.path.join("raw_images", f"{frame_id}.jpg")

        # Wait for file to exist (async disk writes may not have completed yet)
        for _wait in range(100):  # Up to 1 second
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                break
            time.sleep(0.01)

        image = cv2.imread(frame_path)

        img_encoded = cv2.imencode('.jpg', image)[1]
        img_bytes = img_encoded.tobytes()

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
            redis_client = Redis("redis", 6379, db=0)
            redis_client.lpush("inference_times", str(elapsed_ms))
            redis_client.ltrim("inference_times", 0, 9)  # Keep last 10

            # Frame interval = time since last inference completion (measures throughput)
            last_ts_raw = redis_client.get("last_inference_timestamp")
            if last_ts_raw:
                last_ts = float(last_ts_raw.decode('utf-8'))
                interval_ms = (time.time() - last_ts) * 1000
                redis_client.lpush("frame_intervals", str(interval_ms))
                redis_client.ltrim("frame_intervals", 0, 9)
            redis_client.set("last_inference_timestamp", str(time.time()))
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
                    r = Redis("redis", 6379, db=0)
                    shipment_val = r.get("shipment")
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

        # Add frame to timeline regardless of detections (use annotated if available, otherwise original)
        try:
            camera_id = int(frame_id.split('_')[-1])
            timeline_img = annotated_image if (yolo_res and len(yolo_res) > 0) else image
            add_frame_to_timeline(camera_id, timeline_img)
        except Exception as e:
            logger.error(f"Could not add frame to timeline (frame_id={frame_id}): {e}", exc_info=True)

        frame_data = [frame_id, image, yolo_res, 5]

        if yolo_res:
            # Check if specific class counts are equal (only if enabled)
            if cfg.CHECK_CLASS_COUNTS_ENABLED and not check_class_counts(yolo_res):
                frame_data.append(None)
                frame_data[1] = None  # make the frame light by deleting image part
                queue_message = {
                    "ts": frame_data[0],
                    "dms": frame_data[4:],
                    "priority": frame_data[3],
                    "detection": frame_data[2],
                    "shipment": _watcher.shipment
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

def process_frame_helper(frame):
    capture_mode = "single"  # or "multiple"
    return process_frame(frame, capture_mode)

